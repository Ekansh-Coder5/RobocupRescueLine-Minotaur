"""Everything for the line part of the course: the per-frame line follower
(LineFollower), gap handling, red stop detection, green marker turns, finding /
aligning to the line, and obstacle avoidance."""
import time
import numpy as np
import cv2

from config import *
from robot import (
    get_sensor_data, set_status, fresh_frame, drive,
    timed_move, timed_motion, turn_with_gyro, obstacle_gyro_turn,
)
from vision import (
    quit_requested, live_preview, show_preview, put_label,
    get_contour_pose, black_mask_rgb, green_mask_hsv, green_check, detect_red,
)

# What LineFollower.step() reports back to main.py
EVENT_NONE = None
EVENT_QUIT = "quit"      # 'q' pressed in a preview window
EVENT_RED = "red"        # red seen for RED_CONFIRM_FRAMES frames in a row


# ---- Obstacle bypass side ----
# 1 = go around the obstacle on the RIGHT, 0 = go around it on the LEFT.
# Keep this setting here only (not in config.py): this line runs after
# `from config import *`, so it would overwrite a value set there.
OBS_RIGHT = 0

# Every obstacle turn angle is multiplied by this, so 0 mirrors the whole bypass.
OBS_SIGN = 1 if OBS_RIGHT == 1 else -1


# ================================================================ LINE SEARCH / ALIGN
def find_line_after_turn(motors, camera, direction="right"):
    """
    After a gyro turn, keep turning in the SAME direction until a black line is found.
    direction: "right" or "left" - the direction we were already turning
    Returns True if line found, False otherwise.
    The preview is refreshed on every frame while searching.
    """
    print(f"Searching for black line after gyro turn (continuing {direction})...")
    max_search_time = 5.0
    turn_speed = 20  # Continue turning at reduced speed

    # Set continuous turning in the same direction
    if direction == "right":
        left_motor = turn_speed
        right_motor = -turn_speed
    else:  # left
        left_motor = -turn_speed
        right_motor = turn_speed

    t0 = time.time()
    line_detected = False

    try:
        while time.time() - t0 < max_search_time:
            img = camera.get_camera_view()
            if img is None:
                time.sleep(0.05)
                continue

            live_preview(img, "FIND LINE")

            full_line_mask = black_mask_rgb(img)

            contours, _ = cv2.findContours(full_line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pose = get_contour_pose(contours)

            W_img = img.shape[1]
            if pose is not None and abs(pose[0] - W_img / 2) <= FIND_CENTER_TOLERANCE * W_img:
                cx, cy, angle_deg, vx, vy = pose
                print(f"Line centered! cx={cx}")
                print("✓ Line detected - resuming normal line following")
                line_detected = True
                break
            else:
                # No centered line yet, keep rotating in the same direction
                motors.set_motors(left_motor, right_motor)

            time.sleep(0.05)
    finally:
        motors.stop_motors()

    return line_detected


def align_to_line(motors, camera):
    """Pivot in place until the black line (below the junction) is roughly vertical.
    angle_deg > 0 means the line leans right -> pivot right; < 0 -> pivot left.
    Skips alignment if no line is seen or the tilt is larger than ALIGN_MAX_DEG.
    The preview is refreshed on every frame while aligning."""
    W, H = FRAME_SIZE
    y0 = int(H * ALIGN_ROI_TOP)
    t0 = time.time()
    try:
        while time.time() - t0 < ALIGN_TIMEOUT:
            img = camera.get_camera_view()
            if img is None:
                time.sleep(0.02)
                continue

            live_preview(img, "ALIGN")

            green_raw = green_mask_hsv(img)
            mask = black_mask_rgb(img)
            mask[green_raw > 0] = 0
            mask[:y0, :] = 0   # bottom part only, avoids the junction's branch lines
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pose = get_contour_pose(contours)
            if pose is None:
                print("Align: no line seen, skipping")
                break

            angle_deg = pose[2]
            print(f"Align: line angle = {angle_deg:+.1f} deg")

            if abs(angle_deg) <= ALIGN_TOLERANCE_DEG:
                print("Align: done")
                break
            if abs(angle_deg) > ALIGN_MAX_DEG:
                print("Align: tilt too large, skipping")
                break

            if angle_deg > 0:
                motors.set_motors(ALIGN_SPEED, -ALIGN_SPEED)   # pivot right
            else:
                motors.set_motors(-ALIGN_SPEED, ALIGN_SPEED)   # pivot left
            time.sleep(0.03)
    finally:
        motors.stop_motors()


# ================================================================ GREEN HANDLING
def do_uturn(motors, camera):
    timed_move(motors, UTURN_REVERSE_SPEED, UTURN_REVERSE_TIME)
    timed_move(motors, GREEN_FORWARD_SPEED, UTURN_FORWARD_TIME)
    turn_with_gyro(motors, angle_deg=UTURN_DEG, direction="left", speed=GYRO_TURN_SPEED,
                   fallback_time=HARDCODED_UTURN_TIME)
    time.sleep(0.5)
    find_line_after_turn(motors, camera, direction="left")


def do_side_turn(motors, camera, direction):
    timed_move(motors, GREEN_FORWARD_SPEED, LR_FORWARD_TIME)
    if direction == "left":
        angle, fallback_time = LEFT_TURN_DEG, HARDCODED_LEFT_TURN_TIME
    else:
        angle, fallback_time = RIGHT_TURN_DEG, HARDCODED_RIGHT_TURN_TIME
    turn_with_gyro(motors, angle_deg=angle, direction=direction, speed=GYRO_TURN_SPEED,
                   fallback_time=fallback_time)
    time.sleep(0.5)
    motors.set_motors(BACKUP_SPEED, BACKUP_SPEED)
    time.sleep(BACKUP_TIME)
    motors.stop_motors()
    find_line_after_turn(motors, camera, direction=direction)


def handle_green(motors, camera, gleft, gright):
    """The trigger in main() (bottom 50% of the frame) only STARTS this routine.
    From here on the trigger result is ignored: the decision (left / right / U-turn)
    comes ONLY from a fresh FULL-frame green check after aligning. If that check
    finds no valid green, nothing is done and the robot goes back to line following.
    Camera frames and the preview are refreshed during the pause, align and search.
    Returns True if the user pressed 'q' in the preview."""
    motors.stop_motors()
    set_status("GREEN FOUND")
    print(f"Green trigger (bottom half): left={gleft}, right={gright} "
          f"-> pause, align, then decide ONLY from the full-frame check")

    # ---- pause, with live frames so the preview does not freeze ----
    t0 = time.time()
    while time.time() - t0 < GREEN_PAUSE:
        frame = camera.get_camera_view()
        if frame is not None and live_preview(frame, "GREEN PAUSE"):
            return True
        time.sleep(0.02)

    # ---- align (refreshes frames internally) ----
    set_status("GREEN ALIGN")
    align_to_line(motors, camera)
    if quit_requested():
        return True

    # ---- full-frame decision on a fresh frame ----
    set_status("GREEN CHECK")
    frame = fresh_frame(camera)
    if frame is None:
        print("Green: no camera frame, going back to line following")
        set_status("LFR")
        return quit_requested()

    left, right, green_mask, black_vis = green_check(frame, roi_top=0.0)
    print(f"Full-frame check: left={left}, right={right}")

    if SHOW_PREVIEW:
        if show_preview(frame, np.zeros_like(frame), green_mask, black_vis, 0, 0):
            return True

    if left and right:
        set_status("GREEN U-TURN")
        do_uturn(motors, camera)
    elif left:
        set_status("GREEN DOT LEFT")
        do_side_turn(motors, camera, "left")
    elif right:
        set_status("GREEN DOT RIGHT")
        do_side_turn(motors, camera, "right")
    else:
        # Nothing valid in the full frame: do nothing and go back to line following
        print("Green: nothing valid in full frame, going back to line following")
        set_status("LFR")

    return quit_requested()


# ================================================================ OBSTACLE AVOIDANCE
def obstacle_status(text):
    """Report obstacle progress on both the terminal and the OLED status line."""
    print(f"[OBSTACLE] {text}")
    set_status(f"OBS: {text}")


def switch_turned_off(motors):
    """True if the motor switch is off. Obstacle steps call this so the sequence
    stops as soon as the switch is turned off."""
    if motors.motor_switch.is_pressed:
        return False
    obstacle_status("SWITCH OFF")
    motors.stop_motors()
    return True


def obstacle_turn_signed(motors, angle):
    """Gyro-turn through a SIGNED angle.
    Positive = set_motors(+speed, -speed), which turns the robot physically RIGHT
    (obstacle_gyro_turn() calls this direction "left").
    Negative = set_motors(-speed, +speed), physically LEFT.
    obstacle_gyro_turn() applies the sign itself from its direction word, so it
    must get abs(angle): a negative angle there would end the turn instantly."""
    direction = "left" if angle > 0 else "right"
    return obstacle_gyro_turn(motors, direction, angle=abs(angle))


def confirm_obstacle(motors):
    """Double-check a detection: creep forward a little, then re-read the ToF on a
    FRESH packet. Returns True only if the obstacle is still inside the threshold."""
    obstacle_status("CONFIRM FWD")
    timed_motion(
        motors,
        OBSTACLE_CONFIRM_SPEED, OBSTACLE_CONFIRM_SPEED,
        OBSTACLE_CONFIRM_SECONDS,
        f"Creeping forward for {OBSTACLE_CONFIRM_SECONDS:.2f}s to double-check"
    )
    time.sleep(OBSTACLE_CONFIRM_SETTLE)
    if switch_turned_off(motors):
        return False

    # Only trust a packet that arrives AFTER the creep + settle.
    start_counter = get_sensor_data()["counter"]
    deadline = time.monotonic() + OBSTACLE_CONFIRM_WAIT

    while time.monotonic() < deadline:
        data = get_sensor_data()
        if not data["connected"]:
            break

        if data["counter"] != start_counter:
            confirmed = data["distance_valid"] and data["distance_mm"] < OBSTACLE_THRESHOLD_MM
            if data["distance_valid"]:
                print(f"[OBSTACLE] Re-check: {data['distance_mm']:.0f} mm "
                      f"-> {'CONFIRMED' if confirmed else 'false alarm'}")
            else:
                print("[OBSTACLE] Re-check: distance invalid -> treated as false alarm")
            return confirmed

        time.sleep(0.01)

    print("[OBSTACLE] Re-check: no fresh ESP32 reading -> treated as false alarm")
    return False


def drive_until_line(motors, camera, speed, timeout):
    """Drive straight until at least OBSTACLE_LINE_MIN_PIXELS black pixels appear
    below OBSTACLE_LINE_ROI_TOP in the frame. Returns True if the line was seen,
    False if timeout ran out first. The motors are stopped either way."""
    H = FRAME_SIZE[1]
    y0 = int(H * OBSTACLE_LINE_ROI_TOP)
    t0 = time.time()
    try:
        motors.set_motors(speed, speed)
        while time.time() - t0 < timeout:
            img = camera.get_camera_view()
            if img is None:
                time.sleep(0.02)
                continue

            live_preview(img, "SEEK LINE")

            mask = black_mask_rgb(img)
            mask[:y0, :] = 0
            black_px = cv2.countNonZero(mask)
            if black_px >= OBSTACLE_LINE_MIN_PIXELS:
                print(f"[OBSTACLE] Line seen ({black_px} px) after {time.time() - t0:.2f}s")
                return True

            time.sleep(0.02)

        print(f"[OBSTACLE] No line after {timeout:.1f}s, turning anyway")
        return False
    finally:
        motors.stop_motors()
        time.sleep(0.15)


def avoid_obstacle(motors, camera):
    """Run the obstacle-avoidance sequence around the side set by OBS_RIGHT.
    The last step keeps turning until the black line is found. Returns True if the
    full sequence completed, False if a turn failed, the line was not found, or the
    motor switch was turned off."""

    # Signed turn angles (positive = physically right, see obstacle_turn_signed()).
    # OBS_SIGN = -1 negates all of them, mirroring the bypass to the other side.
    first_turn = 45 * OBS_SIGN     # away from the line, around the obstacle
    back_turn = -45 * OBS_SIGN     # used twice, back toward the line
    # The final line search pivots the same physical way as the first turn.
    # find_line_after_turn() calls set_motors(+speed, -speed) "right".
    find_direction = "right" if first_turn > 0 else "left"

    obstacle_status(f"AVOIDING ({'RIGHT' if OBS_SIGN > 0 else 'LEFT'})")

    motors.stop_motors()
    time.sleep(0.2)

    # 1. Reverse slightly to clear immediate danger.
    # OBSTACLE_BACKWARD_SPEED is already negative, so it is passed as-is.
    if OBSTACLE_BACKWARD_SECONDS > 0:
        obstacle_status("REVERSE")
        timed_motion(
            motors,
            OBSTACLE_BACKWARD_SPEED, OBSTACLE_BACKWARD_SPEED,
            OBSTACLE_BACKWARD_SECONDS,
            f"Reversing for {OBSTACLE_BACKWARD_SECONDS:.2f}s"
        )
    if switch_turned_off(motors):
        return False

    # 2. Turn 45 degrees away from the line.
    obstacle_status(f"TURN {first_turn:+d}")
    if not obstacle_turn_signed(motors, first_turn):
        if switch_turned_off(motors):
            return False
        print(f"Turn ({first_turn:+d}°) failed; aborting obstacle protocol.")
        obstacle_status(f"FAILED {first_turn:+d}")
        return False

    # 3. Move forward.
    obstacle_status("FORWARD")
    timed_motion(
        motors,
        OBSTACLE_FORWARD_SPEED, OBSTACLE_FORWARD_SPEED,
        OBSTACLE_FORWARD_LONG_SECONDS,
        f"Driving forward for {OBSTACLE_FORWARD_LONG_SECONDS:.2f}s"
    )
    if switch_turned_off(motors):
        return False

    # 4. First 45-degree turn back toward the line.
    obstacle_status(f"TURN {back_turn:+d} (1)")
    if not obstacle_turn_signed(motors, back_turn):
        if switch_turned_off(motors):
            return False
        print(f"Turn ({back_turn:+d}°) failed; aborting obstacle protocol.")
        obstacle_status(f"FAILED {back_turn:+d}")
        return False

    # 5. Move forward between the two turns back.
    obstacle_status("FORWARD")
    timed_motion(
        motors,
        OBSTACLE_FORWARD_SPEED, OBSTACLE_FORWARD_SPEED,
        OBSTACLE_FORWARD_SHORT_SECONDS,
        f"Driving forward for {OBSTACLE_FORWARD_SHORT_SECONDS:.2f}s"
    )
    if switch_turned_off(motors):
        return False

    # 6. Second 45-degree turn back toward the line.
    obstacle_status(f"TURN {back_turn:+d} (2)")
    if not obstacle_turn_signed(motors, back_turn):
        if switch_turned_off(motors):
            return False
        print(f"Turn ({back_turn:+d}°) failed; aborting obstacle protocol.")
        obstacle_status(f"FAILED {back_turn:+d}")
        return False

    # 7. Drive forward until the camera sees the line, then a little more.
    obstacle_status("SEEK LINE")
    drive_until_line(motors, camera, OBSTACLE_FORWARD_SPEED, OBSTACLE_LINE_SEEK_TIMEOUT)
    if switch_turned_off(motors):
        return False

    obstacle_status("FORWARD")
    timed_motion(
        motors,
        OBSTACLE_FORWARD_SPEED, OBSTACLE_FORWARD_SPEED,
        OBSTACLE_PAST_LINE_SECONDS,
        f"Driving forward for {OBSTACLE_PAST_LINE_SECONDS:.2f}s"
    )
    if switch_turned_off(motors):
        return False

    # 8. Keep turning the same physical way as step 2 until the black line is found,
    # which lines the robot back up with the line.
    obstacle_status("FIND LINE")
    if not find_line_after_turn(motors, camera, direction=find_direction):
        if switch_turned_off(motors):
            return False
        print("Line not found after obstacle; aborting.")
        obstacle_status("LINE NOT FOUND")
        return False

    motors.stop_motors()
    obstacle_status("DONE")
    return True


# ================================================================ LINE FOLLOWER
class LineFollower:
    """One camera frame of line following per step(): red check, obstacle check,
    green markers, then normal line following + gap handling.
    Holds all the state that used to live in main()'s loop."""

    def __init__(self, motors, camera):
        self.motors = motors
        self.camera = camera

        W, H = FRAME_SIZE
        self.W, self.H = W, H
        margin_x = int(12 / 64 * W)
        margin_y = int(10 / 64 * H)
        self.radius = int(10 / 64 * W)
        shift_y = int(ROI_SHIFT * H)
        self.target_y = 20 / 64 * H + shift_y

        self.corner_mask = np.ones((H, W), dtype=np.uint8) * 255
        self.corner_mask[margin_y + shift_y:, margin_x:-margin_x] = 0
        self.corner_mask[:shift_y, margin_x:-margin_x] = 0

        self.black = np.zeros((H, W), dtype=np.uint8)
        self.green_mask = np.zeros((H, W), dtype=np.uint8)
        self.black_vis = np.zeros((H, W, 3), dtype=np.uint8)

        # Obstacle detection state
        self.obstacle_armed = True
        self.obstacle_clear_since = None
        self.last_sensor_counter = None
        self.last_sensor_change = time.monotonic()
        self.sensor_stale_warned = False

        self.reset()

    def reset(self):
        """Forget where the line was. Call after anything that moved the robot
        without line following (green turn, obstacle, red stop, ...)."""
        self.previous_mask = self.black.copy() + 255
        # Gap latch: once True, drive straight forward until the front ROI sees a line again
        self.going_forward_to_line = False
        self.left_pct, self.right_pct = 0.0, 0.0
        self.red_frames = 0

    # ------------------------------------------------------------ one frame
    def step(self):
        """Process one frame. Returns EVENT_NONE, EVENT_QUIT or EVENT_RED."""
        img = self.camera.get_camera_view()
        if img is None:
            self.motors.stop_motors()
            return EVENT_NONE

        clean = img.copy()   # undrawn copy for line following

        # ---- red stop ----
        red_found, _ = detect_red(clean)
        self.red_frames = self.red_frames + 1 if red_found else 0
        if self.red_frames >= RED_CONFIRM_FRAMES:
            self.motors.stop_motors()
            return EVENT_RED

        # ---- obstacle (ToF via ESP32) ----
        if self._check_obstacle():
            return EVENT_NONE

        # ---- green markers ----
        green_event = self._check_green(img)
        if green_event is not False:
            return green_event

        # ---- normal line following + gaps ----
        return self._follow_line(img, clean)

    # ------------------------------------------------------------ obstacle
    def _check_obstacle(self):
        """Returns True if an obstacle was handled this frame (robot moved)."""
        motors = self.motors
        sensor = get_sensor_data()
        now = time.monotonic()

        if sensor["counter"] != self.last_sensor_counter:
            self.last_sensor_counter = sensor["counter"]
            self.last_sensor_change = now

        sensor_fresh = (
            sensor["connected"]
            and now - self.last_sensor_change <= SENSOR_STALE_SECONDS
        )

        if not sensor_fresh:
            # Line following keeps running; only obstacle detection pauses.
            if not self.sensor_stale_warned:
                print("[OBSTACLE] Waiting for fresh ESP32 data; obstacle detection paused.")
                self.sensor_stale_warned = True
            return False

        if self.sensor_stale_warned:
            print("[OBSTACLE] ESP32 data fresh; obstacle detection active.")
            self.sensor_stale_warned = False

        distance = sensor["distance_mm"]
        valid = sensor["distance_valid"]

        if not motors.motor_switch.is_pressed:
            # Switch off: never trigger, and re-arm so the next run starts fresh.
            if not self.obstacle_armed:
                print("[OBSTACLE] Motor switch off; obstacle detection re-armed.")
            self.obstacle_armed = True
            self.obstacle_clear_since = None
            return False

        if valid and self.obstacle_armed and distance < OBSTACLE_THRESHOLD_MM:
            obstacle_status(f"DETECTED {distance:.0f}mm")
            motors.stop_motors()

            # Double-check: creep forward a little, then re-read the ToF.
            if not confirm_obstacle(motors):
                obstacle_status("FALSE ALARM")
                set_status("LFR")
                self.reset()
                return True

            avoid_obstacle(motors, self.camera)

            self.obstacle_armed = False
            self.obstacle_clear_since = None
            print(f"[OBSTACLE] Disarmed until the ToF reads >= {OBSTACLE_REARM_MM} mm"
                  f"{' (or nothing in range)' if OBSTACLE_INVALID_COUNTS_AS_CLEAR else ''}"
                  f" for {OBSTACLE_CLEAR_TIME_SECONDS:.1f}s.")

            set_status("LFR")
            self.reset()
            return True

        # Re-arm once the obstacle has cleared.
        if not self.obstacle_armed:
            clear = (
                (valid and distance >= OBSTACLE_REARM_MM)
                or (not valid and OBSTACLE_INVALID_COUNTS_AS_CLEAR)
            )
            if clear:
                if self.obstacle_clear_since is None:
                    self.obstacle_clear_since = now
                elif now - self.obstacle_clear_since >= OBSTACLE_CLEAR_TIME_SECONDS:
                    self.obstacle_armed = True
                    self.obstacle_clear_since = None
                    print("[OBSTACLE] ToF is clear; obstacle detection re-armed.")
            else:
                self.obstacle_clear_since = None

        return False

    # ------------------------------------------------------------ green
    def _check_green(self, img):
        """The trigger only looks at the bottom of the frame. If it sees any valid
        green, handle_green() pauses, aligns, then decides ONLY from a fresh
        full-frame check.
        Returns False if there was no green, otherwise EVENT_NONE or EVENT_QUIT."""
        gleft, gright, green_mask, black_vis = green_check(img, roi_top=GREEN_FIRST_CHECK_ROI_TOP)
        self.green_mask, self.black_vis = green_mask, black_vis   # kept for the preview

        if not (gleft or gright):
            return False

        self.going_forward_to_line = False   # green handling overrides gap seeking
        W, H = self.W, self.H

        if SHOW_PREVIEW:
            if show_preview(img, np.zeros((H, W, 3), np.uint8), green_mask, black_vis, 0, 0):
                return EVENT_QUIT

        if handle_green(self.motors, self.camera, gleft, gright):
            return EVENT_QUIT

        set_status("LFR")
        self.reset()
        return EVENT_NONE

    # ------------------------------------------------------------ line + gaps
    def _follow_line(self, img, clean):
        W, H = self.W, self.H
        motors = self.motors

        set_status("LFR" if motors.motor_switch.is_pressed else "STOPPED")
        full_line_mask = black_mask_rgb(clean)

        line_mask = full_line_mask.copy()
        line_mask[self.corner_mask == 0] = 0

        mask_with_previous_position = line_mask.copy()
        mask_with_previous_position[self.previous_mask == 0] = 0

        contours, _ = cv2.findContours(mask_with_previous_position,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_view = cv2.cvtColor(mask_with_previous_position, cv2.COLOR_GRAY2BGR)

        roi_pose = get_contour_pose(contours)

        # Only exit forward mode if the ROI sees a line in the front band (cy <= target_y)
        if roi_pose is not None and (not self.going_forward_to_line or roi_pose[1] <= self.target_y):
            self.going_forward_to_line = False
            cx, cy, _, _, _ = roi_pose

            self.previous_mask = cv2.circle(self.black.copy(), (cx, cy), self.radius, 255, -1)

            img = cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
            cv2.circle(mask_view, (cx, cy), self.radius, (255, 0, 0), 1)
            cv2.circle(mask_view, (cx, cy), 4, (0, 255, 0), -1)

            error_x = (cx - W / 2) / W * 64
            error_y = -(cy - self.target_y) / H * 64

            speed_left = np.clip(error_y * 5 + error_x * 3, -255, 255)
            speed_right = np.clip(error_y * 5 - error_x * 3, -255, 255)
            self.left_pct, self.right_pct = drive(motors, left=speed_left, right=speed_right)

        else:
            # No valid front ROI line: reopen previous_mask so the ROI can catch the next line
            self.previous_mask = self.black.copy() + 255

            # If not latched yet, check the full-frame mask for a centered, vertical line ahead
            if not self.going_forward_to_line:
                full_contours, _ = cv2.findContours(full_line_mask,
                                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                full_pose = get_contour_pose(full_contours)

                if full_pose is not None:
                    fcx, fcy, angle_deg, vx, vy = full_pose

                    img = cv2.circle(img, (fcx, fcy), 5, (0, 165, 255), -1)
                    pt1 = (int(fcx - vx * 35), int(fcy - vy * 35))
                    pt2 = (int(fcx + vx * 35), int(fcy + vy * 35))
                    cv2.line(img, pt1, pt2, (0, 165, 255), 2)

                    is_centered = abs(fcx - (W / 2)) <= (CENTER_TOLERANCE * W)
                    is_aligned = abs(angle_deg) <= ANGLE_TOLERANCE_DEG

                    if is_centered and is_aligned:
                        self.going_forward_to_line = True

            if self.going_forward_to_line:
                if SHOW_PREVIEW:
                    put_label(img, "SEEKING FWD...", (8, 45), (0, 165, 255))
                self.left_pct, self.right_pct = drive(motors, left=FWD_SPEED, right=FWD_SPEED)
            else:
                self.left_pct, self.right_pct = drive(motors, left=-50, right=-50)

        if SHOW_PREVIEW:
            if show_preview(img, mask_view, self.green_mask, self.black_vis,
                            self.left_pct, self.right_pct):
                return EVENT_QUIT

        return EVENT_NONE

"""Rescue (evacuation zone). main.py calls run_rescue() once silver is seen.

Flow:
  1. Drive forward RESCUE_ENTER_TIME to get towards the middle of the zone.
  2. Rescue SILVER_BALLS_TO_RESCUE silver balls first, then BLACK_BALLS_TO_RESCUE
     black balls, one at a time:
       find the ball -> centre it -> lower the arm -> approach until the ToF says
       it is in the gripper -> close + lift -> check the ToF (ball gone from the
       floor) -> find the box (silver -> GREEN, black -> RED) -> drive up to it
       -> lower the arm only ARM_DROP_FRACTION of the way -> open -> lift -> back off.
     PICK_MAX_ATTEMPTS failed picks in a row -> turn 90 deg and scan again.
  3. After the last black ball: back off, turn 90 deg right, follow the LEFT wall
     (LEFT_F / LEFT_B ToFs) and look for the exit:
       LEFT_F sees nothing (gap in the wall) -> stop, check the floor with the
       bottom (line) camera:
         black          -> exit: move up, turn left into the gap, drive out
         silver         -> entrance (wrong exit): back up, go straight past it, keep following
         nothing        -> go forward, check LEFT_B too; if both see nothing,
                           turn 90 left into the gap and check the floor again:
                             black         -> exit
                             otherwise     -> turn back straight, go forward, keep following
  4. Return to main.py: True = left through the exit (back to line following).

Ball detection is only ~2 fps on the Pi, so everything is stop-and-look: move a
short pulse (never slower than RESCUE_MIN_SPEED), stop, settle, then decide from a
frame captured AFTER stopping. Searching turns in steps; after SEARCH_ROTATIONS
full turns with nothing found, the robot turns a random angle, drives forward a
bit, and searches again. Wall following is continuous (ToF rate, no ball camera).
"""
import glob
import math
import os
import random
import sys
import threading
import time
from collections import namedtuple

import cv2
import numpy as np

from config import *
from robot import get_sensor_data, set_status, turn_with_gyro, normalize_angle_diff, fresh_frame
from vision import (
    quit_requested, request_quit, put_label, live_preview,
    black_mask_rgb, green_mask_hsv, detect_silver,
)

SILVER = "silver"
BLACK = "black"
BALL_LABEL = {SILVER: "Silver Ball", BLACK: "Black Ball"}   # names in BALL_CLASS_NAMES
BOX_FOR = {SILVER: "green", BLACK: "red"}

# label: text, box: (x1, y1, x2, y2) in frame pixels,
# score: confidence for balls, fraction of the frame covered for boxes
Target = namedtuple("Target", "label box score")


class RescueAbort(Exception):
    """'q' was pressed in the preview: leave rescue."""


def _min_speed(v):
    """Raise a non-zero wheel speed to at least RESCUE_MIN_SPEED, keeping its sign."""
    if v == 0:
        return 0
    return math.copysign(max(abs(v), RESCUE_MIN_SPEED), v)


def _offset(box, width):
    """-1 (far left) .. +1 (far right) of the box centre."""
    cx = (box[0] + box[2]) / 2
    return (cx - width / 2) / (width / 2)


# ================================================================ USB CAMERA
def _ball_camera_source():
    if BALL_CAMERA != "auto":
        return BALL_CAMERA
    if sys.platform.startswith("linux"):
        # Stable names for USB cameras; the Pi camera / ISP use /dev/video0.. too
        usb = sorted(glob.glob("/dev/v4l/by-id/usb-*-video-index0"))
        if usb:
            return usb[0]
    return 0


class BallCamera:
    """USB camera read on its own thread, so frame_after() always gets a fresh frame."""

    def __init__(self):
        source = _ball_camera_source()
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open the ball camera ({source}); set BALL_CAMERA in config.py")
        w, h = BALL_FRAME_SIZE
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[RESCUE] Ball camera opened: {source}")

        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if BALL_CAMERA_FLIP is not None:
                frame = cv2.flip(frame, BALL_CAMERA_FLIP)
            with self._lock:
                self._frame, self._stamp = frame, time.monotonic()

    def frame_after(self, t, timeout=1.0):
        """First frame read after time t (time.monotonic()), or None on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None and self._stamp > t:
                    return self._frame.copy()
            time.sleep(0.005)
        return None

    def close(self):
        self._running = False
        self._thread.join(timeout=0.5)
        self.cap.release()


# ================================================================ DETECTION
class BallDetector:
    """balls.onnx (YOLOv8) through OpenCV DNN, same as ball_hunt.py."""

    def __init__(self):
        path = next((p for p in BALL_MODEL_PATHS if os.path.exists(p)), None)
        if path is None:
            raise RuntimeError(f"Ball model not found (tried {BALL_MODEL_PATHS})")
        self.net = cv2.dnn.readNetFromONNX(path)
        print(f"[RESCUE] Ball model loaded: {path}")

    @staticmethod
    def _letterbox(frame):
        """Fit the frame into a square with grey bars, like YOLO training did."""
        h, w = frame.shape[:2]
        scale = BALL_IMGSZ / max(h, w)
        nw, nh = round(w * scale), round(h * scale)
        px, py = (BALL_IMGSZ - nw) // 2, (BALL_IMGSZ - nh) // 2
        canvas = np.full((BALL_IMGSZ, BALL_IMGSZ, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh))
        return canvas, scale, px, py

    def detect_all(self, frame):
        """All balls as [Target, ...]."""
        img, scale, px, py = self._letterbox(frame)
        self.net.setInput(cv2.dnn.blobFromImage(img, 1 / 255.0, (BALL_IMGSZ, BALL_IMGSZ), swapRB=True))
        out = self.net.forward()[0].T               # (N, 4 + classes): cx, cy, w, h, scores
        scores = out[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= BALL_CONF
        if not keep.any():
            return []
        b, cls, conf = out[keep, :4], cls[keep], conf[keep]
        x1 = (b[:, 0] - b[:, 2] / 2 - px) / scale
        y1 = (b[:, 1] - b[:, 3] / 2 - py) / scale
        bw, bh = b[:, 2] / scale, b[:, 3] / scale
        rects = np.stack([x1, y1, bw, bh], 1).tolist()
        idx = cv2.dnn.NMSBoxes(rects, conf.tolist(), BALL_CONF, BALL_NMS_IOU)
        found = []
        for i in np.array(idx).flatten():
            x, y, w, h = rects[i]
            found.append(Target(BALL_CLASS_NAMES[int(cls[i])], (int(x), int(y), int(x + w), int(y + h)),
                                float(conf[i])))
        return found

    def find(self, frame, kind):
        """The biggest (= closest) ball of this kind (SILVER / BLACK), or None."""
        return closest_ball(self.detect_all(frame), kind)


def closest_ball(balls, kind):
    """The biggest (= closest) ball of this kind in a detect_all() list, or None."""
    balls = [b for b in balls if b.label == BALL_LABEL[kind]]
    if not balls:
        return None
    return max(balls, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]))


def blocking_ball(balls, kind, target, width):
    """A ball of the OTHER kind that is closer than the target (lower in the frame)
    and in the robot's path, or None. The ToF can't tell colours, so driving on
    would grab that ball instead."""
    in_path = [b for b in balls
               if b.label != BALL_LABEL[kind]
               and abs(_offset(b.box, width)) < BLOCK_PATH_OFFSET
               and (target is None or b.box[3] > target.box[3])]
    return max(in_path, key=lambda b: b.box[3]) if in_path else None


def detect_box(frame, colour):
    """The biggest green / red blob big enough to be an evacuation box, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if colour == "green":
        mask = cv2.inRange(hsv, EVAC_GREEN_LOWER, EVAC_GREEN_UPPER)
    else:
        mask = cv2.bitwise_or(cv2.inRange(hsv, EVAC_RED1_LOWER, EVAC_RED1_UPPER),
                              cv2.inRange(hsv, EVAC_RED2_LOWER, EVAC_RED2_UPPER))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    h, w = frame.shape[:2]
    area_frac = cv2.contourArea(biggest) / float(w * h)
    x, y, bw, bh = cv2.boundingRect(biggest)
    if area_frac < EVAC_MIN_AREA_FRAC or bh < EVAC_MIN_HEIGHT_FRAC * h:
        return None
    return Target(f"{colour} box", (x, y, x + bw, y + bh), area_frac)


def classify_floor(frame):
    """What the bottom (line) camera sees at a gap in the wall:
    'silver' (entrance = wrong exit), 'exit' (black line = correct exit) or None.
    There is no green at the exit; the green evacuation box must NOT count, or a
    gap check next to it would drive into the corner.
    Silver uses the silver1.pt model, which the silver thread loaded at start-up."""
    try:
        silver_seen, _ = detect_silver(frame)
    except Exception as e:
        print(f"[RESCUE] Silver check failed: {e}")
        silver_seen = False
    if silver_seen:
        return "silver"

    roi = frame[int(frame.shape[0] * FLOOR_ROI_TOP):]
    black = black_mask_rgb(roi)
    black[green_mask_hsv(roi) > 0] = 0  # dark green (evacuation box) must not count as black
    if cv2.countNonZero(black) >= EXIT_BLACK_MIN_PIXELS:
        return "exit"
    return None


# ================================================================ ToF
class TofBelow:
    """Callable stop check: True once the ESP32 ToF has read a valid distance below
    limit_mm for `reads` fresh packets in a row."""

    def __init__(self, limit_mm, reads):
        self.limit_mm = limit_mm
        self.reads = reads
        self.last_counter = get_sensor_data()["counter"]
        self.run = 0

    def __call__(self):
        d = get_sensor_data()
        if not d["connected"] or d["counter"] == self.last_counter:
            return False
        self.last_counter = d["counter"]
        close = d["distance_valid"] and d["distance_mm"] < self.limit_mm
        self.run = self.run + 1 if close else 0
        return self.run >= self.reads


def fresh_tof_reads(count, timeout):
    """Up to `count` ESP32 packets that arrive from now on, within timeout seconds."""
    reads = []
    last = get_sensor_data()["counter"]
    deadline = time.monotonic() + timeout
    while len(reads) < count and time.monotonic() < deadline:
        d = get_sensor_data()
        if d["connected"] and d["counter"] != last:
            last = d["counter"]
            reads.append(d)
        time.sleep(0.005)
    return reads


# ================================================================ ARM
class Arm:
    """Front gripper + back arm: position-mode servos on the same bus as the wheels.
    Does nothing when the servo board is not connected (motors.dry_run)."""

    REG_MIN_ANGLE = 9
    REG_MAX_ANGLE = 11
    REG_TORQUE = 40
    REG_GOAL_POS = 42
    REG_GOAL_TIME = 44
    REG_SPEED = 46
    REG_LOCK = 48
    REG_PRESENT_POS = 56

    def __init__(self, motors):
        self.motors = motors
        self.enabled = not motors.dry_run and motors.ph is not None

    # ---- bus helpers (share the wheel bus lock) ----
    def _w1(self, ID, reg, val, tries=3):
        for _ in range(tries):
            with self.motors.lock:
                if self.motors.ph.write1ByteTxRx(self.motors.port, ID, reg, val)[0] == 0:
                    return True
            time.sleep(0.01)
        return False

    def _w2(self, ID, reg, val, tries=3):
        for _ in range(tries):
            with self.motors.lock:
                if self.motors.ph.write2ByteTxRx(self.motors.port, ID, reg, val)[0] == 0:
                    return True
            time.sleep(0.01)
        return False

    def _r2(self, ID, reg, tries=3):
        for _ in range(tries):
            with self.motors.lock:
                v, comm, _ = self.motors.ph.read2ByteTxRx(self.motors.port, ID, reg)
            if comm == 0:
                return v
            time.sleep(0.01)
        return None

    def setup(self):
        """Position mode, hold the current position, set speed, torque on."""
        if not self.enabled:
            print("[RESCUE] Servo board not connected: arm disabled.")
            return
        for ID in (FRONT_GRIP_ID, BACK_ARM_ID):
            # Angle limits 0..1023 = position mode (EEPROM, only written if different)
            if (self._r2(ID, self.REG_MIN_ANGLE), self._r2(ID, self.REG_MAX_ANGLE)) != (0, 1023):
                self._w1(ID, self.REG_LOCK, 0)
                self._w2(ID, self.REG_MIN_ANGLE, 0)
                self._w2(ID, self.REG_MAX_ANGLE, 1023)
                self._w1(ID, self.REG_LOCK, 1)
                time.sleep(0.1)
            current = self._r2(ID, self.REG_PRESENT_POS)
            if current is not None:
                self._w2(ID, self.REG_GOAL_POS, current)   # no jump when torque comes on
            self._w2(ID, self.REG_GOAL_TIME, 0)
            self._w2(ID, self.REG_SPEED, ARM_SPEED)
            self._w1(ID, self.REG_TORQUE, 1)

    def move(self, ID, target, squeeze=0):
        """Go to target and wait until there. If blocked (gripping the ball),
        hold `squeeze` units past the contact point."""
        target = max(0, min(1023, int(target)))
        if not self.enabled:
            return
        self._w2(ID, self.REG_GOAL_POS, target)
        t0, last, still_since = time.time(), None, None
        while time.time() - t0 < ARM_MOVE_TIMEOUT:
            time.sleep(ARM_POLL)
            pos = self._r2(ID, self.REG_PRESENT_POS)
            if pos is None:
                continue
            if abs(pos - target) <= ARM_POS_TOL:
                break
            if last is not None and abs(pos - last) <= 2:
                still_since = still_since or time.time()
                if time.time() - still_since > ARM_STALL_TIME:
                    hold = max(0, min(1023, pos + (1 if target > pos else -1) * squeeze))
                    self._w2(ID, self.REG_GOAL_POS, hold)
                    break
            else:
                still_since = None
            last = pos
        time.sleep(ARM_SETTLE)

    def open_grip(self):
        self.move(FRONT_GRIP_ID, GRIP_OPEN)

    def close_grip(self):
        self.move(FRONT_GRIP_ID, GRIP_CLOSED, squeeze=GRIP_SQUEEZE)

    def up(self):
        self.move(BACK_ARM_ID, ARM_UP)

    def down(self):
        self.move(BACK_ARM_ID, ARM_DOWN)

    def to_drop_height(self):
        """Only part of the way down (ARM_DROP_FRACTION), for depositing."""
        self.move(BACK_ARM_ID, ARM_UP + ARM_DROP_FRACTION * (ARM_DOWN - ARM_UP))


# ================================================================ RESCUE
class Rescue:
    def __init__(self, motors, line_camera):
        self.motors = motors
        self.line_cam = line_camera      # bottom camera, for the exit / entrance check
        self.arm = Arm(motors)
        self.cam = None
        self.detector = None
        self._no_frame_warned = False

    # ------------------------------------------------------------ top level
    def run(self):
        set_status("RESCUE ENTER")
        print("[RESCUE] Driving in to the middle of the zone")
        self.drive_for(RESCUE_ENTER_SPEED, RESCUE_ENTER_SPEED, RESCUE_ENTER_TIME,
                       stop_check=TofBelow(RESCUE_ENTER_STOP_MM, 2))

        self.arm.setup()
        self.arm.up()
        self.cam = BallCamera()
        self.detector = BallDetector()

        for kind, count in ((SILVER, SILVER_BALLS_TO_RESCUE), (BLACK, BLACK_BALLS_TO_RESCUE)):
            done = 0
            failures = 0        # failed picks in a row
            rounds = 0          # times PICK_MAX_ATTEMPTS failures happened for this ball
            while done < count:
                print(f"[RESCUE] Looking for {kind} ball {done + 1}/{count}")
                result = self.pick_ball(kind)
                if result is None:
                    continue            # other ball was in the way: not a failed pick
                if not result:
                    failures += 1       # missed or lost it: search again
                    if failures >= PICK_MAX_ATTEMPTS:
                        failures = 0
                        rounds += 1
                        if rounds >= PICK_MAX_ROUNDS:
                            print(f"[RESCUE] Can't pick this {kind} ball after "
                                  f"{rounds * PICK_MAX_ATTEMPTS} tries: skipping it")
                            rounds = 0
                            done += 1   # move on to the next ball
                            continue
                        print(f"[RESCUE] {PICK_MAX_ATTEMPTS} failed picks: turning {PICK_FAIL_TURN_DEG} deg, scanning again")
                        self.turn(PICK_FAIL_TURN_DEG, SEARCH_DIRECTION)
                    continue
                failures = 0
                rounds = 0
                self.deposit(BOX_FOR[kind])
                done += 1
                print(f"[RESCUE] {kind} ball {done}/{count} rescued")

        print("[RESCUE] All balls rescued: following the wall to the exit")
        self.cam.close()        # ball camera not needed any more; frees CPU / USB
        self.cam = None
        # deposit() already backed off the red box
        self.turn(WALL_START_TURN_DEG, "right")
        return self.follow_wall()

    def close(self):
        self.motors.stop_motors()
        if self.cam is not None:
            self.cam.close()
            self.cam = None
        if SHOW_PREVIEW:
            try:
                cv2.destroyWindow("Rescue")
            except cv2.error:
                pass

    # ------------------------------------------------------------ motion
    def drive_for(self, left, right, seconds, stop_check=None):
        """Drive for `seconds` (speeds raised to RESCUE_MIN_SPEED), then stop.
        stop_check() is polled every ~5 ms; True ends the move early.
        Returns True if stop_check ended it."""
        stopped_early = False
        self.motors.set_motors(_min_speed(left), _min_speed(right))
        end = time.monotonic() + seconds
        try:
            while time.monotonic() < end:
                if stop_check is not None and stop_check():
                    stopped_early = True
                    break
                time.sleep(0.005)
        finally:
            self.motors.stop_motors()
        return stopped_early

    def turn_pulse(self, direction, seconds, speed):
        """Short pivot. "right" = set_motors(+, -), same as turn_with_gyro()."""
        s = speed if direction == "right" else -speed
        self.drive_for(s, -s, seconds)

    def turn(self, angle, direction):
        """Gyro turn (timed fallback without the gyro)."""
        turn_with_gyro(self.motors, angle_deg=angle, direction=direction,
                       speed=max(GYRO_TURN_SPEED, RESCUE_MIN_SPEED),
                       fallback_time=angle / RESCUE_TURN_DEG_PER_SEC)

    def align_pulse(self, off):
        """Pivot towards an offset, longer for bigger offsets."""
        seconds = float(np.clip(abs(off) * ALIGN_SECONDS_PER_OFFSET, ALIGN_PULSE_MIN, ALIGN_PULSE_MAX))
        self.turn_pulse("right" if off > 0 else "left", seconds, BALL_ALIGN_SPEED)

    @staticmethod
    def _yaw():
        d = get_sensor_data()
        return d["yaw"] if d["connected"] and d["mpu_ok"] else None

    # ------------------------------------------------------------ looking
    def look(self, detect, label):
        """Stop, settle, then run detect() on a frame captured after that.
        Returns (Target or None, frame shape or None)."""
        self.motors.stop_motors()
        time.sleep(LOOK_SETTLE_TIME)
        frame = self.cam.frame_after(time.monotonic())
        if quit_requested():
            raise RescueAbort()
        if frame is None:
            if not self._no_frame_warned:
                print("[RESCUE] No frame from the ball camera")
                self._no_frame_warned = True
            return None, None

        target = detect(frame)
        self.show(frame, target, label)
        return target, frame.shape

    def show(self, frame, target, label):
        if not SHOW_PREVIEW:
            return
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
        cv2.line(frame, (0, int(BALL_CLOSE_FRAC * h)), (w, int(BALL_CLOSE_FRAC * h)), (0, 0, 255), 1)
        if target is not None:
            x1, y1, x2, y2 = target.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            put_label(frame, f"{target.label} {target.score:.2f}", (x1, max(15, y1 - 6)), (0, 255, 0))
        put_label(frame, label, (8, 20))
        cv2.imshow("Rescue", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            request_quit()
            raise RescueAbort()

    # ------------------------------------------------------------ searching
    def find(self, detect, label):
        """Search until the target is found, moving somewhere else after every
        SEARCH_ROTATIONS full turns with nothing found. Returns the Target."""
        while True:
            target = self.search_rotations(detect, label)
            if target is not None:
                return target
            print(f"[RESCUE] No {label} after {SEARCH_ROTATIONS} rotations, moving somewhere else")
            self.wander()

    def search_rotations(self, detect, label):
        """Turn in steps, looking after each one, for up to SEARCH_ROTATIONS turns.
        Turned angle comes from the gyro, or SEARCH_DEG_PER_STEP per step without it."""
        set_status(f"SEARCH {label.upper()}")
        turned = 0.0
        hits = 0
        while turned < SEARCH_ROTATIONS * 360:
            target, _ = self.look(detect, f"SEARCH {label} {turned:.0f}deg")
            if target is not None:
                hits += 1
                if hits >= TARGET_CONFIRM_LOOKS:
                    print(f"[RESCUE] Found {label}")
                    return target
                continue            # look again without turning
            hits = 0

            yaw0 = self._yaw()
            self.turn_pulse(SEARCH_DIRECTION, SEARCH_TURN_TIME, SEARCH_TURN_SPEED)
            yaw1 = self._yaw()
            step = abs(normalize_angle_diff(yaw1, yaw0)) if yaw0 is not None and yaw1 is not None else 0.0
            turned += step if step >= 1.0 else SEARCH_DEG_PER_STEP   # frozen / no gyro: estimate
        return None

    def wander(self):
        """Turn a random angle to a random side, then drive forward a bit."""
        set_status("RESCUE MOVE")
        angle = random.uniform(WANDER_MIN_DEG, WANDER_MAX_DEG)
        direction = random.choice(("left", "right"))
        print(f"[RESCUE] Turning {direction} {angle:.0f} deg, then forward")
        self.turn(angle, direction)
        self.drive_for(WANDER_SPEED, WANDER_SPEED, WANDER_TIME,
                       stop_check=TofBelow(WANDER_STOP_MM, 2))

    # ------------------------------------------------------------ balls
    def pick_ball(self, kind):
        """Find, approach, grab and lift one ball.
        Returns True if the ToF check after lifting says the ball left the floor,
        False for a failed pick, None if a ball of the other kind was in the way
        (not counted as a failed pick)."""
        label = f"{kind} ball"
        self.last_balls = []

        def detect(frame):
            self.last_balls = self.detector.detect_all(frame)   # kept for the blocking checks
            return closest_ball(self.last_balls, kind)

        self.find(detect, label)
        set_status(f"APPROACH {kind.upper()}")
        result = self.approach_ball(detect, label, kind)
        if result is None:
            return None
        if not result:
            print(f"[RESCUE] Lost the {label}, searching again")
            self.arm.up()
            return False

        # Last check before closing: is the closest ball in front the right kind?
        _, shape = self.look(detect, f"CHECK {label}")
        h, w = shape[:2] if shape is not None else (BALL_FRAME_SIZE[1], BALL_FRAME_SIZE[0])
        wrong = blocking_ball(self.last_balls, kind, closest_ball(self.last_balls, kind), w)
        if wrong is not None and wrong.box[3] >= BALL_CLOSE_FRAC * h:
            print(f"[RESCUE] {wrong.label} is in the gripper, not a {label}: backing off")
            self.arm.up()
            self.go_around(wrong, w)
            return None
        return self.grab_and_check(label)

    def go_around(self, blocker, width):
        """Back off and come at the target from another angle, away from the
        ball that is in the way."""
        self.drive_for(-GRAB_BACKUP_SPEED, -GRAB_BACKUP_SPEED, GRAB_BACKUP_TIME)
        away = "left" if _offset(blocker.box, width) >= 0 else "right"
        self.turn(BLOCK_TURN_DEG, away)
        self.drive_for(APPROACH_SPEED, APPROACH_SPEED, BLOCK_SIDESTEP_TIME,
                       stop_check=TofBelow(WANDER_STOP_MM, 2))

    def approach_ball(self, detect, label, kind):
        """Centre the ball, lower the arm (only once centred), then drive at it in
        pulses until the ToF says it is in the gripper. Returns True when it is,
        False if lost, None if a ball of the other kind is in the way."""
        arm_down = False
        lost = 0
        last_bottom = 0.0       # bottom edge of the last sighting, fraction of frame height

        while True:
            target, shape = self.look(detect, f"APPROACH {label}")

            if target is None:
                lost += 1
                if arm_down and last_bottom >= BALL_CLOSE_FRAC:
                    return self.creep_to_ball()
                if lost >= LOST_LOOKS:
                    return False
                continue

            lost = 0
            h, w = shape[:2]
            last_bottom = target.box[3] / h
            off = _offset(target.box, w)

            if abs(off) > BALL_CENTER_TOLERANCE:
                self.align_pulse(off)
                continue

            # A ball of the other kind between us and the target: the ToF would grab it
            blocker = blocking_ball(self.last_balls, kind, target, w)
            if blocker is not None:
                print(f"[RESCUE] {blocker.label} is in the way of the {label}: going around")
                self.arm.up()
                self.go_around(blocker, w)
                return None

            if not arm_down:
                print("[RESCUE] Ball centred: opening the grip and lowering the arm")
                set_status("ARM DOWN")
                self.arm.open_grip()
                self.arm.down()
                arm_down = True
                continue        # look again: lowering the arm may have nudged the robot

            # Centred, arm down: forward pulse, shorter as the ball gets closer
            pulse = float(np.interp(last_bottom, [0.0, BALL_CLOSE_FRAC],
                                    [APPROACH_PULSE_MAX, APPROACH_PULSE_MIN]))
            if self.drive_for(APPROACH_SPEED, APPROACH_SPEED, pulse,
                              stop_check=TofBelow(GRAB_MM, GRAB_CONFIRM_READS)):
                print("[RESCUE] ToF: ball in the gripper")
                return True

    def creep_to_ball(self):
        """The ball went under the camera: creep straight and let the ToF decide."""
        print("[RESCUE] Ball went under the camera: creeping, waiting for the ToF")
        if not get_sensor_data()["connected"]:
            # No ToF to wait for: creep the full time and grab anyway
            self.drive_for(BLIND_SPEED, BLIND_SPEED, BLIND_TIME)
            return True
        if self.drive_for(BLIND_SPEED, BLIND_SPEED, BLIND_TIME,
                          stop_check=TofBelow(GRAB_MM, GRAB_CONFIRM_READS)):
            print("[RESCUE] ToF: ball in the gripper")
            return True
        return False

    def grab_and_check(self, label):
        """Close, lift, then check the ToF: picked = every fresh reading is invalid
        (nothing there) or above PICK_CONFIRM_MM. Backs off either way."""
        set_status("GRAB")
        print(f"[RESCUE] Grabbing the {label}")
        self.arm.close_grip()
        self.arm.up()

        # Checked BEFORE backing off: after reversing, a missed ball would be
        # further than PICK_CONFIRM_MM anyway and the check would always pass.
        time.sleep(PICK_CHECK_SETTLE)
        reads = fresh_tof_reads(PICK_CONFIRM_READS, PICK_CHECK_TIMEOUT)
        if not reads:
            print("[RESCUE] No ToF data for the pick check: assuming it was picked")
            picked = True
        else:
            picked = all(not r["distance_valid"] or r["distance_mm"] > PICK_CONFIRM_MM for r in reads)
            shown = ", ".join(f"{r['distance_mm']:.0f}" if r["distance_valid"] else "none" for r in reads)
            print(f"[RESCUE] Pick check ToF: {shown} mm -> {'PICKED' if picked else 'MISSED'}")

        if not picked:
            self.arm.open_grip()   # let go of whatever is half-held; search again

        self.drive_for(-GRAB_BACKUP_SPEED, -GRAB_BACKUP_SPEED, GRAB_BACKUP_TIME)
        return picked

    # ------------------------------------------------------------ boxes
    def deposit(self, colour):
        """Find the box, drive up to it, drop the ball from ARM_DROP_FRACTION height."""
        label = f"{colour} box"
        detect = lambda frame: detect_box(frame, colour)

        while True:
            self.find(detect, label)
            set_status(f"TO {colour.upper()} BOX")
            if self.approach_box(detect, label):
                break
            print(f"[RESCUE] Lost the {label}, searching again")

        set_status(f"DROP {colour.upper()}")
        print(f"[RESCUE] At the {label}: dropping the ball")
        self.arm.to_drop_height()
        self.arm.open_grip()
        time.sleep(DEPOSIT_RELEASE_TIME)
        self.arm.up()
        self.drive_for(-DEPOSIT_BACKUP_SPEED, -DEPOSIT_BACKUP_SPEED, DEPOSIT_BACKUP_TIME)

    def approach_box(self, detect, label):
        """Centre on the box and pulse forward until it fills EVAC_STOP_AREA_FRAC of
        the frame or the ToF reads closer than EVAC_STOP_MM. Returns True on arrival."""
        lost = 0
        while True:
            target, shape = self.look(detect, f"APPROACH {label}")
            if target is None:
                lost += 1
                if lost >= LOST_LOOKS:
                    return False
                continue
            lost = 0

            off = _offset(target.box, shape[1])
            if abs(off) > EVAC_CENTER_TOLERANCE:
                self.align_pulse(off)
                continue

            if target.score >= EVAC_STOP_AREA_FRAC:
                return True
            if self.drive_for(APPROACH_SPEED, APPROACH_SPEED, EVAC_APPROACH_PULSE,
                              stop_check=TofBelow(EVAC_STOP_MM, 2)):
                return True


    # ------------------------------------------------------------ wall following
    @staticmethod
    def near_wall(mm):
        """Left ToF reading -> mm, or None if nothing within WALL_SEE_MAX_MM."""
        return mm if mm is not None and mm <= WALL_SEE_MAX_MM else None

    @staticmethod
    def wall_speeds(lf, lb, lost_for=0.0):
        """(left, right) % to keep WALL_TARGET_MM from the left wall (wall_follow.py
        logic). No wall at all: curve left to find it again, but after
        WALL_LOST_ARC_TIME seconds go straight until the front ToF finds a wall."""
        if lf is None and lb is None:
            if lost_for > WALL_LOST_ARC_TIME:
                return WALL_SPEED, WALL_SPEED
            return WALL_LOST_LEFT_SPEED, WALL_SPEED
        if lf is not None and lb is not None:
            d = (lf + lb) / 2
            steer = WALL_KP_DIST * (d - WALL_TARGET_MM) + WALL_KP_ANGLE * (lf - lb)
        else:
            d = lf if lf is not None else lb
            steer = WALL_KP_DIST * (d - WALL_TARGET_MM)
        # steer > 0: too far / angled away from the wall -> turn left (slow the left side)
        steer = float(np.clip(steer, -WALL_MAX_STEER, WALL_MAX_STEER))
        return _min_speed(WALL_SPEED - steer), _min_speed(WALL_SPEED + steer)

    def follow_wall(self):
        """Follow the left wall until the exit is found and driven through.
        Returns True on exit, False after WALL_FOLLOW_TIMEOUT."""
        set_status("WALL FOLLOW")
        print("[RESCUE] Following the left wall")
        t0 = time.monotonic()
        last = get_sensor_data()["counter"]
        front_hits = 0
        gap_reads = 0
        gap_armed = False       # a gap only counts after LEFT_F has seen a wall
        lost_since = None       # when both left sensors last stopped seeing a wall

        try:
            while time.monotonic() - t0 < WALL_FOLLOW_TIMEOUT:
                if quit_requested():
                    raise RescueAbort()
                d = get_sensor_data()
                if not d["connected"]:
                    self.motors.stop_motors()       # no sensors: don't drive blind
                    time.sleep(0.05)
                    continue
                if d["counter"] == last:
                    time.sleep(0.005)
                    continue
                last = d["counter"]
                lf, lb = self.near_wall(d["left_f"]), self.near_wall(d["left_b"])

                # ---- wall ahead: corner turn to the right ----
                close = d["distance_valid"] and d["distance_mm"] < WALL_FRONT_STOP_MM
                front_hits = front_hits + 1 if close else 0
                if front_hits >= WALL_FRONT_CONFIRM:
                    print("[RESCUE] Wall ahead: corner turn")
                    self.motors.stop_motors()
                    self.turn(WALL_CORNER_TURN_DEG, "right")
                    front_hits = 0
                    lost_since = None
                    last = get_sensor_data()["counter"]
                    continue

                # ---- gap in the left wall ----
                if lf is not None or lb is not None:
                    gap_armed = True        # either left sensor seeing a wall re-arms the gap check
                if lf is not None:
                    gap_reads = 0
                elif gap_armed:
                    gap_reads += 1
                    if gap_reads >= WALL_GAP_CONFIRM:
                        gap_armed = False
                        gap_reads = 0
                        if self.handle_gap():
                            return True
                        set_status("WALL FOLLOW")
                        lost_since = None
                        last = get_sensor_data()["counter"]
                        continue

                if lf is None and lb is None:
                    lost_since = lost_since if lost_since is not None else time.monotonic()
                else:
                    lost_since = None
                lost_for = time.monotonic() - lost_since if lost_since is not None else 0.0
                left, right = self.wall_speeds(lf, lb, lost_for)
                self.motors.set_motors(left, right)
        finally:
            self.motors.stop_motors()

        print("[RESCUE] No exit found before WALL_FOLLOW_TIMEOUT")
        return False

    def check_floor(self, label):
        """Stop and classify FLOOR_CHECK_FRAMES fresh bottom-camera frames.
        Exit is only correct with NO silver and black present, so:
        'silver' if any frame saw silver, else 'exit' if any frame saw black, else None."""
        self.motors.stop_motors()
        time.sleep(LOOK_SETTLE_TIME)
        votes = []
        for _ in range(FLOOR_CHECK_FRAMES):
            frame = fresh_frame(self.line_cam)
            if frame is None:
                continue
            votes.append(classify_floor(frame))
            if live_preview(frame, label):
                raise RescueAbort()
        if "silver" in votes:
            result = "silver"           # any silver = entrance (wrong exit), even if black was seen
        elif "exit" in votes:
            result = "exit"
        else:
            result = None
        print(f"[RESCUE] {label}: {votes} -> {result}")
        return result

    def fresh_left(self):
        """(left_f, left_b) from a packet that arrives from now on."""
        reads = fresh_tof_reads(1, 0.5)
        d = reads[-1] if reads else get_sensor_data()
        return self.near_wall(d["left_f"]), self.near_wall(d["left_b"])

    def handle_gap(self):
        """LEFT_F sees nothing. Returns True if this was the exit and the robot
        drove out, False to carry on following the wall."""
        set_status("GAP CHECK")
        print("[RESCUE] Left-front sees nothing: checking the floor")
        floor = self.check_floor("GAP: FLOOR CHECK")

        if floor == "exit":
            self.exit_zone(turned_in=False)
            return True

        if floor == "silver":
            print("[RESCUE] Silver: this is the entrance, going past it")
            self.drive_for(-WALL_SPEED, -WALL_SPEED, WALL_BACKUP_TIME)
            if ENTRANCE_TURN_DEG > 0:
                self.turn(ENTRANCE_TURN_DEG, "right")
            self.drive_for(WALL_SPEED, WALL_SPEED, ENTRANCE_PASS_TIME)
            return False

        # Nothing on the floor yet: move up so LEFT_B reaches the gap too
        self.drive_for(WALL_SPEED, WALL_SPEED, WALL_GAP_FWD_TIME)
        lf, lb = self.fresh_left()
        if lf is not None or lb is not None:
            print("[RESCUE] Left wall is back: carrying on")
            return False

        print("[RESCUE] Both left sensors see nothing: turning in to check")
        self.turn(90, "left")
        floor = self.check_floor("GAP: TURNED IN")
        if floor == "exit":
            self.exit_zone(turned_in=True)
            return True

        print(f"[RESCUE] Not the exit ({floor}): turning back and carrying on")
        self.turn(90, "right")
        self.drive_for(WALL_SPEED, WALL_SPEED, WALL_PASS_TIME)
        return False

    def exit_zone(self, turned_in):
        """Face the gap (if not already) and drive out; line following takes over."""
        set_status("EXIT")
        print("[RESCUE] Exit found: leaving the evacuation zone")
        if not turned_in:
            self.drive_for(WALL_SPEED, WALL_SPEED, WALL_GAP_FWD_TIME)
            self.turn(90, "left")
        self.arm.up()
        self.drive_for(WALL_SPEED, WALL_SPEED, EXIT_DRIVE_TIME)


def run_rescue(motors, camera):
    """Called by main.py once silver is seen. `camera` is the line (bottom) camera,
    used for the exit check; balls use their own USB camera.
    Returns True if the robot left through the exit (go back to line following),
    False if it stopped (quit, timeout, or the ball camera / model failed)."""
    rescue = Rescue(motors, camera)
    try:
        return bool(rescue.run())
    except RescueAbort:
        print("[RESCUE] Quit pressed")
    except RuntimeError as e:
        print(f"[RESCUE] Can't run rescue: {e}")
    finally:
        rescue.close()
    return False

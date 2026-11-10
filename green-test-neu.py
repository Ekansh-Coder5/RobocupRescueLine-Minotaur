"""
Intersection + green-dot test (standalone, does not use the other robot files).

On the Pi:      python cross_test.py
On a PC/Pi with a picture (no robot needed):
                python cross_test.py photo.png

How it works, every camera frame:
 1. Split the frame into WHITE areas separated by the black line.
      2 areas = plain line, 3 = T junction, 4 = X cross.
 2. Find where the line enters at the bottom of the frame (PURPLE point).
    The white pixel just left of it is BLUE, just right of it is RED.
    The white area holding the blue pixel is the BLUE area, the one holding the
    red pixel is the RED area. Every other white area is YELLOW.
 3. Green squares only count if they are inside the blue or red area (others
    are ignored). Each green has a score that goes up while it is inside
    blue/red and down while it is outside, so a green must stay inside for a
    few frames before it counts.
 4. Decide, then delete the black pixels the robot should NOT follow:
      green in red only    -> keep black next to the red area  (turn right)
      green in blue only   -> keep black next to the blue area (turn left)
      green in both, and the grown greens touch -> double green (U-turn)
      no green, T junction -> near the junction, the blue/red area WITHOUT a
                              corner is the straight side; keep black next to it
      no green, X cross    -> keep the straight line, delete the side arms
 5. Follow what is left of the black line with a simple P controller.
"""
import os
import sys
import time
import threading
import numpy as np
import cv2


# ==================== SETTINGS ====================
FRAME_SIZE = (320, 240)
SHOW_PREVIEW = bool(os.environ.get("DISPLAY"))   # preview only when a screen is available
DRIVE = True                   # False = only detect and show, motors stay stopped

# ---- Colours (same as the main robot) ----
LOWER_BLACK = np.array([0, 0, 0])          # RGB
UPPER_BLACK = np.array([60, 60, 60])       # RGB
LOWER_GREEN_HSV = np.array([45, 80, 50])   # HSV, OpenCV scale
UPPER_GREEN_HSV = np.array([85, 255, 255])

# ---- White areas ----
MIN_WHITE_AREA = 400           # px; smaller white bits are noise and not counted

# ---- Line entry (purple / blue / red points) ----
ENTRY_ROWS = 6                 # bottom rows used to find where the line enters
ENTRY_SIDE_OFFSET = 4          # px past the line edge for the blue / red points
MIN_ENTRY_WIDTH = 6            # px; narrower black runs at the bottom are ignored

# ---- Green squares ----
MIN_GREEN_AREA = 150           # px; smaller green blobs are ignored
GREEN_SCORE_MAX = 10
GREEN_SCORE_UP = 2             # added each frame the green is inside blue/red
GREEN_SCORE_DOWN = 1           # removed each frame the green is outside
GREEN_CONFIRM_SCORE = 6        # score needed before a green counts
GREEN_MATCH_DIST = 40          # px; the same green in the next frame must be this close
GREEN_EXPAND = 25              # px each green is grown by for the double-green "touch" test
DOUBLE_GREEN_TRIGGER_Y = 0.55  # U-turn once both greens are below this fraction of frame height

# ---- Keeping / deleting black ----
KEEP_EXPAND = 30               # px; a bit more than the line width in the camera image

# ---- Corner test (T junction without green) ----
CORNER_CIRCLE_RADIUS = 45      # px around the junction centre
CORNER_MAX_ANGLE = 130         # polygon corner sharper than this (degrees) counts as an "angle"
CORNER_APPROX_EPS = 4          # px, polygon simplification before looking for corners

# ---- Driving (motor speeds in percent) ----
BASE_SPEED = 22
KP = 0.12                      # turn per pixel of error from the frame centre
SPEED_LIMIT = 40
MIN_LINE_PIXELS = 150          # fewer black pixels than this = line lost, stop
UTURN_SPEED = 25
UTURN_SECONDS = 1.8            # pivot time for a U-turn; calibrate on the robot

# ---- Hardware ----
SWITCH_GPIO = 27
SERVO_PORT = "/dev/ttyACM0"
SERVO_BAUDRATE = 1000000

KERNEL3 = np.ones((3, 3), np.uint8)


def disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


# ==================== MOTORS ====================
class ServoMotors:
    """Same servo setup as the main robot: IDs 1,2 left, 3,4 right (3,4 reversed).
    The master switch on SWITCH_GPIO forces every motor to zero when off."""
    REG_ANGLE_LIMITS = 9
    REG_TORQUE = 40
    REG_PWM = 44
    REG_LOCK = 48

    def __init__(self, left_ids=(1, 2), right_ids=(3, 4), reversed_ids=(3, 4)):
        from gpiozero import Button
        from scservo_sdk import PortHandler, PacketHandler

        self.motor_switch = Button(SWITCH_GPIO, pull_up=True)
        self.left_ids = list(left_ids)
        self.right_ids = list(right_ids)
        self.reversed_ids = set(reversed_ids)
        self.all_ids = self.left_ids + self.right_ids
        self.lock = threading.Lock()
        self._warned = set()
        self.dry_run = False
        self.port = None
        self.ph = None

        try:
            self.port = PortHandler(SERVO_PORT)
            if not self.port.openPort():
                raise RuntimeError(f"Could not open {SERVO_PORT}")
            self.port.setBaudRate(SERVO_BAUDRATE)
            self.ph = PacketHandler(1)

            for ID in self.all_ids:
                model, comm, err = self.ph.ping(self.port, ID)
                if comm != 0:
                    raise RuntimeError(f"No reply from servo ID {ID}: {self.ph.getTxRxResult(comm)}")
                print(f"Connected ID {ID}, model:", model, "| error byte:", err)

            for ID in self.all_ids:
                self._eeprom_write(ID, self.REG_ANGLE_LIMITS, [0, 0, 0, 0])   # continuous mode
                self.ph.write1ByteTxRx(self.port, ID, self.REG_TORQUE, 1)

        except Exception as e:
            print(f"[DRY RUN] Servo board unavailable ({e}). Motors disabled.")
            self.dry_run = True
            if self.port is not None:
                try:
                    self.port.closePort()
                except Exception:
                    pass
            self.port = None

    def _eeprom_write(self, servo_id, reg, data):
        self.ph.write1ByteTxRx(self.port, servo_id, self.REG_LOCK, 0)
        time.sleep(0.05)
        self.ph.writeTxRx(self.port, servo_id, reg, len(data), data)
        time.sleep(0.05)
        self.ph.write1ByteTxRx(self.port, servo_id, self.REG_LOCK, 1)
        time.sleep(0.05)

    def _set_speed(self, servo_id, percent):
        if self.dry_run:
            return
        percent = max(-100, min(100, float(percent)))
        if servo_id in self.reversed_ids:
            percent = -percent
        raw = int(abs(percent) * 10) | (0x400 if percent < 0 else 0)
        comm, err = self.ph.write2ByteTxRx(self.port, servo_id, self.REG_PWM, raw)
        if comm != 0:
            print(f"ID {servo_id} WRITE FAILED: {self.ph.getTxRxResult(comm)}")
        elif err != 0 and servo_id not in self._warned:
            self._warned.add(servo_id)
            print(f"ID {servo_id} status byte = {err} (bit0 = voltage error) - check power supply")

    def set_motors(self, left, right):
        with self.lock:
            # Master switch OFF: force every motor to zero
            if not self.motor_switch.is_pressed:
                left = 0
                right = 0
            for ID in self.left_ids:
                self._set_speed(ID, left)
            for ID in self.right_ids:
                self._set_speed(ID, right)

    def stop_motors(self):
        self.set_motors(0, 0)

    def close(self):
        self.stop_motors()
        if self.port is not None:
            self.port.closePort()
        self.motor_switch.close()


class Camera:
    def __init__(self, size=FRAME_SIZE):
        from picamera2 import Picamera2
        from libcamera import Transform

        self.cam = Picamera2()
        full_w, full_h = self.cam.camera_properties["PixelArraySize"]
        config = self.cam.create_video_configuration(
            main={"size": size, "format": "RGB888"},
            transform=Transform(hflip=True, vflip=True),
            controls={"ScalerCrop": (0, 0, full_w, full_h)})
        self.cam.configure(config)
        self.cam.start()
        time.sleep(0.5)

    def get_camera_view(self):
        return self.cam.capture_array()

    def release(self):
        self.cam.stop()
        self.cam.close()


# ==================== STEP 1: MASKS AND WHITE AREAS ====================
def make_masks(frame):
    """Return (black, green, white) 0/255 masks. Green counts as white, never black."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    black = cv2.inRange(rgb, LOWER_BLACK, UPPER_BLACK)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, LOWER_GREEN_HSV, UPPER_GREEN_HSV)
    black[green > 0] = 0
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, KERNEL3)
    white = cv2.bitwise_not(black)
    return black, green, white


def white_areas(white):
    """Label the white areas. 4-connectivity, so a thin diagonal black line still
    separates two areas. Returns (labels, ids of areas big enough to count)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    areas = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_WHITE_AREA]
    return labels, areas


# ==================== STEP 2: LINE ENTRY (PURPLE / BLUE / RED) ====================
def find_entry(black, prev_x):
    """Find where the line enters at the bottom of the frame. If there are several
    black runs, take the one closest to last frame's entry. Returns a dict with the
    purple, blue and red points, or None if no line touches the bottom."""
    H, W = black.shape
    cols = black[H - ENTRY_ROWS:, :].max(axis=0) > 0

    runs = []
    x = 0
    while x < W:
        if cols[x]:
            start = x
            while x < W and cols[x]:
                x += 1
            if x - start >= MIN_ENTRY_WIDTH:
                runs.append((start, x - 1))
        else:
            x += 1
    if not runs:
        return None

    ref = prev_x if prev_x is not None else W / 2
    start, end = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - ref))
    y = H - 1 - ENTRY_ROWS // 2
    return {
        "purple": ((start + end) // 2, y),
        "blue": (max(0, start - ENTRY_SIDE_OFFSET), y),
        "red": (min(W - 1, end + ENTRY_SIDE_OFFSET), y),
    }


def area_at(labels, areas, point):
    label = int(labels[point[1], point[0]])
    return label if label in areas else None


# ==================== STEP 3: GREEN SQUARES ====================
def green_blobs(green, labels):
    """Green blobs big enough to count, each with the white area it sits in."""
    n, glabels, stats, centroids = cv2.connectedComponentsWithStats(green, connectivity=8)
    blobs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_GREEN_AREA:
            continue
        mask = glabels == i
        under = labels[mask]
        under = under[under > 0]
        region = int(np.bincount(under).argmax()) if under.size else 0
        blobs.append({"mask": mask, "center": tuple(centroids[i]), "region": region})
    return blobs


class GreenTracker:
    """Gives each green a score that follows it from frame to frame (matched by
    nearest centre). Inside blue/red: score goes up. Outside: score goes down.
    A green only counts once it is inside AND its score reached GREEN_CONFIRM_SCORE,
    so greens that were first seen outside need a few frames before they count."""

    def __init__(self):
        self.tracks = []

    def reset(self):
        self.tracks = []

    def update(self, blobs, blue_id, red_id):
        unused = list(self.tracks)
        new_tracks = []
        for b in blobs:
            best, best_d = None, GREEN_MATCH_DIST
            for t in unused:
                d = np.hypot(b["center"][0] - t["center"][0], b["center"][1] - t["center"][1])
                if d <= best_d:
                    best, best_d = t, d
            score = 0
            if best is not None:
                unused.remove(best)
                score = best["score"]

            inside = b["region"] != 0 and b["region"] in (blue_id, red_id)
            if inside:
                score = min(GREEN_SCORE_MAX, score + GREEN_SCORE_UP)
            else:
                score = max(0, score - GREEN_SCORE_DOWN)

            b["inside"] = inside
            b["score"] = score
            b["valid"] = inside and score >= GREEN_CONFIRM_SCORE
            new_tracks.append({"center": b["center"], "score": score})
        self.tracks = new_tracks


def greens_touch(left_blobs, right_blobs, shape):
    """Grow the left and right greens by GREEN_EXPAND and check if they overlap."""
    gl = np.zeros(shape, np.uint8)
    gr = np.zeros(shape, np.uint8)
    for b in left_blobs:
        gl[b["mask"]] = 255
    for b in right_blobs:
        gr[b["mask"]] = 255
    gl = cv2.dilate(gl, disk(GREEN_EXPAND))
    gr = cv2.dilate(gr, disk(GREEN_EXPAND))
    return bool(np.any(cv2.bitwise_and(gl, gr)))


# ==================== STEP 4: JUNCTION CENTRE AND CORNERS ====================
def junction_center(labels, area_ids):
    """Centre of the crossing: where at least 3 grown white areas overlap.
    Grows the areas a bit more each try until they meet."""
    for radius in (4, 8, 13, 18, 25):
        kernel = disk(radius)
        count = np.zeros(labels.shape, np.uint8)
        for a in area_ids:
            grown = cv2.dilate((labels == a).astype(np.uint8), kernel)
            count += grown
        ys, xs = np.nonzero(count >= 3)
        if xs.size:
            return int(xs.mean()), int(ys.mean())
    return None


def corners_near(area_mask, center, radius):
    """Corners ("angles") of a white area's outline inside the circle around the
    junction. Frame-edge corners are ignored."""
    H, W = area_mask.shape
    contours, _ = cv2.findContours(area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    corners = []
    for c in contours:
        poly = cv2.approxPolyDP(c, CORNER_APPROX_EPS, True).reshape(-1, 2).astype(float)
        n = len(poly)
        if n < 3:
            continue
        for i in range(n):
            p, a, b = poly[i], poly[i - 1], poly[(i + 1) % n]
            if np.hypot(p[0] - center[0], p[1] - center[1]) > radius:
                continue
            if p[0] <= 2 or p[1] <= 2 or p[0] >= W - 3 or p[1] >= H - 3:
                continue
            v1, v2 = a - p, b - p
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            if angle < CORNER_MAX_ANGLE:
                corners.append((int(p[0]), int(p[1])))
    return corners


def keep_near(black, labels, ids):
    """Keep only the black pixels within KEEP_EXPAND of the given white areas."""
    keep = np.isin(labels, ids).astype(np.uint8) * 255
    keep = cv2.dilate(keep, disk(KEEP_EXPAND))
    return cv2.bitwise_and(black, keep)


def straight_through_x(black, labels, blue_id, red_id, yellow_ids):
    """X cross, no green: keep the line we came on (between blue and red) and the
    line straight ahead (between yellow areas). Delete the side arms, which sit
    between a yellow area and the blue or red area: for an arm pixel, the distance
    to yellow plus the distance to blue/red is about one line width."""
    def dist_to(mask):
        # distance of every pixel to the nearest pixel of mask
        return cv2.distanceTransform(np.where(mask, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)

    d_blue = dist_to(labels == blue_id)
    d_red = dist_to(labels == red_id)
    d_yellow = dist_to(np.isin(labels, yellow_ids))

    side_arm = d_yellow + np.minimum(d_blue, d_red) <= KEEP_EXPAND
    centre = (d_blue <= KEEP_EXPAND) & (d_red <= KEEP_EXPAND)
    safe = ~side_arm | centre
    return cv2.bitwise_and(black, safe.astype(np.uint8) * 255)


# ==================== ALL STEPS FOR ONE FRAME ====================
def process_frame(frame, state):
    """Run the whole algorithm on one frame. Returns a dict with the cleaned black
    mask to follow, the decision, the action (FOLLOW / UTURN / LOST) and debug info.
    state keeps the last entry position and the green tracker between frames."""
    H, W = frame.shape[:2]
    black, green, white = make_masks(frame)
    labels, areas = white_areas(white)
    n_areas = len(areas)
    kind = "LINE" if n_areas <= 2 else ("T" if n_areas == 3 else "X")

    # ---- step 2: blue / red / yellow areas ----
    entry = find_entry(black, state.get("entry_x"))
    blue_id = red_id = None
    if entry is not None:
        state["entry_x"] = entry["purple"][0]
        blue_id = area_at(labels, areas, entry["blue"])
        red_id = area_at(labels, areas, entry["red"])
        if blue_id is not None and blue_id == red_id:
            blue_id = red_id = None      # both points in one area: no line between them
    yellow_ids = [a for a in areas if a not in (blue_id, red_id)]

    # ---- step 3: greens ----
    blobs = green_blobs(green, labels)
    state["tracker"].update(blobs, blue_id, red_id)
    left = [b for b in blobs if b["valid"] and b["region"] == blue_id]
    right = [b for b in blobs if b["valid"] and b["region"] == red_id]

    # ---- step 4: decide and clean the black mask ----
    clean = black.copy()
    action = "FOLLOW"
    decision = None
    center = None
    corners = {"blue": [], "red": []}

    if left and right:
        if greens_touch(left, right, black.shape):
            decision = "DOUBLE GREEN"
            if all(b["center"][1] >= DOUBLE_GREEN_TRIGGER_Y * H for b in left + right):
                action = "UTURN"
        else:
            # Two greens far apart are not a real double green: ignore both.
            left, right = [], []

    if decision is None:
        if left and not right:
            decision = "GREEN LEFT"
            clean = keep_near(black, labels, [blue_id])
        elif right and not left:
            decision = "GREEN RIGHT"
            clean = keep_near(black, labels, [red_id])
        elif kind == "T" and blue_id is not None and red_id is not None:
            center = junction_center(labels, areas)
            if center is None:
                decision = "T (no centre found)"
            else:
                for name, aid in (("blue", blue_id), ("red", red_id)):
                    corners[name] = corners_near((labels == aid).astype(np.uint8),
                                                 center, CORNER_CIRCLE_RADIUS)
                if corners["red"] and not corners["blue"]:
                    decision = "T STRAIGHT (keep blue)"
                    clean = keep_near(black, labels, [blue_id])
                elif corners["blue"] and not corners["red"]:
                    decision = "T STRAIGHT (keep red)"
                    clean = keep_near(black, labels, [red_id])
                else:
                    decision = "T (unclear, no change)"
        elif kind == "X" and blue_id is not None and red_id is not None and yellow_ids:
            decision = "X STRAIGHT"
            center = junction_center(labels, areas)
            clean = straight_through_x(black, labels, blue_id, red_id, yellow_ids)
        else:
            decision = kind

    if cv2.countNonZero(clean) < MIN_LINE_PIXELS:
        action = "LOST" if action == "FOLLOW" else action

    return {
        "black": black, "clean": clean, "labels": labels, "areas": areas,
        "kind": kind, "entry": entry, "blue_id": blue_id, "red_id": red_id,
        "yellow_ids": yellow_ids, "blobs": blobs, "decision": decision,
        "action": action, "center": center, "corners": corners,
    }


# ==================== STEP 5: STEERING ====================
def steering(clean):
    """P controller on the centre of the cleaned black mask. Returns (left, right)
    motor percent, or None if there is not enough line."""
    M = cv2.moments(clean, binaryImage=True)
    if M["m00"] < MIN_LINE_PIXELS:
        return None
    cx = M["m10"] / M["m00"]
    error = cx - clean.shape[1] / 2
    turn = KP * error
    left = float(np.clip(BASE_SPEED + turn, -SPEED_LIMIT, SPEED_LIMIT))
    right = float(np.clip(BASE_SPEED - turn, -SPEED_LIMIT, SPEED_LIMIT))
    return left, right


# ==================== PREVIEW ====================
def put_label(img, text, org, color=(255, 255, 255), scale=0.45):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_debug(frame, r, speeds):
    """3 panels: camera + points, coloured areas, cleaned mask that is followed."""
    H, W = frame.shape[:2]
    cam = frame.copy()

    # Areas: blue / red / yellow, black line dark grey, green on top
    area_view = np.full((H, W, 3), 230, np.uint8)
    for a in r["yellow_ids"]:
        area_view[r["labels"] == a] = (0, 220, 255)
    if r["blue_id"] is not None:
        area_view[r["labels"] == r["blue_id"]] = (255, 160, 60)
    if r["red_id"] is not None:
        area_view[r["labels"] == r["red_id"]] = (60, 60, 230)
    area_view[r["black"] > 0] = (50, 50, 50)

    for b in r["blobs"]:
        area_view[b["mask"]] = (0, 200, 0) if b["valid"] else (150, 230, 150)
        cx, cy = int(b["center"][0]), int(b["center"][1])
        # Circle grows with the green's score (like the video)
        cv2.circle(cam, (cx, cy), 2 + b["score"], (0, 255, 0) if b["valid"] else (0, 140, 0), -1)
        put_label(cam, str(b["score"]), (cx + 8, cy), (0, 255, 0))

    if r["entry"] is not None:
        for name, color in (("purple", (255, 0, 200)), ("blue", (255, 0, 0)), ("red", (0, 0, 255))):
            cv2.circle(cam, r["entry"][name], 6, color, -1)
            cv2.circle(area_view, r["entry"][name], 6, color, -1)

    if r["center"] is not None:
        cv2.circle(cam, r["center"], CORNER_CIRCLE_RADIUS, (0, 255, 255), 1)
        cv2.circle(area_view, r["center"], CORNER_CIRCLE_RADIUS, (0, 160, 255), 1)
    for name, color in (("blue", (255, 0, 0)), ("red", (0, 0, 255))):
        for p in r["corners"][name]:
            cv2.circle(cam, p, 5, color, 2)
            cv2.circle(area_view, p, 5, (0, 0, 0), 2)

    clean_view = cv2.cvtColor(r["clean"], cv2.COLOR_GRAY2BGR)
    M = cv2.moments(r["clean"], binaryImage=True)
    if M["m00"] > 0:
        cv2.circle(clean_view, (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])), 6, (0, 255, 0), -1)

    put_label(cam, f"{r['kind']}  areas:{len(r['areas'])}", (6, 16))
    put_label(cam, r["decision"], (6, 34), (0, 255, 255))
    put_label(cam, r["action"] + (f"  L{speeds[0]:+.0f} R{speeds[1]:+.0f}" if speeds else ""), (6, 52))
    put_label(area_view, "areas", (6, 16), (0, 0, 0))
    put_label(clean_view, "line followed", (6, 16))
    return np.hstack([cam, area_view, clean_view])


def show(debug):
    """Show the preview. Returns True if 'q' was pressed."""
    H, W = debug.shape[:2]
    cv2.imshow("Cross test", cv2.resize(debug, (int(W * 1.5), int(H * 1.5))))
    return (cv2.waitKey(1) & 0xFF) == ord("q")


# ==================== RUN ====================
def run_on_image(path):
    """Test the algorithm on a picture: no camera or motors needed.
    Runs it several times so green scores have time to build up."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"Could not read {path}")
        return
    frame = cv2.resize(frame, FRAME_SIZE)
    state = {"tracker": GreenTracker()}
    for _ in range(GREEN_CONFIRM_SCORE // GREEN_SCORE_UP + 1):
        r = process_frame(frame, state)
    print(f"{r['kind']} | areas: {len(r['areas'])} | {r['decision']} | {r['action']}")
    debug = draw_debug(frame, r, steering(r["clean"]))
    out = os.path.splitext(path)[0] + "_result.png"
    cv2.imwrite(out, debug)
    print(f"Saved {out}")
    if SHOW_PREVIEW:
        show(debug)
        cv2.waitKey(0)


def main():
    motors = ServoMotors()
    camera = Camera()
    state = {"tracker": GreenTracker()}
    last_decision = None

    try:
        print("Cross test running. Ctrl+C (or q in the preview) to stop.")
        while True:
            frame = camera.get_camera_view()
            if frame is None:
                continue

            r = process_frame(frame, state)
            if r["decision"] != last_decision:
                print(f"{r['kind']} | areas: {len(r['areas'])} | {r['decision']}")
                last_decision = r["decision"]

            speeds = steering(r["clean"])
            if r["action"] == "UTURN":
                print("U-TURN")
                if DRIVE:
                    motors.set_motors(-UTURN_SPEED, UTURN_SPEED)   # pivot left
                    time.sleep(UTURN_SECONDS)
                    motors.stop_motors()
                state["tracker"].reset()
                state["entry_x"] = None
                continue

            if not DRIVE or speeds is None:
                motors.stop_motors()
            else:
                motors.set_motors(*speeds)

            if SHOW_PREVIEW and show(draw_debug(frame, r, speeds)):
                break

    except KeyboardInterrupt:
        pass
    finally:
        motors.close()
        camera.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_on_image(sys.argv[1])
    else:
        main()

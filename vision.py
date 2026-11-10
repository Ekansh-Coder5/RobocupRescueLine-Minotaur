"""Camera image processing: color masks, line pose, green markers, red stop,
silver detection, and the preview window."""
import os
import numpy as np
import cv2

from config import *

# Set when 'q' is pressed in any preview window; main() checks it and exits.
_quit_requested = False


def quit_requested():
    return _quit_requested


def request_quit():
    """Set the quit flag (used by previews outside this file, e.g. rescue)."""
    global _quit_requested
    _quit_requested = True


# ==================== PREVIEW ====================
def put_label(img, text, org, color=(0, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def show_preview(img, mask_view, green_mask, black_vis, left_pct, right_pct):
    """4-panel preview. black_vis is already BGR. Returns True if 'q' was pressed."""
    W, H = FRAME_SIZE
    green_view = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
    black_view = black_vis
    put_label(img, f"L: {left_pct:+.0f}%  R: {right_pct:+.0f}%", (8, 20))
    put_label(mask_view, "line mask", (8, 20), (255, 255, 255))
    put_label(green_view, "green mask", (8, 20), (255, 255, 255))
    put_label(black_view, "black full", (8, 20), (255, 255, 255))
    combined = np.hstack([img, mask_view, green_view, black_vis])
    cv2.imshow('Line Follower', cv2.resize(combined, (int(W * 4 * 1.5), int(H * 1.5))))
    return (cv2.waitKey(1) & 0xFF) == ord('q')


def live_preview(img, label=""):
    """Show a refreshed 4-panel preview for a frame grabbed in the middle of a green
    routine (pause, align, line search). Works on a copy, so the caller's frame is
    untouched. Sets the global quit flag if 'q' is pressed; returns that flag."""
    global _quit_requested
    if not SHOW_PREVIEW:
        return _quit_requested

    view = img.copy()
    _, _, gmask, bvis = green_check(view, roi_top=0.0)
    mask_view = cv2.cvtColor(black_mask_rgb(img), cv2.COLOR_GRAY2BGR)
    if label:
        put_label(view, label, (8, 45), (0, 165, 255))
    if show_preview(view, mask_view, gmask, bvis, 0, 0):
        _quit_requested = True
    return _quit_requested


# ==================== LINE POSE ====================
def get_contour_pose(contours):
    """Return (cx, cy, angle_deg, vx, vy) of the largest contour, or None.
    angle_deg is relative to vertical: 0 = straight ahead, +deg = tilted right, -deg = tilted left."""
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M['m00'] == 0 or len(largest) < 2:
        return None

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    vx, vy, _, _ = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx[0]), float(vy[0])

    # Point vector upward (toward the top of the frame)
    if vy > 0:
        vx, vy = -vx, -vy

    angle_deg = float(np.degrees(np.arctan2(vx, -vy)))
    return cx, cy, angle_deg, vx, vy


# ==================== COLOR MASKS ====================
def black_mask_rgb(frame, upper_offset=0):
    """Return a 0/255 mask of pixels inside the RGB black range.
    upper_offset is added to UPPER_BLACK (clipped to 255) to loosen the range."""
    # Camera frames are BGR-ordered; convert to RGB so the [R, G, B] limits apply
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    upper = np.clip(UPPER_BLACK + upper_offset, 0, 255)
    return cv2.inRange(rgb, LOWER_BLACK, upper)


def green_mask_hsv(frame):
    """Return a 0/255 mask of pixels inside the HSV green range."""
    # Camera frames are BGR-ordered, so convert BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, LOWER_GREEN_HSV, UPPER_GREEN_HSV)


def black_mask_full(frame, green_raw_mask):
    """Full-frame black mask exclusively for green marker checks (does NOT touch LFR).
    green_raw_mask: Raw HSV green detection (before morphology)"""
    black = black_mask_rgb(frame, upper_offset=GREEN_BLACK_OFFSET)

    # Use raw green mask to exclude exactly the green area, not the morphologically expanded version
    black[green_raw_mask > 0] = 0

    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return black


def detect_red(frame):
    """Returns (found, red_mask). found = at least MIN_RED_PIXELS red pixels
    below RED_ROI_TOP in the frame."""
    # Camera frames are BGR-ordered, so convert BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red is at both ends of the hue scale, so combine the two ranges
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, LOWER_RED1_HSV, UPPER_RED1_HSV),
                              cv2.inRange(hsv, LOWER_RED2_HSV, UPPER_RED2_HSV))
    if RED_ROI_TOP > 0.0:
        red_mask[:int(red_mask.shape[0] * RED_ROI_TOP), :] = 0
    return cv2.countNonZero(red_mask) >= MIN_RED_PIXELS, red_mask


# ==================== GREEN DETECTION ====================
def green_check(frame, roi_top=0.0):
    """Returns (any_left, any_right, green_mask, black_vis).
    Draws on frame and black_vis (BGR).
    Front box and side boxes are counted on the SAME mask that is shown in black_vis.
    roi_top: fraction of frame height (from top) to ignore when detecting green,
             e.g. 0.5 restricts green detection to the bottom 50% of the frame.
             0.0 (default) uses the full frame."""
    fh_img, fw_img = frame.shape[:2]

    # Raw HSV green mask
    green_raw = green_mask_hsv(frame)

    # Green used for detection (and shown in the preview): only the part of the
    # frame below roi_top. The black mask below still uses the FULL raw green mask
    # so green pixels are never mistaken for black anywhere in the frame.
    green_mask = green_raw.copy()
    if roi_top > 0.0:
        green_mask[:int(fh_img * roi_top), :] = 0

    # The one black mask used for BOTH the preview and all pixel counts
    black_mask_green = black_mask_full(frame, green_raw)
    black_vis = cv2.cvtColor(black_mask_green, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    any_left = False
    any_right = False

    def draw_both(p1, p2, color, thick=2):
        cv2.rectangle(frame, p1, p2, color, thick)
        cv2.rectangle(black_vis, p1, p2, color, thick)

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_GREEN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        # ---- front box: check area directly ahead (above) of the green marker ----
        fx0 = max(0, cx - FRONT_BOX_WIDTH // 2)
        fx1 = min(fw_img, cx + FRONT_BOX_WIDTH // 2)
        # Start from top of green marker and extend upward by FRONT_BOX_HEIGHT
        fy0 = max(0, y - FRONT_BOX_HEIGHT)
        fy1 = max(fy0 + 1, y)  # Extend to the top of green marker

        draw_both((fx0, fy0), (fx1, fy1), (0, 255, 255))

        # Count on black_mask_green (same mask as the preview)
        front = black_mask_green[fy0:fy1, fx0:fx1]
        black_px = cv2.countNonZero(front) if front.size else 0
        put_label(frame, f"{black_px}px", (fx0, max(12, fy0 + 12)))
        put_label(black_vis, f"{black_px}px", (fx0, max(12, fy0 + 12)))

        if black_px < MIN_BLACK_PIXELS:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

        # ---- side boxes ----
        sh = int(h * 0.8)
        lx0, lx1 = max(0, x - w), x
        rx0, rx1 = x + w, min(fw_img, x + 2 * w)
        # Count on black_mask_green (same mask as the preview)
        left = black_mask_green[y:y + sh, lx0:lx1]
        right = black_mask_green[y:y + sh, rx0:rx1]
        lb = cv2.countNonZero(left) if left.size else 0
        rb = cv2.countNonZero(right) if right.size else 0
        draw_both((lx0, y), (lx1, y + sh), (255, 255, 0), 1)
        draw_both((rx0, y), (rx1, y + sh), (255, 255, 0), 1)

        if lb > rb:      # black on left -> square is right of line -> turn right
            color, arrow_end, label = (255, 0, 0), (cx + 70, cy), "TURN RIGHT"
            any_right = True
        else:
            color, arrow_end, label = (0, 0, 255), (cx - 70, cy), "TURN LEFT"
            any_left = True

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.arrowedLine(frame, (cx, cy), arrow_end, color, 4)
        cv2.putText(frame, label, (cx - 40, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if any_left and any_right:
        cv2.putText(frame, "U-TURN", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    return any_left, any_right, green_mask, black_vis


# ==================== SILVER DETECTION ====================
# Only called from the silver thread (SilverWatcher in main.py), never from the
# line-following loop. ultralytics/torch are imported lazily so the rest of the
# robot still runs if they are missing.
_silver_model = None


def load_silver_model():
    """Load and warm up silver1.pt. Returns True on success, False otherwise."""
    global _silver_model
    path = next((p for p in SILVER_MODEL_PATHS if os.path.exists(p)), None)
    if path is None:
        print(f"[SILVER] Model not found (tried {SILVER_MODEL_PATHS}); silver detection off.")
        return False
    try:
        import torch
        torch.set_num_threads(SILVER_THREADS)
        from ultralytics import YOLO

        model = YOLO(path)
        W, H = FRAME_SIZE
        model.predict(np.zeros((H, W, 3), np.uint8), imgsz=SILVER_IMGSZ, verbose=False)  # warm-up
    except Exception as e:
        print(f"[SILVER] Could not load model ({e}); silver detection off.")
        return False

    _silver_model = model
    print(f"[SILVER] Model loaded: {path}")
    return True


def detect_silver(frame):
    """Returns (found, best_conf). found = at least one silver detection with
    confidence >= SILVER_CONF whose centre is below SILVER_ROI_TOP."""
    if _silver_model is None:
        return False, 0.0

    # Camera frames are BGR-ordered, which is what ultralytics expects for numpy input
    result = _silver_model.predict(frame, imgsz=SILVER_IMGSZ, conf=SILVER_CONF, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return False, 0.0

    y_min = SILVER_ROI_TOP * frame.shape[0]
    best = 0.0
    for (x1, y1, x2, y2), conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        if (y1 + y2) / 2 >= y_min:
            best = max(best, conf)
    return best > 0.0, best

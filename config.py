"""All tunable settings for the robot. Other files use `from config import *`."""
import os
import numpy as np
from libcamera import Transform

SWITCH_GPIO = 27

_HERE = os.path.dirname(os.path.abspath(__file__))

# ==================== WHEEL SERVOS ====================
# Same IDs as ball_hunt.py. IDs 4 (front grip) and 2 (back arm) are the gripper,
# NOT wheels, so they must never be put in these lists.
LEFT_WHEEL_IDS = (13, 14)
RIGHT_WHEEL_IDS = (11, 12)
REVERSED_WHEEL_IDS = (11, 12)  # right side mounted mirrored -> reversed so + = forward

# ==================== SILVER (evacuation zone entrance) ====================
# Runs in its own thread (see SilverWatcher in main.py) on the latest camera frame.
# silver1.pt = YOLO segmentation model, one class: "silver_line".
SILVER_ENABLED = True
SILVER_MODEL_PATHS = [                      # first one that exists is used
    os.path.join(_HERE, "models", "silver1.pt"),
    os.path.join(_HERE, "..", "models", "silver1.pt"),
]
SILVER_CONF = 0.5              # minimum detection confidence
SILVER_IMGSZ = 320             # inference size; 320 is ~4x faster than the 640 it was trained at
SILVER_ROI_TOP = 0.0           # ignore detections whose centre is above this fraction of frame height
SILVER_CONFIRM_FRAMES = 2      # detections in a row before rescue starts
SILVER_THREADS = 2             # CPU threads torch may use (leaves the rest for line following)
SILVER_MIN_INTERVAL = 0.0      # seconds to wait between inferences (raise to free more CPU)

# ==================== GREEN TURN ANGLES (degrees) ====================
# Gyro turn angle for each green marker. find_line_after_turn() then keeps
# pivoting until the line is centred, so these can be a bit short.
LEFT_TURN_DEG = 65             # green on the left
RIGHT_TURN_DEG = 65            # green on the right
UTURN_DEG = 145                # green on both sides

# ---- Hard-coded turn fallback (when the MPU can't be trusted) ----
# Used instead of the gyro when the ESP32 is not connected, mpu_ok is False, or
# yaw stays exactly the same for GYRO_FROZEN_SECONDS while turning. The robot
# pivots at GYRO_TURN_SPEED for these times, then find_line_after_turn() runs.
# Calibrate: time how long the robot takes to turn each angle above.
HARDCODED_LEFT_TURN_TIME = 0.8     # seconds
HARDCODED_RIGHT_TURN_TIME = 0.8    # seconds
HARDCODED_UTURN_TIME = 1.8         # seconds
GYRO_FROZEN_SECONDS = 2.0          # yaw unchanged this long mid-turn = MPU frozen

# ---- Moves before a green turn (speeds in percent, times in seconds) ----
GREEN_FORWARD_SPEED = 20       # forward speed before turning
LR_FORWARD_TIME = 0.9          # forward time before a left/right turn
UTURN_REVERSE_SPEED = -30      # already negative: reverse
UTURN_REVERSE_TIME = 0.5       # reverse time before a U-turn
UTURN_FORWARD_TIME = 0.6       # forward time after reversing, before the U-turn

# ==================== COLOR RANGES ====================
# Black (line) -- RGB order: [R, G, B]. Black has no hue, so RGB works best.
# Measured (0,0,0)-(70,70,70); upper raised to 90 for glare on the tape.
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([70, 70, 70])

# Green (markers) -- HSV, OpenCV scale: H 0-179, S 0-255, V 0-255
# Measured RGB (44,155,104)-(47,173,123) = HSV H 76-78, S 183-186, V 155-173.
# Hue +-14, S/V minimums lowered for shadow and glare.
LOWER_GREEN_HSV = np.array([62, 100, 60])
UPPER_GREEN_HSV = np.array([92, 255, 255])

# White (not used by any detection yet) -- RGB order
LOWER_WHITE = np.array([190, 200, 200])
UPPER_WHITE = np.array([255, 255, 255])

# Red (stop) -- HSV, same scale as green.
# Measured RGB (151,45,42)-(185,51,44) = HSV H 1, S 184-194, V 151-185.
# Red sits at BOTH ends of the hue scale (just above 0 and just below 179),
# so it needs two ranges that are combined.
LOWER_RED1_HSV = np.array([0, 110, 60])
UPPER_RED1_HSV = np.array([10, 255, 255])
LOWER_RED2_HSV = np.array([170, 110, 60])
UPPER_RED2_HSV = np.array([179, 255, 255])

OBSTACLE_LINE_SEEK_TIMEOUT = 4.0   # seconds; give up and turn anyway after this
OBSTACLE_LINE_MIN_PIXELS = 300     # black pixels needed to count as "line seen"
OBSTACLE_LINE_ROI_TOP = 0.5        # only look below this fraction of frame height (0.5 = bottom half)

OBSTACLE_INVALID_COUNTS_AS_CLEAR = True

SHOW_PREVIEW = bool(os.environ.get("DISPLAY"))   # preview only when a screen is available
SPEED_LIMIT_PERCENT = 50
FRAME_SIZE = (320, 240)
ROI_SHIFT = 0.3
CAMERA_TRANSFORM = Transform(hflip=True, vflip=True)


# ---- Red detection settings ----
MIN_RED_PIXELS = 1500          # red pixels needed (frame is 320x240 = 76,800 px)
RED_ROI_TOP = 0.5              # only look below this fraction of frame height (0 = whole frame)
RED_CONFIRM_FRAMES = 2         # frames in a row with red before stopping (1 = instant)

# ---- Gap handling (full-frame check, triggers driving forward) ----
CENTER_TOLERANCE = 0.20        # max fraction of frame width from center
ANGLE_TOLERANCE_DEG = 25.0     # max tilt from vertical
FWD_SPEED = 28                 # forward speed (-255..255 scale) while seeking the next line

# ---- Green marker settings ----
MIN_GREEN_AREA = 500
GREEN_BLACK_OFFSET = 25        # Added to UPPER_BLACK for glare tolerance ahead of green (90+25 = 115; doesn't touch LFR)
GREEN_FIRST_CHECK_ROI_TOP = 0.3  # the TRIGGER only looks at the bottom 70% of the frame for green
FLUSH_FRAMES = 3               # frames grabbed and thrown away before the full-frame decision check

# ---- Front box (black line check ahead of green), fixed pixels ----
FRONT_BOX_WIDTH = 50
FRONT_BOX_HEIGHT = 70
FRONT_BOX_OVERLAP = 10
MIN_BLACK_PIXELS = 300

TURN_SPEED = 30                # percent
TURN_TIMEOUT = 3.0

# ---- Gyro turn settings ----
GYRO_TURN_SPEED = 25           # Speed for gyro-based turns
GYRO_TOLERANCE = 2.0           # Tolerance in degrees
GYRO_TIMEOUT = 5.0             # Timeout for gyro turns

# ---- Align-to-line settings ----
ALIGN_TOLERANCE_DEG = 4.0      # stop when line is within this angle of vertical
ALIGN_MAX_DEG = 40.0           # ignore lines tilted more than this (likely a junction/branch)
ALIGN_SPEED = 15               # pivot speed (percent)
ALIGN_TIMEOUT = 2.0            # seconds
ALIGN_ROI_TOP = 0.5            # only use the bottom part of the frame (fraction of height from top)

# ---- Green handling ----
GREEN_PAUSE = 4.0              # stop before reacting (was 2.0 slept twice = 4 s)
BACKUP_SPEED = 0
BACKUP_TIME = .6
FIND_CENTER_TOLERANCE = 0.12   # after-turn search stops when line cx is within this fraction of frame width from center

# ---- Obstacle detection (ToF on the ESP32) ----
OBSTACLE_THRESHOLD_MM = 200    # trigger avoidance below this distance
OBSTACLE_REARM_MM = 250        # distance that counts as "clear" for re-arming (keep above the threshold)
OBSTACLE_CLEAR_TIME_SECONDS = 0.5  # must stay clear this long before re-arming
SENSOR_STALE_SECONDS = 1.5     # ESP32 data older than this is treated as stale

# ---- Obstacle double-check (creep forward a little, then re-read the ToF) ----
OBSTACLE_CONFIRM_SPEED = 20        # percent, small forward creep
OBSTACLE_CONFIRM_SECONDS = 0.2     # how long to creep
OBSTACLE_CONFIRM_SETTLE = 0.2      # let the robot/sensor settle before re-reading
OBSTACLE_CONFIRM_WAIT = 1.0        # max seconds to wait for a fresh ESP32 packet

# ---- Obstacle avoidance motor speeds (percent, sent straight to set_motors) ----
OBSTACLE_TURN_SPEED = 30
OBSTACLE_BACKWARD_SPEED = -30      # already negative: pass it as-is
OBSTACLE_FORWARD_SPEED = 28

# ---- Obstacle avoidance movement durations ----
OBSTACLE_BACKWARD_SECONDS = 0.5
OBSTACLE_FORWARD_SHORT_SECONDS = 2
OBSTACLE_FORWARD_LONG_SECONDS = 2.5
OBSTACLE_PAST_LINE_SECONDS = 0.3       # extra forward after the line is seen

# ---- Obstacle avoidance gyro turn settings (defaults) ----
OBSTACLE_LEFT_TURN_ANGLE = 90
OBSTACLE_RIGHT_TURN_ANGLE = 90
OBSTACLE_GYRO_TURN_TIMEOUT = 5.0

# ---- OLED safety display settings ----
OLED_ENABLED = True
OLED_ADDRESS = 0x3C
OLED_WIDTH = 128
OLED_HEIGHT = 64
OLED_REFRESH_SECONDS = 0.1

CAUTION_DISTANCE = 250         # mm (the ToF only reads up to 255, then -1 = nothing in range)
DANGER_DISTANCE = 150          # mm
MAX_SAFE_TILT = 20             # degrees
DANGER_TILT = 35               # degrees


# ==================== SENSOR READER CONFIGURATION ====================
PORT_ESP = "/dev/ttyUSB0"
BAUD_ESP = 115200

# ---- ESP32 auto-reconnect ----
# The reader closes and reopens the port by itself (the same thing as rerunning
# the script) when the port fails to open, throws an error, or data stops.
ESP_READ_TIMEOUT = 0.5             # seconds readline() waits before checking the timers
ESP_FIRST_PACKET_TIMEOUT = 8.0     # after (re)opening, allow this long for the ESP32 to boot and send
ESP_NO_DATA_TIMEOUT = 2.0          # reopen the port if valid packets stop for this long
ESP_RECONNECT_DELAY = 1.0          # wait this long before trying to open the port again
ESP_TRY_OTHER_USB_PORTS = True     # if PORT_ESP fails, also try any other /dev/ttyUSB* (port renamed after a USB drop)

# ---- Sensor-hub format ("DATA KEY=value ...", same keys as wall_follow.py) ----
# robot.py reads this format as well as the "test ..." one.
HUB_FRONT_KEY = "FRONT_L"          # used as distance_mm (obstacles, grabbing, wall ahead)
HUB_LEFT_F_KEY = "LEFT_F"          # left side, front: wall following
HUB_LEFT_B_KEY = "LEFT_B"          # left side, back: wall following


# ==================== RESCUE (evacuation zone) ====================
# Ball detection runs at ~2 fps on the Pi, so rescue is stop-and-look: every move
# is a short pulse, then the robot stops, settles, and checks a fresh frame.
RESCUE_MIN_SPEED = 20              # percent; any wheel speed in rescue is raised to at least this

# ---- Entering the zone ----
RESCUE_ENTER_SPEED = 25            # percent
RESCUE_ENTER_TIME = 2.0            # seconds forward after silver, to get to the middle
RESCUE_ENTER_STOP_MM = 150         # stop the entry drive early if the ToF sees a wall closer than this

# ---- How many balls, in order ----
SILVER_BALLS_TO_RESCUE = 2         # picked first, dropped in the GREEN box
BLACK_BALLS_TO_RESCUE = 1          # picked after the silver ones, dropped in the RED box

# ---- USB ball camera ----
BALL_CAMERA = "auto"               # "auto" = first USB camera in /dev/v4l/by-id, or an index (0) / path ("/dev/video8")
BALL_FRAME_SIZE = (640, 480)
BALL_CAMERA_FLIP = -1              # None, 0 (vertical), 1 (mirror), -1 (rotate 180), same as ball_hunt.py

# ---- Ball model (same as ball_hunt.py: YOLOv8 ONNX through OpenCV DNN) ----
BALL_MODEL_PATHS = [
    os.path.join(_HERE, "models", "balls.onnx"),
    os.path.join(_HERE, "..", "models", "balls.onnx"),
]
BALL_CLASS_NAMES = ["Black Ball", "Silver Ball"]   # training order
BALL_IMGSZ = 640                   # the exported model is fixed at 640x640
BALL_CONF = 0.7
BALL_NMS_IOU = 0.45

# ---- Stop-and-look ----
LOOK_SETTLE_TIME = 0.15            # seconds to wait after stopping before taking the frame
TARGET_CONFIRM_LOOKS = 2           # looks in a row that must see the target before it counts as found
LOST_LOOKS = 3                     # looks in a row without the target before searching again

# ---- Search: turn in steps, look after each step ----
SEARCH_DIRECTION = "right"
SEARCH_TURN_SPEED = 25             # percent
SEARCH_TURN_TIME = 0.25            # seconds per step
SEARCH_ROTATIONS = 2               # full turns with nothing found -> move somewhere else
SEARCH_DEG_PER_STEP = 20           # degrees counted per step when the gyro can't be used

# ---- Move somewhere else (after SEARCH_ROTATIONS with nothing found) ----
WANDER_MIN_DEG = 60                # random turn between these angles, random side
WANDER_MAX_DEG = 180
WANDER_SPEED = 25                  # percent
WANDER_TIME = 1.5                  # seconds forward after the turn
WANDER_STOP_MM = 150               # stop driving early if a wall is closer than this
RESCUE_TURN_DEG_PER_SEC = 90       # used to time the random turn when the gyro is unavailable

# ---- Centring (offset is -1 = far left .. +1 = far right) ----
BALL_CENTER_TOLERANCE = 0.12       # ball counts as centred inside this offset
BALL_ALIGN_SPEED = 20              # percent, pivot speed for centring pulses
ALIGN_SECONDS_PER_OFFSET = 0.35    # pivot time per 1.0 of offset
ALIGN_PULSE_MIN = 0.06             # seconds
ALIGN_PULSE_MAX = 0.30             # seconds

# ---- Approaching the ball ----
APPROACH_SPEED = 25                # percent
APPROACH_PULSE_MAX = 0.6           # seconds forward when the ball is far (top of frame)
APPROACH_PULSE_MIN = 0.2           # seconds forward when the ball is close
BALL_CLOSE_FRAC = 0.70             # ball's bottom edge below this fraction of frame height = close
BLIND_SPEED = 20                   # percent, creep when the ball has gone under the camera
BLIND_TIME = 2.0                   # seconds of blind creeping before giving up

# ---- Grabbing (ToF = ESP32 distance_mm) ----
GRAB_MM = 70                       # grab when the ToF reads below this...
GRAB_CONFIRM_READS = 3             # ...for this many fresh readings in a row
PICK_CONFIRM_MM = 70               # after lifting: picked if every reading is invalid or above this
PICK_CONFIRM_READS = 3             # fresh readings checked after lifting
PICK_CHECK_SETTLE = 0.3            # seconds to wait after lifting before checking
PICK_CHECK_TIMEOUT = 1.0           # max seconds to wait for those readings
GRAB_BACKUP_SPEED = 30             # percent, reverse after the pick check
GRAB_BACKUP_TIME = 0.8             # seconds

# ---- Arm (position-mode servos on the wheel bus, values from ball_hunt.py) ----
FRONT_GRIP_ID = 4
BACK_ARM_ID = 2
ARM_SPEED = 300                    # servo speed limit; below ~150 the SC15 stalls
GRIP_OPEN, GRIP_CLOSED = 0, 150
ARM_DOWN, ARM_UP = 150, 600
ARM_DROP_FRACTION = 0.5            # deposit height: 0 = up, 1 = fully down (0.5 = halfway)
GRIP_SQUEEZE = 15                  # when the grip stalls on the ball, hold this far past contact
ARM_POS_TOL = 8                    # "arrived" when this close to the target
ARM_MOVE_TIMEOUT = 4.0
ARM_POLL = 0.03
ARM_STALL_TIME = 0.25              # no progress for this long = blocked (holding the ball)
ARM_SETTLE = 0.15

# ---- Evacuation boxes (USB camera, HSV, OpenCV scale) ----
# Start from the line-tape colours; re-measure on the real boxes with the USB camera.
EVAC_GREEN_LOWER = np.array([62, 100, 60])
EVAC_GREEN_UPPER = np.array([92, 255, 255])
EVAC_RED1_LOWER = np.array([0, 110, 60])
EVAC_RED1_UPPER = np.array([10, 255, 255])
EVAC_RED2_LOWER = np.array([170, 110, 60])
EVAC_RED2_UPPER = np.array([179, 255, 255])
EVAC_MIN_AREA_FRAC = 0.01          # blob must cover at least this fraction of the frame
EVAC_MIN_HEIGHT_FRAC = 0.05        # and be at least this tall (ignores thin floor tape)
EVAC_CENTER_TOLERANCE = 0.15
EVAC_STOP_AREA_FRAC = 0.30         # box this big in the frame = arrived
EVAC_STOP_MM = 100                 # or the ToF reads closer than this = arrived
EVAC_APPROACH_PULSE = 0.4          # seconds forward per step
DEPOSIT_RELEASE_TIME = 0.5         # seconds with the grip open before lifting the arm
DEPOSIT_BACKUP_SPEED = 30          # percent
DEPOSIT_BACKUP_TIME = 0.8          # seconds

# ---- Failed picks ----
PICK_MAX_ATTEMPTS = 3              # failed picks in a row -> turn PICK_FAIL_TURN_DEG, then scan again
PICK_FAIL_TURN_DEG = 90
PICK_MAX_ROUNDS = 2                # that happens this many times (2 x 3 = 6 misses) -> skip the ball

# ---- Wrong ball in the way (the ToF can't tell silver from black) ----
BLOCK_PATH_OFFSET = 0.35           # other-kind ball closer than the target and within this offset = in the way
BLOCK_TURN_DEG = 45                # back off, turn this much away from it...
BLOCK_SIDESTEP_TIME = 0.8          # ...drive forward this long, then search for the target again

# ---- Wall following (after the last black ball, wall on the LEFT) ----
WALL_START_TURN_DEG = 90           # after backing off the red box: turn right this much, then follow
WALL_SPEED = 30                    # percent, base forward speed
WALL_TARGET_MM = 60                # distance to keep from the left wall
WALL_SEE_MAX_MM = 250              # left readings further than this count as "nothing" (gap / no wall);
                                   # the ToF reads up to 255 mm, then -1
WALL_KP_DIST = 0.2                 # % steer per mm of distance error
WALL_KP_ANGLE = 0.3                # % steer per mm of (LEFT_F - LEFT_B)
WALL_MAX_STEER = 10                # % (keeps both wheels at or above RESCUE_MIN_SPEED)
WALL_LOST_LEFT_SPEED = 20          # no wall at all: left wheel this, right WALL_SPEED (curve left)...
WALL_LOST_ARC_TIME = 2.0           # ...for at most this long, then straight until a wall is ahead
                                   # (stops it circling forever in the open middle of the zone)
WALL_FRONT_STOP_MM = 80            # wall ahead closer than this...
WALL_FRONT_CONFIRM = 2             # ...for this many readings -> corner turn
WALL_CORNER_TURN_DEG = 60          # right, away from the left wall (same as wall_follow.py)
WALL_GAP_CONFIRM = 3               # LEFT_F readings in a row with nothing = gap in the wall
WALL_GAP_FWD_TIME = 0.5            # seconds forward before checking LEFT_B (and before turning into an exit)
WALL_BACKUP_TIME = 0.5             # seconds back after seeing silver (the entrance)
ENTRANCE_TURN_DEG = 0              # turn right this much after backing up from the entrance.
                                   # 0 = keep going straight past it. 90 skips the wall just after
                                   # the entrance, so an exit there is never found (loops forever)
ENTRANCE_PASS_TIME = 1.8           # seconds forward to get past the entrance (covers the back-up too)
WALL_PASS_TIME = 1.0               # seconds forward after turning back from a gap that wasn't the exit
WALL_FOLLOW_TIMEOUT = 120          # seconds; give up (stop) if no exit is found

# ---- Exit check (bottom / line camera) ----
# Silver = the entrance (wrong exit). Black line = the correct exit. No green at the exit.
FLOOR_ROI_TOP = 0.4                # only look below this fraction of frame height
FLOOR_CHECK_FRAMES = 3             # frames checked; silver in ANY frame = not the exit
EXIT_BLACK_MIN_PIXELS = 400        # black pixels in the ROI = line outside the exit
EXIT_DRIVE_TIME = 1.0              # seconds forward through the exit before line following takes over

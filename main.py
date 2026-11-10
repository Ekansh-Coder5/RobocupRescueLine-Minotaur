"""Main program: starts the hardware and background threads, then runs the
state machine.

States:
  LINE      line following (line.py), with the silver thread watching for the
            evacuation zone entrance
  RED_STOP  red seen: stopped until the motor switch is cycled OFF -> ON
  RESCUE    silver seen: runs rescue.py (balls, then wall following to the exit)
            exit found -> back to LINE (silver detection stays off)
  DONE      rescue stopped without finding the exit: stopped

Threads (all daemon threads):
  ESP32 reader     robot.start_sensor_thread()   ToF + MPU over serial
  OLED display     SafetyDisplay                 status + sensor readout
  Silver watcher   SilverWatcher (below)         YOLO silver detection

Run this file: python main.py"""
import threading
import time

import cv2

from config import *
from robot import (
    ServoMotors, Camera, SafetyDisplay,
    start_sensor_thread, stop_sensor_thread, set_status,
)
from vision import quit_requested, live_preview, load_silver_model, detect_silver
from line import LineFollower, EVENT_QUIT, EVENT_RED
from rescue import run_rescue


STATE_LINE = "LINE"
STATE_RED_STOP = "RED_STOP"
STATE_RESCUE = "RESCUE"
STATE_DONE = "DONE"


# ==================== SILVER WATCHER THREAD ====================
class SilverWatcher:
    """Runs silver detection on its own thread so the slow YOLO model never
    slows the line-following loop. It never touches the camera: it reads the
    newest frame the main loop already captured (Camera.latest_frame()), and
    skips frames it has already checked.
    found() turns True after SILVER_CONFIRM_FRAMES detections in a row."""

    def __init__(self, camera):
        self.camera = camera
        self._found = threading.Event()
        self._active = threading.Event()     # set = detection running, clear = paused
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not SILVER_ENABLED:
            print("[SILVER] Disabled in config.")
            return
        self._active.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def found(self):
        return self._found.is_set()

    def pause(self):
        """Stop running the model (frees the CPU, e.g. during rescue)."""
        self._active.clear()

    def clear(self):
        """Forget a detection (the thread stays paused or running as it was)."""
        self._found.clear()

    def resume(self):
        """Clear any old detection and start watching again."""
        self._found.clear()
        self._active.set()

    def stop(self):
        self._stop.set()
        self._active.set()   # wake the thread if it is paused
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self):
        # Loading torch + the model is slow, so it happens here, not in main()
        if not load_silver_model():
            return

        last_seq = None
        hits = 0
        error_reported = False

        while not self._stop.is_set():
            if not self._active.wait(timeout=0.1):
                hits = 0
                continue

            frame, seq = self.camera.latest_frame()
            if frame is None or seq == last_seq:
                time.sleep(0.005)    # no new frame yet
                continue
            last_seq = seq

            try:
                seen, conf = detect_silver(frame)
            except Exception as e:
                if not error_reported:
                    print(f"[SILVER] Detection error: {e}")
                    error_reported = True
                continue

            hits = hits + 1 if seen else 0
            if hits >= SILVER_CONFIRM_FRAMES and not self._found.is_set():
                print(f"[SILVER] Silver confirmed (conf {conf:.2f}).")
                self._found.set()

            if SILVER_MIN_INTERVAL > 0:
                self._stop.wait(SILVER_MIN_INTERVAL)


# ==================== MAIN ====================
def main():
    motors = ServoMotors()
    camera = Camera()
    oled = SafetyDisplay()
    silver = SilverWatcher(camera)
    follower = LineFollower(motors, camera)

    # Background threads for the whole run
    start_sensor_thread()
    oled.start()
    silver.start()

    state = STATE_LINE
    set_status("LFR")
    red_switch_was_off = False

    try:
        print(
            f"ToF obstacle detection active "
            f"(< {OBSTACLE_THRESHOLD_MM} mm). Press Ctrl+C to stop."
        )

        while not quit_requested():

            # ---------------- LINE ----------------
            if state == STATE_LINE:
                if silver.found():
                    motors.stop_motors()
                    print("SILVER: starting rescue.")
                    state = STATE_RESCUE
                    continue

                event = follower.step()
                if event == EVENT_QUIT:
                    break
                if event == EVENT_RED:
                    print("RED DETECTED: robot stopped. Flip the motor switch OFF and ON to continue.")
                    red_switch_was_off = False
                    state = STATE_RED_STOP

            # ---------------- RED_STOP ----------------
            elif state == STATE_RED_STOP:
                motors.stop_motors()
                set_status("STOPPED (RED)")

                # Motor switch OFF then ON = continue (move the robot off the red first)
                if not motors.motor_switch.is_pressed:
                    red_switch_was_off = True
                elif red_switch_was_off:
                    print("Motor switch cycled: leaving red stop.")
                    follower.reset()
                    state = STATE_LINE
                    continue

                frame = camera.get_camera_view()
                if frame is not None and live_preview(frame, "RED - STOPPED"):
                    break
                time.sleep(0.02)

            # ---------------- RESCUE ----------------
            elif state == STATE_RESCUE:
                silver.pause()           # free the CPU for rescue
                set_status("RESCUE")
                exited = run_rescue(motors, camera)
                motors.stop_motors()
                # Only one evacuation zone: silver detection stays paused from here on
                silver.clear()
                if exited:
                    print("Rescue finished: out of the zone, back to line following.")
                    follower.reset()
                    set_status("LFR")
                    state = STATE_LINE
                else:
                    print("Rescue stopped without leaving the zone.")
                    state = STATE_DONE

            # ---------------- DONE ----------------
            elif state == STATE_DONE:
                motors.stop_motors()
                set_status("DONE")
                frame = camera.get_camera_view()
                if frame is not None and live_preview(frame, "DONE"):
                    break
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        silver.stop()
        oled.stop()
        stop_sensor_thread()
        motors.close()
        camera.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

import os
import cv2
import time
from typing import Dict, Optional, Tuple, List

import mediapipe as mp

from vision import detect_piano_region, detect_piano_keys 
from pianodata import PianoData
from pianosynth import PianoLikeSynth

from touch import (
    _pushd,
    _open_capture,
    _landmark_to_px,
    _velocity_to_gain,
    GOING_UP, IDLE, GOING_DOWN, TOUCHING,
    TIP_IDS,
    FingerTrack,
    SmoothedTrack,
    NoteRefCounter,
    update_automaton_state,
)

# MAIN 

def main():
    # camera - change the parameters if you want to use a different camera
    cam_src = os.environ.get("CAM_SRC", "1") # or "0", "2", ecc...
    output_dir = os.environ.get("OUTPUT_DIR", "output")

    cap = _open_capture(cam_src)
    if not cap.isOpened():
        print(f"[ERROR] Impossible to open camera CAM_SRC={cam_src}")
        print("Try CAM_SRC=1,2,3...")
        return

    target_w, target_h = 720, 480

    # synth
    synth = PianoLikeSynth(volume=0.30, brightness=0.9, hammer_amount=0.35)
    note_bus = NoteRefCounter(synth)

    # mediapipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    pianodata: Optional[PianoData] = None
    force_redetect = True

    tracks: Dict[str, FingerTrack] = {}
    smoothers: Dict[str, SmoothedTrack] = {}

    # params
    T_VEL = 30.0                # threshold t (px/s) del diagramma
    TRIGGER_COOLDOWN_S = 0.08    # evita doppio colpo
    RELEASE_CONFIRM_FRAMES = 2   # anti-jitter release 
    V_FOR_GAIN_MIN = 180.0
    V_FOR_GAIN_MAX = 1400.0

    print("Commands: q=quit | r=re-detect | s=snapshot")

    # main loop
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (target_w, target_h))
            show = frame.copy()
            now = time.time()

            # 1) detect keyboard
            if force_redetect:
                with _pushd(output_dir):
                    box, piano_region = detect_piano_region(frame)
                    if box is not None and piano_region is not None:
                        try:
                            white_keys, black_keys, piano = detect_piano_keys(piano_region, prefix="live_")
                            pianodata = PianoData(white_keys, black_keys, piano, box)
                            pianodata.create_overlay(frame)
                            print(f"[OK] pianodata ready box={box} (debug in ./{output_dir})")
                        except Exception as e:
                            print("[ERROR] detect_piano_keys / PianoData:", e)
                            pianodata = None
                    else:
                        print("[INFO] keyboard not found (box None)")
                        pianodata = None

                note_bus.all_notes_off()
                tracks.clear()
                smoothers.clear()
                force_redetect = False

            # 2) overlay
            if pianodata is not None:
                pianodata.draw_overlay(show)

            # 3) touch detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            notes_xy: List[Tuple[float, float, str]] = []  
            seen_fingers = set()

            if res.multi_hand_landmarks and pianodata is not None:
                h, w = frame.shape[:2]

                # handedness list allined
                handed_list = []
                if res.multi_handedness:
                    for hd in res.multi_handedness:
                        try:
                            handed_list.append(hd.classification[0].label)  # "Left"/"Right"
                        except Exception:
                            handed_list.append("Unknown")
                else:
                    handed_list = ["Unknown"] * len(res.multi_hand_landmarks)

                for hand_idx, hand_lms in enumerate(res.multi_hand_landmarks):
                    handed = handed_list[hand_idx] if hand_idx < len(handed_list) else "Unknown"
                    handed_short = "L" if handed.lower().startswith("l") else "R" if handed.lower().startswith("r") else "U"

                    mp_draw.draw_landmarks(show, hand_lms, mp_hands.HAND_CONNECTIONS)

                    for tid in TIP_IDS:
                        lm = hand_lms.landmark[tid]
                        px, py = _landmark_to_px(lm, w, h)

                        finger_id = f"{handed_short}-{tid}"
                        seen_fingers.add(finger_id)

                        tr = tracks.get(finger_id)
                        if tr is None:
                            tr = FingerTrack(finger_id=finger_id, last_pos=(px, py), last_t=now, prev_sy=None, prev_t=now, state=IDLE)
                            tracks[finger_id] = tr

                        sm = smoothers.get(finger_id)
                        if sm is None:
                            sm = SmoothedTrack(new_frame_weigth=0.25, discard_th=200, reset_after_n_discarded=5)
                            smoothers[finger_id] = sm

                        prev_sm_state = sm.state  # last smoothed state

                        # update position
                        tr.last_pos = (px, py)
                        tr.last_t = now
                        sm.update(tr)

                        sx = sm.pos_x if sm.pos_x is not None else float(px)
                        sy = sm.pos_y if sm.pos_y is not None else float(py)

                        if tr.prev_sy is None:
                            tr.prev_sy = sy
                            tr.prev_t = now
                            tr.v = 0.0
                        else:
                            dt = max(1e-3, now - tr.prev_t)
                            tr.v = (sy - tr.prev_sy) / dt
                            tr.prev_sy = sy
                            tr.prev_t = now

                        # memorize last strong descent
                        if tr.v > T_VEL:
                            tr.last_down_v = max(tr.last_down_v, tr.v)

                        # update automa 
                        raw_next = update_automaton_state(tr.state, tr.v, T_VEL)
                        tr.state = raw_next

                        # update smoothing 
                        sm.update_state_only(raw_next)
                        sm_state = sm.state  

                        if sm_state == TOUCHING:
                            notes_xy.append((sx, sy, finger_id))

                        # drawing
                        cv2.circle(show, (int(round(sx)), int(round(sy))), 6, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(show, (int(round(sx)), int(round(sy))), 6, (0, 0, 0), 2, cv2.LINE_AA)
                        label = f"{finger_id} v={tr.v:+.0f} raw={raw_next} sm={sm_state}"
                        if tr.pressed_note:
                            label += f" [{tr.pressed_note}]"
                        cv2.putText(
                            show, label,
                            (int(round(sx)) + 10, int(round(sy)) + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (255, 255, 255), 2, cv2.LINE_AA
                        )

                        # trigger audio only in TOUCHING
                        entered_touching = (prev_sm_state != TOUCHING and sm_state == TOUCHING)
                        can_trigger = (now - tr.last_trigger_t) >= TRIGGER_COOLDOWN_S

                        if entered_touching and can_trigger and pianodata is not None:
                            note, idx, is_black = pianodata.point_to_note((sx, sy))
                            if note:
                                gain = _velocity_to_gain(max(tr.last_down_v, abs(tr.v)), V_FOR_GAIN_MIN, V_FOR_GAIN_MAX)
                                note_bus.note_on(note, gain)
                                tr.pressed_note = note
                                tr.last_trigger_t = now
                                tr.last_down_v = 0.0  

                            if note:
                                pianodata.draw_point_note(show, (sx, sy), idx, is_black, note_name=note)

                        # release
                        note_under = None
                        if pianodata is not None:
                            note_under, _, _ = pianodata.point_to_note((sx, sy))

                        if note_under is None:
                            tr.pressed_note = tr.pressed_note  # keep
                            # counter
                            off = getattr(tr, "_off_frames", 0) + 1
                            setattr(tr, "_off_frames", off)
                        else:
                            setattr(tr, "_off_frames", 0)

                        if tr.pressed_note is not None and getattr(tr, "_off_frames", 0) >= RELEASE_CONFIRM_FRAMES:
                            note_bus.note_off(tr.pressed_note)
                            tr.pressed_note = None
                            setattr(tr, "_off_frames", 0)

            # if a finger disappears: release note
            for fid, tr in list(tracks.items()):
                if fid not in seen_fingers:
                    missing = getattr(tr, "_missing", 0) + 1
                    setattr(tr, "_missing", missing)
                    if tr.pressed_note is not None and missing >= RELEASE_CONFIRM_FRAMES:
                        note_bus.note_off(tr.pressed_note)
                        tr.pressed_note = None
                    if missing > 20:
                        tracks.pop(fid, None)
                        smoothers.pop(fid, None)
                else:
                    setattr(tr, "_missing", 0)

            cv2.putText(show, f"touching points (notes set): {len(notes_xy)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(show, f"active notes: {len(note_bus.counts)}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(show, "q=quit | r=re-detect | s=snapshot", (10, show.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("PaperPiano (Automaton Touching)", show)
            k = cv2.waitKey(1) & 0xFF

            if k == ord("q"):
                break
            elif k == ord("r"):
                force_redetect = True
            elif k == ord("s"):
                os.makedirs(output_dir, exist_ok=True)
                ts = int(time.time())
                out = os.path.join(output_dir, f"snapshot_automaton_{ts}.png")
                cv2.imwrite(out, show)
                print("[OK] salvato", out)

    finally:
        note_bus.all_notes_off()
        try:
            hands.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        synth.close()


if __name__ == "__main__":
    main()

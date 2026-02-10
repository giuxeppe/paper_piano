import os
import cv2
import time
import math
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, List

# utils
@contextmanager
def _pushd(path: str):
    old = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _open_capture(src: str):
    try:
        idx = int(src)
        return cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    except ValueError:
        return cv2.VideoCapture(src)


def _landmark_to_px(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))


def _clamp(x, a, b):
    return a if x < a else b if x > b else x


def _velocity_to_gain(v_px_s: float, v_min: float, v_max: float) -> float:
    # gain 0.55..1.0
    t = (v_px_s - v_min) / max(1e-6, (v_max - v_min))
    t = _clamp(t, 0.0, 1.0)
    return 0.55 + 0.45 * t


# states 
GOING_UP = "going_up"
IDLE = "idle"
GOING_DOWN = "going_down"
TOUCHING = "touching"

TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky


@dataclass
class FingerTrack:
    finger_id: str
    last_pos: Optional[Tuple[int, int]] = None
    last_t: float = 0.0

    # automaton state (raw)
    state: str = IDLE

    # for velocity estimation (we will use smoothed y)
    prev_sy: Optional[float] = None
    prev_t: float = 0.0
    v: float = 0.0  # vertical velocity px/s (positive = down)

    # audio
    pressed_note: Optional[str] = None
    last_trigger_t: float = 0.0
    last_down_v: float = 0.0  # store last big down velocity to compute gain


class SmoothedTrack:

    def __init__(self, new_frame_weigth=0.45, discard_th=260, reset_after_n_discarded=5):
        self.new_frame_weigth = new_frame_weigth
        self.discard_th = discard_th
        self.reset_after_n_discarded = reset_after_n_discarded

        self.pos_x = None
        self.pos_y = None
        self.num_discards = 0

        self.default_state = {
            GOING_UP: 0.25,
            IDLE: 0.25,
            GOING_DOWN: 0.25,
            TOUCHING: 0.25,
        }
        self.state_probs = self.default_state.copy()
        self.state = IDLE

    def reset(self):
        self.pos_x = None
        self.pos_y = None
        self.num_discards = 0
        self.state_probs = self.default_state.copy()
        self.state = IDLE

    def update(self, track: FingerTrack):
        if self.num_discards >= self.reset_after_n_discarded:
            self.reset()

        if track.last_pos is None:
            return

        x, y = track.last_pos[0], track.last_pos[1]

        if self.pos_x is None or self.pos_y is None:
            self.pos_x = float(x)
            self.pos_y = float(y)
        else:
            if max(abs(self.pos_x - x), abs(self.pos_y - y)) > self.discard_th:
                self.num_discards += 1
                return

            a = float(self.new_frame_weigth)
            self.pos_x = self.pos_x * (1 - a) + a * float(x)
            self.pos_y = self.pos_y * (1 - a) + a * float(y)

        a = float(self.new_frame_weigth)
        max_prob = -1.0
        best = self.state
        for key in self.state_probs:
            increment = 1.0 if track.state == key else 0.0
            self.state_probs[key] = self.state_probs[key] * (1 - a) + a * increment
            if self.state_probs[key] > max_prob:
                best = key
                max_prob = self.state_probs[key]
        self.state = best

    def update_state_only(self, state: str):
        a = float(self.new_frame_weigth)
        max_prob = -1.0
        best = self.state
        for key in self.state_probs:
            increment = 1.0 if state == key else 0.0
            self.state_probs[key] = self.state_probs[key] * (1 - a) + a * increment
            if self.state_probs[key] > max_prob:
                best = key
                max_prob = self.state_probs[key]
        self.state = best


class NoteRefCounter:
    def __init__(self, synth):
        self.synth = synth
        self.counts: Dict[str, int] = {}

    def note_on(self, note: str, velocity: float):
        c = self.counts.get(note, 0)
        if c == 0:
            self.synth.note_on(note, velocity)
        self.counts[note] = c + 1

    def note_off(self, note: str):
        c = self.counts.get(note, 0)
        if c <= 1:
            if c == 1:
                self.synth.note_off(note)
            self.counts.pop(note, None)
        else:
            self.counts[note] = c - 1

    def all_notes_off(self):
        for n in list(self.counts.keys()):
            self.synth.note_off(n)
        self.counts.clear()


# automaton update 
def update_automaton_state(prev_state: str, v: float, t: float) -> str:
    """
      - v < -t  => going_up
      - -t <= v <= t => idle (or touching from going_down)
      - v > t => going_down
    """
    if prev_state == GOING_UP:
        if v < -t:
            return GOING_UP
        else:
            # -t <= v <= t 
            return IDLE

    if prev_state == IDLE:
        if v > t:
            return GOING_DOWN
        if v < -t:
            return GOING_UP
        return IDLE

    if prev_state == GOING_DOWN:
        if v > t:
            return GOING_DOWN
        # -t <= v <= t
        return TOUCHING

    if prev_state == TOUCHING:
        if v < -t:
            return GOING_UP
        return TOUCHING

    return IDLE

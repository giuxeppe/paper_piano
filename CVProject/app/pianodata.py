import numpy as np
from vision import detect_piano_keys, draw_quads
import os
import time
import sys
import cv2

# building our virtual piano

WHITE_CYCLE = ["C", "D", "E", "F", "G", "A", "B"]
# black-present pattern across white gaps for a cycle starting at C:
# gaps: C-D, D-E, E-F, F-G, G-A, A-B, B-C
GAP_HAS_BLACK_C = [1, 1, 0, 1, 1, 1, 0]

BLACK_NAME_BETWEEN = {
    ("C", "D"): "C#",
    ("D", "E"): "D#",
    ("F", "G"): "F#",
    ("G", "A"): "G#",
    ("A", "B"): "A#",
}

def _point_in_poly(point, quad) -> bool:
    """
    point: (x,y)
    quad: (4,2) array-like
    """
    x, y = float(point[0]), float(point[1])
    poly = np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2)
    # >= 0 means inside or on edge
    return cv2.pointPolygonTest(poly, (x, y), False) >= 0

def quad_bottom_midpoint_xy(quad: np.ndarray):
    """
    quad: (4,2)
    Returns (x,y) mean of the two bottommost vertices
    """
    q = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    idx = np.argsort(-q[:, 1])  
    p1, p2 = q[idx[0]], q[idx[1]]
    return float((p1[0] + p2[0]) * 0.5), float((p1[1] + p2[1]) * 0.5)

def put_label(vis, text, x, y, color, scale=0.55, thickness=2):
    """
    Draw outlined text for readability.
    """
    x = int(round(x))
    y = int(round(y))

    # outline (black)
    cv2.putText(vis, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    # main text
    cv2.putText(vis, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)
    
def quad_bottom_midpoint(quad: np.ndarray):
    """
    quad: (4,2) float32 (or any array-like shape (4,2))
    Returns (x,y) = mean of the two bottommost vertices (largest y).
    """
    q = np.asarray(quad, dtype=np.float32).reshape(-1, 2)

    # sort by y descending (bottom first)
    idx = np.argsort(-q[:, 1])
    p1 = q[idx[0]]
    p2 = q[idx[1]]

    return float((p1[0] + p2[0]) * 0.5), float((p1[1] + p2[1]) * 0.5)


def draw_bottom_points(vis, quads, color=(255, 255, 0), radius=5, label_prefix=None):
    """
    Draws bottom-midpoint for each quad.
    """
    for i, q in enumerate(quads):
        x, y = quad_bottom_midpoint(q)
        p = (int(round(x)), int(round(y)))

        # draw point
        cv2.circle(vis, p, radius, color, -1)

        # optional index label
        if label_prefix is not None:
            cv2.putText(
                vis,
                f"{label_prefix}{i}",
                (p[0] + 6, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

def quad_top_midpoint(quad: np.ndarray):
    """
    quad: (4,2) float32
    Returns (x,y) = mean of the two topmost vertices
    """
    q = np.asarray(quad, dtype=np.float32).reshape(-1, 2)

    # sort points by y (ascending = top first)
    idx = np.argsort(q[:, 1])
    p1 = q[idx[0]]
    p2 = q[idx[1]]

    return float((p1[0] + p2[0]) * 0.5)

def _quad_center_x(quad: np.ndarray) -> float:
    return quad_bottom_midpoint(quad)

def _sort_quads_left_to_right(quads):
    return sorted(quads, key=_quad_center_x)

def _infer_white_start_index(white_xs, black_xs):
    """
    Returns start index s in WHITE_CYCLE such that white[0] is WHITE_CYCLE[s].
    Uses best match between observed black-in-gap pattern and cyclic shifts.
    """
    nW = len(white_xs)
    if nW < 2:
        return 0  # default C if too little info

    black_xs = np.asarray(black_xs, dtype=np.float32)
    
    obs = []
    for i in range(nW - 1):
        L, R = white_xs[i], white_xs[i + 1]
        if L > R:
            L, R = R, L
        has_black = bool(np.any((black_xs > L) & (black_xs < R))) if len(black_xs) else False
        obs.append(1 if has_black else 0)
    print(white_xs)
    print(black_xs)
    print(obs)

    best_s = 0
    best_score = -1
    for s in range(7):
        ideal = [GAP_HAS_BLACK_C[(s + i) % 7] for i in range(len(obs))]
        score = sum(int(a == b) for a, b in zip(obs, ideal))
        if score > best_score:
            best_score = score
            best_s = s
    return best_s

def _extract_octave_number(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 0

def translate_coordlist(coordlist, box: tuple):
    x0, y0, x1, y1 = box
    for i, coord in enumerate(coordlist):
      
        coordlist[i] = (coord[0] + x0, coord[1] +y0)
    return coordlist

class PianoData:
    def __init__(self, white_keys, black_keys, piano, box):
       
        for key in white_keys: 
            key = translate_coordlist(key, box)
        for key in black_keys: 
            key = translate_coordlist(key, box)
        self.piano = translate_coordlist(piano, box)
        self.white_keys = white_keys
        self.black_keys = black_keys
        self.white_key_names = []
        self.black_key_names = []
        self.mask = None
        self.overlay = None
        self._assign_note_names()
        print(self.white_key_names)
        print(self.black_key_names)

    def _assign_note_names(self, debug: bool = False):
        """
        assigns notes in the opposite direction (right->left musical order)
        self.white_keys / self.black_keys remain in original coordinate system.
        """
        base_octave = 4
        if len(self.white_keys) > 10:
            base_octave = 3
        
        if len(self.white_keys) > 24:
            base_octave = 2

        if len(self.white_keys) > 38:
            base_octave = 1

        if len(self.white_keys) > 52:
            base_octave = 0
        
        # 1) sort by x in image space (left -> right)
        white_sorted_lr = _sort_quads_left_to_right(self.white_keys)
        black_sorted_lr = _sort_quads_left_to_right(self.black_keys)

        # Use the same x-measure you already chose in _quad_center_x (currently top midpoint)
        white_xs_lr = [_quad_center_x(q) for q in white_sorted_lr]
        black_xs_lr = [_quad_center_x(q) for q in black_sorted_lr]

        # 2) REVERSE musical direction (right -> left)
        white_sorted = list(reversed(white_sorted_lr))
        white_xs = list(reversed(white_xs_lr))

        # black keys we keep sorted left->right geometrically, but we will test gaps against the reversed white_xs (musical order)
        black_sorted = black_sorted_lr

        # 3) infer starting white note index using the reversed direction 
        start = _infer_white_start_index(white_xs, black_xs_lr)
        if debug:
            print("FORCED reversed direction")
            print("Start white note:", WHITE_CYCLE[start])

        # 4) assign white names (in reversed order), with octave increments at B->C
        white_names_by_id = {}
        octave = int(base_octave)
        prev = None

        for i, q in enumerate(white_sorted):
            base = WHITE_CYCLE[(start + i) % 7]
            if prev == "B" and base == "C":
                octave += 1
            name = f"{base}{octave}"
            white_names_by_id[id(q)] = name
            prev = base

        # 5) assign black names by which reversed white gap they lie in
        black_names_by_id = {}
        for bq in black_sorted:
            bx = _quad_center_x(bq)
            assigned = "?"

            # find neighboring whites in the REVERSED musical order
            for i in range(len(white_xs) - 1):
                L = white_xs[i]
                R = white_xs[i + 1]
                lo, hi = (L, R) if L < R else (R, L)

                if lo < bx < hi:
                    wL = WHITE_CYCLE[(start + i) % 7]
                    wR = WHITE_CYCLE[(start + i + 1) % 7]
                    sharp = BLACK_NAME_BETWEEN.get((wL, wR), None)

                    if sharp is not None:
                        left_white_quad = white_sorted[i]
                        left_white_name = white_names_by_id[id(left_white_quad)]
                        o = _extract_octave_number(left_white_name)
                        assigned = f"{sharp}{o}"
                    else:
                        assigned = "?"
                    break

            black_names_by_id[id(bq)] = assigned

        # 6) map back to original list ordering (so it aligns with self.white_keys / self.black_keys)
        self.white_key_names = [white_names_by_id.get(id(q), "?") for q in self.white_keys]
        self.black_key_names = [black_names_by_id.get(id(q), "?") for q in self.black_keys]
     
        if debug:
            # show L->R geometric order vs forced musical order
            print("white_xs_lr:", white_xs_lr)
            print("white_xs_rev:", white_xs)
            print("white names (forced musical order):", [white_names_by_id[id(q)] for q in white_sorted])
            print("black names:", [black_names_by_id[id(q)] for q in black_sorted])

        return self.white_key_names, self.black_key_names


    def create_overlay(self, img):
        empty = np.zeros((img.shape), np.uint8)
        self.draw_on(empty)
        self.overlay = empty
        mask = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
        mask[mask>0] = 255
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self.mask = mask

    def draw_overlay(self, vis):
        vis[self.mask!=0] = self.overlay[self.mask!=0]

    def draw_on(self, vis):
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # piano trapezoid
        cv2.polylines(
            vis,
            [self.piano.astype(np.int32).reshape(-1, 1, 2)],
            True,
            (255, 0, 0),
            2
        )

        # draw + label WHITE keys
        for i, key in enumerate(self.white_keys):
            poly = key.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly], True, (0, 0, 255), 2)
        for i, key in enumerate(self.white_keys):
            name = self.white_key_names[i] if i < len(self.white_key_names) else "?"
            cx, cy = quad_bottom_midpoint_xy(key)

            # move label a bit upward from the bottom edge
            put_label(vis, name, cx - 12, cy - 8, (0, 255, 0), scale=0.55, thickness=2)

        # draw + label BLACK keys
        for i, key in enumerate(self.black_keys):
            poly = key.astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(vis, [poly], (0, 0, 255))
            cv2.polylines(vis, [poly], True, (0, 0, 255), 2)

            name = self.black_key_names[i] if i < len(self.black_key_names) else "?"
            cx, cy = quad_bottom_midpoint_xy(key)

            # for black keys, place slightly higher so it doesn't sit on the bottom
            q = np.asarray(key, dtype=np.float32)
            y_top = float(np.min(q[:, 1]))
            y_bot = float(np.max(q[:, 1]))
            y_text = y_top + 0.65 * (y_bot - y_top)

            put_label(vis, name, cx - 14, y_text, (255, 255, 255), scale=0.55, thickness=2)

        for i, key in enumerate(self.white_keys):
            name = self.white_key_names[i] if i < len(self.white_key_names) else "?"
            cx, cy = quad_bottom_midpoint_xy(key)

            # move label a bit upward from the bottom edge
            put_label(vis, name, cx - 12, cy - 8, (0, 255, 0), scale=0.55, thickness=2)

        cv2.putText(vis, f"White keys: {len(self.white_keys)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Black keys: {len(self.black_keys)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        draw_bottom_points(vis, self.white_keys, color=(0, 255, 255), label_prefix="W")
        draw_bottom_points(vis, self.black_keys, color=(255, 255, 0), label_prefix="B")
        return vis


    def point_to_note(self, point):
        """
        Given point (x,y), return note name string.
        Black keys have priority over white keys.
        Returns None if point is not inside any key.
        """
        if point is None:
            return None, -1, False

        x, y = point
        if not (np.isfinite(x) and np.isfinite(y)):
            return None, -1, False

        # black keys first
        for idx, (quad, name) in enumerate(zip(self.black_keys, self.black_key_names)):
            if _point_in_poly((x, y), quad):
                return name, idx, True

        # then white keys
        for idx, (quad, name) in enumerate(zip(self.white_keys, self.white_key_names)):
            if _point_in_poly((x, y), quad):
                return name, idx, False

        return None, -1, False
    
    def draw_point_note(
        self,
        vis,
        point,
        idx,
        is_black,
        note_name=None,
        point_color=(0, 255, 255),
        white_color=(0, 255, 0),
        black_color=(0, 255, 0),
        fill_alpha=0.25,
    ):
        """
        Draws:
        - the touch point
        - highlights the key given by (idx, is_black)
        - draws the note name
        """
    
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        out = vis.copy()
        # draw point
        if point is not None and np.all(np.isfinite(point)):
            px, py = int(round(point[0])), int(round(point[1]))
            cv2.circle(out, (px, py), 6, point_color, -1, cv2.LINE_AA)
            cv2.circle(out, (px, py), 6, (0, 0, 0), 2, cv2.LINE_AA)

        # nothing to highlight
        if idx < 0 or is_black is None:
            return out

        # select quad
        if is_black:
            quad = self.black_keys[idx]
            color = black_color
            name = note_name or self.black_key_names[idx]
        else:
            quad = self.white_keys[idx]
            color = white_color
            name = note_name or self.white_key_names[idx]

        cnt = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)

        # fill overlay
        overlay = out.copy()
        cv2.fillPoly(overlay, [cnt], color)
        out = cv2.addWeighted(overlay, fill_alpha, out, 1.0 - fill_alpha, 0)

        # outline
        cv2.polylines(out, [cnt], True, color, 2, cv2.LINE_AA)

        # label near point (fallback to quad center)
        if point is not None and np.all(np.isfinite(point)):
            tx, ty = px + 10, py - 10
        else:
            q = np.asarray(quad, dtype=np.float32)
            tx, ty = int(np.mean(q[:, 0])), int(np.mean(q[:, 1]))

        # outlined text
        cv2.putText(out, name, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, name, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vis[:,:,:] = out[:,:,:]
        return out

    
    
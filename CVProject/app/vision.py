import cv2
import sys
import os
import numpy as np
from numba import njit, prange



def find_keyboard_barcode_band(gray: np.ndarray,
                               smooth_ksize: int = 31,
                               band_frac: float = 0.50,
                               min_band_h: int = 40):
    """
    Returns bounding box (x0,y0,x1,y1) for a piano-like 'barcode band' using vertical edge energy.
    smooth_ksize: smoothing size for the E(y) signal (odd)
    band_frac: how much of peak energy to keep when expanding the band (0.35-0.65 typical)
    min_band_h: minimum band height in pixels
    """
    if gray.ndim != 2:
        raise ValueError("gray must be single-channel")

    H, W = gray.shape

    # 1) vertical-edge map (barcode stripes)
    # dx highlights vertical edges (key separators).
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    abs_dx = np.abs(dx)

    # use a gentle blur so energy is more stable
    abs_dx_blur = cv2.GaussianBlur(abs_dx, (5, 5), 0)

    # 2) collapse to 1D energy over rows: E(y)
    E = abs_dx_blur.sum(axis=1)  

    # smooth E to make it stable
    k = smooth_ksize if smooth_ksize % 2 == 1 else smooth_ksize + 1
    E_s = cv2.GaussianBlur(E.reshape(-1, 1).astype(np.float32), (1, k), 0).reshape(-1)

    # 3) find peak row and expand to a band
    y_peak = int(np.argmax(E_s))
    peak = float(E_s[y_peak])
    if peak <= 1e-6:
        return None, {"E": E_s, "abs_dx": abs_dx_blur}

    thr = peak * band_frac

    y0 = y_peak
    while y0 > 0 and E_s[y0] >= thr:
        y0 -= 1
    y1 = y_peak
    while y1 < H - 1 and E_s[y1] >= thr:
        y1 += 1

    # ensure minimum band height (helps when peak is sharp)
    if (y1 - y0) < min_band_h:
        pad = (min_band_h - (y1 - y0)) // 2 + 1
        y0 = max(0, y0 - pad)
        y1 = min(H, y1 + pad)

    # 4) determine left/right by x-projection inside the band
    band = abs_dx_blur[y0:y1, :]  
    P = band.sum(axis=0)          

    # smooth P
    P_s = cv2.GaussianBlur(P.reshape(1, -1).astype(np.float32), (k, 1), 0).reshape(-1)

    # threshold to find contiguous "active" region
    p_thr = float(P_s.max()) * 0.25 
    active = P_s >= p_thr

    if not np.any(active):
        return None, {"E": E_s, "P": P_s, "abs_dx": abs_dx_blur, "band_y": (y0, y1)}

    xs = np.where(active)[0]
    x0, x1 = int(xs[0]), int(xs[-1])

    # slight expand to include borders
    margin = int(0.02 * W)
    x0 = max(0, x0 - margin)
    x1 = min(W - 1, x1 + margin)

    # slight expand to include borders
    marginY = int(0.05 * W)
    y0 = max(0, y0 - margin)
    y1 = min(H - 1, y1 + margin)
    return (x0, y0, x1, y1), {"E": E_s, "P": P_s, "abs_dx": abs_dx_blur, "band_y": (y0, y1)}


            

def sauvola_binarize(gray: np.ndarray, window: int = 31, k: float = 0.34, R: float = 128.0):
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be odd and >= 3")

    gray_f = gray.astype(np.float32)

    # local mean
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(window, window), normalize=True)

    # local mean of squares
    mean_sq = cv2.boxFilter(gray_f * gray_f, ddepth=-1, ksize=(window, window), normalize=True)

    # variance/std
    var = mean_sq - mean * mean
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    # Sauvola threshold
    thresh = mean * (1.0 + k * (std / R - 1.0))

    # binarize
    bin = (gray_f > thresh).astype(np.uint8) * 255

    return bin

def detect_piano_region(img):
  
    # Bilateral filter parameters:
    # d: neighborhood diameter (pixels). If -1, OpenCV derives it from sigmaSpace.
    # sigmaColor: how much colors can differ to still be averaged (higher => more smoothing across colors)
    # sigmaSpace: how far pixels can be to still influence each other (higher => smoother, more global)
    d = 20
    sigmaColor = 150
    sigmaSpace = 150

    filtered = cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    filtered = cv2.erode(img, kernel=np.ones((5,5)))
    filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    
    box, dbg = find_keyboard_barcode_band(
        filtered,
        smooth_ksize=31,
        band_frac=0.55,   
        min_band_h=50
    )


    vis = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    vis2 = vis.copy()
    region = filtered[box[1]:box[3],box[0]:box[2]]

    if box is None:
        cv2.putText(vis, "Keyboard NOT found", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    else:
        x0, y0, x1, y1 = box
        cv2.rectangle(vis2, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(vis2, "Keyboard band", (x0, max(0, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite("box.png", vis2)
    #cv2.imshow("Detected keyboard (barcode band)", vis)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    #cv2.imshow("Bilateral Filtered", filtered)
    cv2.imwrite("filtered.png", filtered)
    cv2.imwrite("region.png", region)
    #cv2.imshow("Side by side (Original | Filtered)", side_by_side)

    #print("Press any key in an image window to close...")
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return box, region


def _items_to_filled_mask(items, shape_hw, dilate_px=3):
    # fill each quad into a mask. Optionally dilate to enlarge slightly
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)

    for it in items:
        q = np.asarray(it["quad"], dtype=np.float32)
        q_i = np.round(q).astype(np.int32)
        cv2.fillPoly(mask, [q_i], 255)

    if dilate_px and dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, ker, iterations=1)

    return mask


def _largest_component_mask(binary_255, connectivity=8):
    #Given binary (0/255) return mask (0/255) of largest foreground CC (excluding background).
    bin01 = (binary_255 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=connectivity)

    if num <= 1:
        return None, None  # no components

    # stats: [x,y,w,h,area] for each label
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1  # +1 because we skipped background
    out = (labels == idx).astype(np.uint8) * 255
    return out, stats[idx]


def _quad_corners_inside_trapezoid(quad, trap_quad):
    """
    Fast inclusion test: all 4 corners of quad must be inside trapezoid polygon.
    Uses cv2.pointPolygonTest.
    """
    poly = np.asarray(trap_quad, dtype=np.float32).reshape(-1, 2)
    q = np.asarray(quad, dtype=np.float32).reshape(-1, 2)

    for (x, y) in q:
        # >= 0 means inside or on edge
        if cv2.pointPolygonTest(poly, (float(x), float(y)), False) < 0:
            return False
    return True


def refine_keys_by_main_white_trapezoid(
    white_keys,
    black_keys,
    roi_shape_hw,
    dilate_white_px=4,
    slope_range=(-3.0, 3.0),
    coarse_steps=151,
    refine_iters=4,
    refine_shrink=6.0,
    enforce_narrower_top=False,
    debug_prefix="",
):
    """
    1) Make a mask of white keys (filled quads) and dilate slightly
    2) Take biggest CC from that mask
    3) Fit horizontal-bases min-area trapezoid enclosing that CC
    4) Remove all key items (white+black) whose quads lie OUTSIDE the trapezoid

    Returns:
      trap_quad (4x2 float32) or None
      white_in, black_in  (filtered lists)
      dbg dict with masks and stats
    """
    H, W = roi_shape_hw

    # 1) mask of white keys (dilated slightly)
    white_mask = _items_to_filled_mask(white_keys, (H, W), dilate_px=dilate_white_px)

    # 2) biggest CC
    biggest_mask, biggest_stats = _largest_component_mask(white_mask, connectivity=8)
    if biggest_mask is None:
        return None, [], [], {"reason": "no white CCs", "white_mask": white_mask}

    # 3) fit trapezoid around biggest CC: use convex hull points from that mask
    hull_pts, hull_area = _hull_points_from_mask(biggest_mask)  
    if hull_pts is None or len(hull_pts) < 3:
        return None, [], [], {"reason": "failed hull on biggest CC", "white_mask": white_mask, "biggest_mask": biggest_mask}

    trap_quad, info = min_area_horizontal_trapezoid_from_points(
        hull_pts,
        slope_range=slope_range,
        coarse_steps=coarse_steps,
        refine_iters=refine_iters,
        refine_shrink=refine_shrink,
        enforce_narrower_top=enforce_narrower_top,
    )
    if trap_quad is None:
        return None, [], [], {
            "reason": "trapezoid fit failed",
            "white_mask": white_mask,
            "biggest_mask": biggest_mask,
            "biggest_stats": biggest_stats,
        }

    # 4) filter items (keep only those fully inside trapezoid)
    white_in = [it for it in white_keys if _quad_corners_inside_trapezoid(it["quad"], trap_quad)]
    black_in = [it for it in black_keys if _quad_corners_inside_trapezoid(it["quad"], trap_quad)]

    dbg = {
        "reason": "ok",
        "white_mask": white_mask,
        "biggest_mask": biggest_mask,
        "biggest_stats": biggest_stats,
        "trap_quad": trap_quad,
        "trap_info": info,
    }

    # optional debug images
    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}white_mask.png", white_mask)
        cv2.imwrite(f"{debug_prefix}biggest_white_cc.png", biggest_mask)

    return trap_quad, white_in, black_in, dbg

def binarize_global(gray: np.ndarray, thresh: int = 140) -> np.ndarray:
    """Non-adaptive threshold. Returns uint8 binary (0 or 255)."""
    if gray.ndim != 2:
        raise ValueError("binarize_global expects a single-channel grayscale image")
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return bw


def order_quad(quad: np.ndarray) -> np.ndarray:
    """Order 4 points as [tl, tr, br, bl]."""
    quad = np.asarray(quad, dtype=np.float32)
    s = quad.sum(axis=1)
    diff = np.diff(quad, axis=1).reshape(-1)
    tl = quad[np.argmin(s)]
    br = quad[np.argmax(s)]
    tr = quad[np.argmin(diff)]
    bl = quad[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _hull_points_from_mask(mask: np.ndarray):
    """Return convex hull points (N,2) float32 from a component mask (255 foreground)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area <= 1.0:
        return None, area
    hull = cv2.convexHull(c)  # (M,1,2)
    pts = hull.reshape(-1, 2).astype(np.float32)
    return pts, area


def min_area_horizontal_trapezoid_from_points(
    pts_xy: np.ndarray,
    slope_range=(-3.0, 3.0),
    coarse_steps=151,
    refine_iters=4,
    refine_shrink=6.0,
    enforce_narrower_top=False,
    min_height_px=3.0,
    min_width_px=3.0,
    debug=False,
):
    """
    Minimum-area enclosing trapezoid with horizontal top/bottom, containing ALL points.

    Model:
      top/bottom: y = const
      left edge:  x >= aL*y + bL   where bL = min(x - aL*y)
      right edge: x <= aR*y + bR   where bR = max(x - aR*y)

    Returns:
      quad (4,2) float32 in [tl,tr,br,bl] order (already consistent, no heuristic reorder),
      info dict
    """
    pts = np.asarray(pts_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return None, {"reason": "need (N,2) with N>=3"}

    xs = pts[:, 0]
    ys = pts[:, 1]

    y_top = float(np.min(ys))
    y_bot = float(np.max(ys))
    H = y_bot - y_top

    # HARD GUARD: avoid "single line" trapezoids
    if H < float(min_height_px):
        return None, {"reason": "degenerate height", "H": float(H), "y_top": y_top, "y_bot": y_bot}

    lo, hi = slope_range
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        return None, {"reason": "invalid slope_range", "slope_range": slope_range}

    best = None  # (area, aL, aR, bL, bR, w_top, w_bot)

    span = (hi - lo)

    def quad_area(q):
        # polygon area for [tl,tr,br,bl]
        x = q[:, 0]
        y = q[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    for it in range(refine_iters):
        if it == 0:
            aL_vals = np.linspace(lo, hi, coarse_steps, dtype=np.float32)
            aR_vals = np.linspace(lo, hi, coarse_steps, dtype=np.float32)
        else:
            # refine around best
            _, aL_center, aR_center, _, _, _, _ = best
            half = span / (2.0 * (refine_shrink ** it))
            aL_vals = np.linspace(aL_center - half, aL_center + half, coarse_steps, dtype=np.float32)
            aR_vals = np.linspace(aR_center - half, aR_center + half, coarse_steps, dtype=np.float32)

        for aL in aL_vals:
            aL = float(aL)

            # tight left intercept (guarantees containment)
            bL = float(np.min(xs - aL * ys))
            xL_top = aL * y_top + bL
            xL_bot = aL * y_bot + bL

            for aR in aR_vals:
                aR = float(aR)

                # tight right intercept (guarantees containment)
                bR = float(np.max(xs - aR * ys))
                xR_top = aR * y_top + bR
                xR_bot = aR * y_bot + bR

                w_top = xR_top - xL_top
                w_bot = xR_bot - xL_bot

                # HARD GUARD: avoid near-line quads
                if w_top < float(min_width_px) or w_bot < float(min_width_px):
                    continue

                if enforce_narrower_top and not (w_top < w_bot):
                    continue

                area = 0.5 * (w_top + w_bot) * H
                if not np.isfinite(area):
                    continue

                if (best is None) or (area < best[0]):
                    best = (area, aL, aR, bL, bR, w_top, w_bot)

        if best is None:
            if enforce_narrower_top:
                enforce_narrower_top = False
                continue
            return None, {"reason": "no feasible trapezoid in slope_range"}

    area, aL, aR, bL, bR, w_top, w_bot = best

    # Build quad in a guaranteed consistent order (no order_quad heuristic)
    tl = np.array([aL * y_top + bL, y_top], dtype=np.float32)
    tr = np.array([aR * y_top + bR, y_top], dtype=np.float32)
    br = np.array([aR * y_bot + bR, y_bot], dtype=np.float32)
    bl = np.array([aL * y_bot + bL, y_bot], dtype=np.float32)
    quad = np.stack([tl, tr, br, bl], axis=0).astype(np.float32)

    # final sanity checks
    A = quad_area(quad)
    if abs(A) < float(min_width_px) * float(min_height_px) * 0.25:
        # area still too small => effectively a line/needle
        return None, {"reason": "degenerate quad area", "area": float(A), "H": float(H),
                      "w_top": float(w_top), "w_bot": float(w_bot)}

    # ensure tl is left of tr, bl is left of br (avoid flipped)
    if not (quad[0, 0] <= quad[1, 0] and quad[3, 0] <= quad[2, 0]):
        # this can happen numerically for extreme slopes; reject
        return None, {"reason": "invalid ordering (x)", "quad": quad.tolist()}

    info = {
        "reason": "ok",
        "area": float(area),
        "aL": float(aL), "aR": float(aR),
        "bL": float(bL), "bR": float(bR),
        "y_top": float(y_top), "y_bot": float(y_bot),
        "H": float(H),
        "w_top": float(w_top), "w_bot": float(w_bot),
    }
    if debug:
        print("[min_area_horizontal_trapezoid] OK", info)
    return quad, info


def quad_from_component_mask(mask: np.ndarray):
    """
    QUAD THAT CONTAINS THE WHOLE CONNECTED COMPONENT
    Returns: (quad_4x2 float32, contour_area float, (cx, cy))
    """
    hull_pts, area = _hull_points_from_mask(mask)
    if hull_pts is None:
        return None, 0.0, (0.0, 0.0)

    quad, info = min_area_horizontal_trapezoid_from_points(
        hull_pts,
        slope_range=(-3.0, 3.0),
        coarse_steps=151,
        refine_iters=4,
        refine_shrink=6.0,
        enforce_narrower_top=False,  
    )
    if quad is None:
        print(info)
        exit()
        return None, float(area), (0.0, 0.0)

    cx = float(np.mean(quad[:, 0]))
    cy = float(np.mean(quad[:, 1]))
    return quad.astype(np.float32), float(area), (cx, cy)


def filter_area_outliers(items, method="med", iqr_k=2.5, med_factor=4):
    """items: list of dicts with 'area' field -> filtered list."""
    if not items:
        return items
    areas = np.array([it["area"] for it in items], dtype=np.float32)
    if len(areas) < 4:
        return items

    if method == "iqr":
        q1 = np.quantile(areas, 0.25)
        q3 = np.quantile(areas, 0.75)
        iqr = q3 - q1
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
        return [it for it in items if lo <= it["area"] <= hi]

    med = float(np.median(areas))
    lo = med / med_factor
    hi = med * med_factor
    return [it for it in items if lo <= it["area"] <= hi]

def connected_components_quads(binary: np.ndarray,
                               min_area: int = 50,
                               max_area: int | None = None,
                               area_outlier_method: str = "med",
                               reject_touch_lr: bool = True,
                               reject_horizontal: bool = True,
                               border_margin: int = 0):
    """
    extract quads from connected components (foreground=255), optionally rejecting
    components that touch the left/right image borders.

    reject_touch_lr: if True, drop components whose bbox touches left or right border
    border_margin: treat components within this many pixels of border as "touching"
    """
    if binary.ndim != 2:
        raise ValueError("connected_components_quads expects single-channel binary")

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    items = []
    H, W = binary.shape
    if max_area is None:
        max_area = int(0.9 * H * W)

    for i in range(1, num):  # skip background
        x, y, w, h, a = stats[i]
        if a < min_area or a > max_area:
            continue

        if reject_touch_lr:
            touches_left = x <= border_margin
            touches_right = (x + w) >= (W - border_margin)
            if touches_left or touches_right:
                continue
        
        if reject_horizontal: 
            
            if w > h * 3:
                continue

        comp_mask = (labels == i).astype(np.uint8) * 255
        quad, c_area, (cx, cy) = quad_from_component_mask(comp_mask)
        if quad is None:
            continue

        items.append({"quad": quad, "area": float(c_area), "cx": float(cx), "cy": float(cy)})

    items = filter_area_outliers(items, method=area_outlier_method)
    items.sort(key=lambda it: it["cx"])
    return items


def detect_black_keys_from_roi(gray_roi: np.ndarray,
                               thresh: int = 140,
                               erode_ksize: int = 3,
                               dilate_ksize: int = 7,
                               min_area: int = 80,
                               area_outlier_method: str = "med"):
    """
    black keys:
    binarize -> invert (black becomes 255) -> erode/dilate -> CC -> enclosing trapezoid per CC -> filter -> sort.
    """
    bw = binarize_global(gray_roi, thresh=thresh)
    inv = cv2.bitwise_not(bw)

    kE = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_ksize, erode_ksize))
    kD = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ksize, dilate_ksize))

    proc = cv2.erode(inv, kE, iterations=1)
    proc = cv2.dilate(proc, kD, iterations=1)
    cv2.imwrite("black keys.png", cv2.bitwise_not(proc))
    items = connected_components_quads(
        proc,
        min_area=min_area,
        area_outlier_method=area_outlier_method,
    )
    

    return bw, proc, items


def detect_white_keys_from_roi(gray_roi: np.ndarray,
                               thresh: int = 140,
                               open_ksize: int = 5,
                               min_area: int = 150,
                               area_outlier_method: str = "med"):
    """
    white keys:
    binarize -> open (optional) -> CC -> enclosing trapezoid per CC -> filter -> sort.
    """
    bw = binarize_global(gray_roi, thresh=thresh)
    bw = cv2.bitwise_not(bw)

    bw = cv2.ximgproc.thinning(
    bw,
    thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
)
    bw = cv2.bitwise_not(bw)
    kE = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bw = cv2.erode(bw, kE, iterations=1)
    cv2.imwrite("thinned.png", bw)
    kO = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
    proc = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kO, iterations=1)

    cv2.imwrite("proc.png",proc)
    items = connected_components_quads(
        proc,
        min_area=min_area,
        area_outlier_method=area_outlier_method,
    )

    return bw, proc, items


def draw_quads(vis_bgr: np.ndarray, items, color=(0, 255, 0), thickness=2, fill=False, alpha=0.25, tuple_mode=False):
    """
    draw quads on a BGR image.
    If fill=True, fills with alpha blending.
    """
    out = vis_bgr.copy()
    if not fill:
        for it in items:
            if tuple_mode:
                q = it.astype(np.int32)
            else:
                q = it["quad"].astype(np.int32)
            cv2.polylines(out, [q], True, color, thickness)
        return out

    overlay = out.copy()
    for it in items:
        if tuple_mode:
            q = it.astype(np.int32)
        else:
            q = it["quad"].astype(np.int32)
        cv2.fillPoly(overlay, [q], color)
        cv2.polylines(out, [q], True, color, thickness)
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
    return out




def detect_piano_keys(gray, prefix="", show=False):
    """
    loads region.png (ROI grayscale) and visualizes detected black/white keys with different colors.
    each CC gets a MIN-AREA enclosing horizontal-trapezoid quad (no clipping).
    """

    # small blur helps CC stability
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    THRESH = 170
    _, bw = cv2.threshold(gray, THRESH, 255, cv2.THRESH_BINARY)
    cv2.imwrite("treshtest.png", bw)
    bw, black_proc, black_keys = detect_black_keys_from_roi(
        gray_blur,
        thresh=THRESH,
        erode_ksize=10,
        dilate_ksize=10,
        min_area=500,
        area_outlier_method="med",
    )

    bw2, white_proc, white_keys = detect_white_keys_from_roi(
        gray_blur,
        thresh=THRESH,
        open_ksize=5,
        min_area=500,
        area_outlier_method="med",
    )

    trap_quad, white_keys, black_keys2, dbg = refine_keys_by_main_white_trapezoid(
    white_keys, black_keys,
    roi_shape_hw=gray.shape,
    dilate_white_px=4,
    debug_prefix=prefix
    )

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if trap_quad is not None:
        cv2.polylines(vis, [trap_quad.astype(np.int32)], True, (255, 0, 0), 2)  # blue trapezoid

    # draw white first, then black on top
    vis = draw_quads(vis, white_keys, color=(0, 255, 0), thickness=2, fill=True, alpha=0.15)  # green
    vis = draw_quads(vis, black_keys, color=(0, 0, 255), thickness=2, fill=True, alpha=0.20)  # red

    cv2.putText(vis, f"White keys: {len(white_keys)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, f"Black keys: {len(black_keys)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # save debug outputs
    cv2.imwrite(f"{prefix}dbg_bw.png", bw)
    cv2.imwrite(f"{prefix}dbg_black_proc.png", black_proc)
    cv2.imwrite(f"{prefix}dbg_white_proc.png", white_proc)
    cv2.imwrite(f"{prefix}keys_overlay.png", vis)
    cv2.imwrite(f"{prefix}black_keys.png", cv2.bitwise_not(black_proc))
    if show:
        # show windows
        cv2.imshow("ROI gray", gray)
        cv2.imshow("Binary (global)", bw)
        cv2.imshow("Black keys mask (proc)", black_proc)
        cv2.imshow("White keys mask (proc)", white_proc)
        cv2.imshow("Keys overlay (green=white, red=black)", vis)

    print("Saved: dbg_bw.png, dbg_black_proc.png, dbg_white_proc.png, black_keys.png, keys_overlay.png")
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return [x["quad"] for x in white_keys], [x["quad"] for x in black_keys], trap_quad


if __name__ == "__main__":
    detect_piano_keys()

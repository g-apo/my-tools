import csv
from operator import itemgetter
import cv2 as cv
import numpy as np


first = itemgetter(0)
last = itemgetter(-1)


def group_adjacent_indices(indices, max_gap=2):
    """Groups adjacent pixel indices belonging to the same line and returns each line's center."""
    if len(indices) == 0:
        return []
    splits = first(np.nonzero(np.diff(indices) > max_gap)) + 1
    groups = np.split(indices, splits)
    return [int(round(np.mean(g))) for g in groups if len(g) > 0]


def detect_l_axes(img):
    """Step 1: Locks onto the L-axis corner, finds the exact '0' and '5' Y-ticks, and measures scale."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    # 1. Left Y-axis (x = 0) between cols 15 and 80
    v_profile = np.mean(gray[slice(120, 480), slice(15, 80)], axis=0)
    y_axis_col = int(np.argmin(v_profile)) + 15

    # 2. Physical bottom X-axis corner between rows 525 and 552
    h_profile = np.mean(gray[slice(525, 552), slice(100, w_img - 50)], axis=1)
    x_axis_corner_row = int(np.argmin(h_profile)) + 525

    # 3. Measure major tick marks sticking out below the X-axis (every 50 min)
    is_tick_gray = gray < 230
    x_tick_strip = is_tick_gray[slice(x_axis_corner_row + 2, min(h_img, x_axis_corner_row + 6)), slice(y_axis_col, w_img - 10)]
    x_ticks = group_adjacent_indices(first(np.nonzero(np.mean(x_tick_strip, axis=0) > 0.3)), max_gap=3)

    # 4. Find all tick marks sticking out left of the Y-axis (1 to 6 px left of y_axis_col)
    y_tick_strip = is_tick_gray[slice(65, x_axis_corner_row - 2), slice(max(0, y_axis_col - 6), max(1, y_axis_col - 1))]
    all_y_ticks = group_adjacent_indices(first(np.nonzero(np.mean(y_tick_strip, axis=1) > 0.3)), max_gap=2)

    # The '0' tick mark sits 4 to 15 pixels above the physical corner (safe fallback: 8 px above corner)
    zero_candidates = [t for t in all_y_ticks if (x_axis_corner_row - 15) <= t <= (x_axis_corner_row - 4)]
    y_zero_row = last(zero_candidates) if len(zero_candidates) > 0 else (x_axis_corner_row - 8)

    # The '5' tick mark sits near the top of the Y-axis (rows 75 to 115; safe fallback: row 88)
    five_candidates = [t for t in all_y_ticks if 75 <= t <= 115]
    y_five_row = first(five_candidates) if len(five_candidates) > 0 else (first(all_y_ticks) if len(all_y_ticks) > 0 else 88)

    px_per_x_tick = float(np.median(np.diff(x_ticks))) if len(x_ticks) >= 2 else (w_img - y_axis_col) / 9.0
    px_per_y_unit = (y_zero_row - y_five_row) / 5.0

    bounds = {
        "left": y_axis_col,
        "right": w_img - 10,
        "top": 65,
        "bottom": x_axis_corner_row,   # Encloses the whole Y-axis down to the zoomed-in corner
        "y_zero": y_zero_row,          # Locked directly onto the '0' tick mark
        "px_per_x_unit": px_per_x_tick / 50.0,
        "px_per_y_unit": px_per_y_unit,
    }
    print(f"[CHECK 1 PASSED] Corner at ({bounds.get('left')}, {bounds.get('bottom')}) | Y=0 tick at row {bounds.get('y_zero')} | Scale: {bounds.get('px_per_x_unit'):.3f} px/min, {bounds.get('px_per_y_unit'):.3f} px/Norm")
    return bounds



def find_colored_pixels(img, bounds):
    """Step 2: Collects every (column, row) pixel for Red, Green, Magenta, and Blue curves."""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    color_ranges = {
        "red":     (np.array((0, 35, 50), dtype=np.uint8),     np.array((14, 255, 230), dtype=np.uint8)),
        "green":   (np.array((38, 30, 40), dtype=np.uint8),    np.array((88, 255, 220), dtype=np.uint8)),
        "magenta": (np.array((140, 80, 100), dtype=np.uint8),  np.array((170, 255, 255), dtype=np.uint8)),
        "blue":    (np.array((105, 120, 80), dtype=np.uint8),  np.array((130, 255, 255), dtype=np.uint8)),
    }

    # Enclose the entire plot area down to the physical bottom corner
    plot_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    plot_mask[slice(bounds.get("top"), bounds.get("bottom")), slice(bounds.get("left") + 1, bounds.get("right"))] = 255

    # Exclude the top-left legend box (DAD1 D, Sig=254,16...)
    plot_mask[slice(0, 95), slice(0, 260)] = 0

    color_pixels = {}
    for name, (lower, upper) in color_ranges.items():
        mask = cv.inRange(hsv, lower, upper)
        if name == "red":
            mask_wrap = cv.inRange(hsv, np.array((170, 35, 50), dtype=np.uint8), np.array((179, 255, 230), dtype=np.uint8))
            mask = cv.bitwise_or(mask, mask_wrap)

        mask = cv.bitwise_and(mask, plot_mask)

        # Keep connected segments >= 4 pixels so steep curve segments aren't dropped
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask)
        clean_mask = np.zeros_like(mask)
        for label_id in range(1, num_labels):
            if stats.item((label_id, cv.CC_STAT_AREA)) >= 4:
                clean_mask[labels == label_id] = 255

        rows, cols = np.nonzero(clean_mask)
        color_pixels[name] = list(zip(cols.tolist(), rows.tolist()))

        assert len(color_pixels.get(name)) > 0, f"Check Failed: 0 pixels found for {name}!"
        print(f"[CHECK 2 PASSED] {name.upper()}: {len(color_pixels.get(name))} pixels found")

    return color_pixels


def pixels_to_data(bounds, color_pixels):
    """Step 3: Selects the middle pixel of each vertical range and converts to (min, Norm.) values."""
    x_px_0 = bounds.get("left")
    y_px_0 = bounds.get("y_zero")   # Measures Y relative to the '0' tick mark
    px_per_x = bounds.get("px_per_x_unit")
    px_per_y = bounds.get("px_per_y_unit")

    extracted_data = {}
    middle_pixels = {}

    for name, coords in color_pixels.items():
        rows_by_col = {}
        for x_px, y_px in coords:
            rows_by_col.setdefault(x_px, []).append(y_px)

        data_points = []
        single_pixel_path = []

        for x_px in sorted(rows_by_col.keys()):
            # When there are multiple rows in the same column, select the middle pixel of that vertical range
            mid_y_px = float(np.median(rows_by_col.get(x_px)))
            single_pixel_path.append((x_px, int(round(mid_y_px))))

            x_val = (x_px - x_px_0) / px_per_x
            y_val = (y_px_0 - mid_y_px) / px_per_y

            data_points.append((round(float(x_val), 2), round(float(y_val), 4)))

        extracted_data[name] = data_points
        middle_pixels[name] = single_pixel_path
        print(f"[CHECK 3 PASSED] {name.upper()}: {len(data_points)} points | Sample: {data_points[:3]}")

    return extracted_data, middle_pixels


def export_to_csv(extracted_data, output_path="extracted_graph2_data.csv"):
    """Step 4: Aligns all four curves by Time and exports a Pandas-ready CSV feature matrix."""
    colors = ["red", "green", "magenta", "blue"]
    series_lookups = {c: dict(pts) for c, pts in extracted_data.items()}
    all_times = sorted({t for pts in extracted_data.values() for t, _ in pts})

    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time"] + colors)
        for t in all_times:
            writer.writerow([t] + [series_lookups.get(c).get(t, "") for c in colors])

    assert len(all_times) > 0, "Check Failed: No rows written to CSV!"
    print(f"[CHECK 4 PASSED] Saved {len(all_times)} aligned Time rows to '{output_path}'")


def run_visual_checks(img, bounds, color_pixels, middle_pixels):
    """Cumulative Visual Check: L-Axes corner box (1), Y=0 tick line (1b), Curves (2), Middle path (3)."""
    gray_bg = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    canvas = cv.addWeighted(gray_bg, 0.35, np.zeros_like(gray_bg), 0.65, 0)

    # Check 1: Magenta bounding box hugging the physical L-axis corner
    cv.rectangle(canvas, (bounds.get("left"), bounds.get("top")), (bounds.get("right"), bounds.get("bottom")), (255, 0, 255), 2)

    # Check 1b: Thin cyan line aligned directly with the '0' tick mark on the Y-axis
    cv.line(canvas, (bounds.get("left"), bounds.get("y_zero")), (bounds.get("right"), bounds.get("y_zero")), (255, 255, 0), 1)

    # Check 2: Neon highlights for every detected curve pixel
    highlight_bgr = {
        "red":     (0, 0, 255),
        "green":   (0, 255, 0),
        "magenta": (255, 0, 255),
        "blue":    (255, 100, 0),
    }
    for name, coords in color_pixels.items():
        bgr = highlight_bgr.get(name)
        for col, row in coords:
            canvas.__setitem__((row, col), bgr)

    # Check 3: Thin white line showing the middle pixel selected per column
    for name, path in middle_pixels.items():
        for col, row in path:
            canvas.__setitem__((row, col), (255, 255, 255))

    title = "Graph 2 Checks: Corner Box (1), Y=0 Line (1b), Curves (2), Path (3)"
    cv.imshow(title, canvas)
    cv.setWindowProperty(title, cv.WND_PROP_TOPMOST, 1)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)


if __name__ == "__main__":
    original_img = cv.imread("graph2.jpg")
    assert original_img is not None, "Check Failed: Could not open graph2.jpg!"

    plot_bounds = detect_l_axes(original_img)
    detected_pixels = find_colored_pixels(original_img, plot_bounds)
    data_by_color, single_pixel_paths = pixels_to_data(plot_bounds, detected_pixels)
    export_to_csv(data_by_color, "extracted_graph2_data.csv")

    run_visual_checks(original_img, plot_bounds, detected_pixels, single_pixel_paths)

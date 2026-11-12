#Grace Apodaca
#img2datatable

import cv2 as cv
import numpy as np


def detect_gridlines(img):
    """Step 1: Finds the inner plot box and gridlines automatically."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Light-gray Excel gridlines have a brightness between 190 and 235
    is_grid = (gray > 190) & (gray < 235)

    # Find columns and rows where more than 40% of the line is grid-colored
    v_cols = np.where(np.mean(is_grid, axis=0) > 0.4)[0]
    h_rows = np.where(np.mean(is_grid, axis=1) > 0.4)[0]

    # Gridlines can be 1 to 2 pixels thick, so group adjacent pixels and find the middle
    def group_lines(indices):
        if len(indices) == 0:
            return []
        splits = np.where(np.diff(indices) > 2)[0] + 1
        groups = np.split(indices, splits)
        return [int(round(np.mean(g))) for g in groups if len(g) > 0]

    v_lines = group_lines(v_cols)
    h_lines = group_lines(h_rows)

    # Ignore any border lines sitting within 10 pixels of the outer image edges
    h_img, w_img = gray.shape
    v_lines = [x for x in v_lines if 10 < x < (w_img - 10)]
    h_lines = [y for y in h_lines if 10 < y < (h_img - 10)]

    assert len(v_lines) >= 2 and len(h_lines) >= 2, "Check Failed: Not enough gridlines found!"

    bounds = {
        "left": v_lines[0],
        "right": v_lines[-1],
        "top": h_lines[0],
        "bottom": h_lines[-1],
        "y_zero": h_lines[-2],  # The second-to-last horizontal gridline is y = 0
        "v_lines": v_lines,
        "h_lines": h_lines,
    }
    print(f"[CHECK 1 PASSED] Plot Box -> Left: {bounds['left']}, Right: {bounds['right']}, Top: {bounds['top']}, Bottom: {bounds['bottom']}, Y=0: {bounds['y_zero']}")
    return bounds


def find_colored_pixels(img, bounds):
    """Step 2: Collects every (column, row) pixel belonging to Blue, Orange, Yellow, and Gray."""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Color ranges formatted as [Hue (0-179), Saturation (0-255), Value (0-255)]
    # Tightened Orange (Hue 5-16, Sat >= 130) so anti-aliased Yellow edges don't bleed in
    color_ranges = {
        "blue":   (np.array([95, 80, 80]),   np.array([130, 255, 255])),
        "orange": (np.array([5, 130, 80]),   np.array([16, 255, 255])),
        "yellow": (np.array([20, 80, 80]),   np.array([35, 255, 255])),
        "gray":   (np.array([0, 0, 140]),    np.array([179, 40, 175])),
    }

    # Build a mask of the inner plot box so Gray ignores outside labels and legends
    plot_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    plot_mask[bounds["top"]:bounds["bottom"], bounds["left"]:bounds["right"]] = 255

    # Block out the area below y=0 after the initial spike so X-axis numbers (200-1000) aren't selected
    y_zero_row = bounds["y_zero"]
    plot_mask[y_zero_row + 2 : bounds["bottom"], bounds["left"] + 25 : bounds["right"]] = 0
    plot_mask[y_zero_row + 2 : bounds["bottom"], bounds["left"] : bounds["left"] + 4] = 0

    color_pixels = {}
    for name, (lower, upper) in color_ranges.items():
        mask = cv.inRange(hsv, lower, upper)
        if name == "gray":
            mask = cv.bitwise_and(mask, plot_mask)

        # Remove tiny isolated color-bleed specks (keep only connected strokes >= 15 pixels)
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask)
        clean_mask = np.zeros_like(mask)
        for label_id in range(1, num_labels):  # Skip label 0 (background)
            if stats[label_id, cv.CC_STAT_AREA] >= 15:
                clean_mask[labels == label_id] = 255

        # Grab matching pixels and store them as (column, row) tuples
        rows, cols = np.where(clean_mask > 0)
        color_pixels[name] = list(zip(cols.tolist(), rows.tolist()))

        assert len(color_pixels[name]) > 0, f"Check Failed: 0 pixels found for {name}!"
        print(f"[CHECK 2 PASSED] {name.upper()}: {len(color_pixels[name])} pixels found")

    return color_pixels

def pixels_to_data(bounds, color_pixels, x_max=1000.0, y_max=1.2):
    """Step 3: Ensures one Y value per X column, then converts pixel positions to graph numbers."""
    x_px_0 = bounds["left"]
    x_px_max = bounds["right"]
    y_px_0 = bounds["y_zero"]
    y_px_max = bounds["top"]

    extracted_data = {}
    middle_pixels = {}

    for name, coords in color_pixels.items():
        # Group all row numbers (y) by their column number (x)
        rows_by_col = {}
        for x_px, y_px in coords:
            if x_px_0 <= x_px <= x_px_max and y_px_max <= y_px <= bounds["bottom"]:
                rows_by_col.setdefault(x_px, []).append(y_px)

        data_points = []
        single_pixel_path = []

        for x_px in sorted(rows_by_col.keys()):
            # Because the drawn line is several pixels thick, there are multiple rows in the same column.
            # Select the middle pixel of that vertical range so each X column gets exactly one Y value.
            mid_y_px = float(np.median(rows_by_col[x_px]))
            single_pixel_path.append((x_px, int(round(mid_y_px))))

            # Find how far across the X-axis this column sits (as a percentage) and multiply by x_max
            x_val = (x_px - x_px_0) / (x_px_max - x_px_0) * x_max

            # Find how far above the y=0 line this pixel sits (as a percentage) and multiply by y_max
            y_val = (y_px_0 - mid_y_px) / (y_px_0 - y_px_max) * y_max

            data_points.append((round(float(x_val), 2), round(float(y_val), 4)))

        extracted_data[name] = data_points
        middle_pixels[name] = single_pixel_path
        print(f"[CHECK 3 PASSED] {name.upper()}: {len(data_points)} unique (Time, Value) points | Sample: {data_points[:3]}")

    return extracted_data, middle_pixels


def run_visual_checks(img, bounds, color_pixels, middle_pixels):
    """Draws all checks onto one image: Box (Step 1), Highlights (Step 2), Single-pixel path (Step 3)."""
    gray_bg = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    canvas = cv.addWeighted(gray_bg, 0.35, np.zeros_like(gray_bg), 0.65, 0)

    # Step 1 Check: Magenta plot box & cyan y=0 axis line
    cv.rectangle(canvas, (bounds["left"], bounds["top"]), (bounds["right"], bounds["bottom"]), (255, 0, 255), 2)
    cv.line(canvas, (bounds["left"], bounds["y_zero"]), (bounds["right"], bounds["y_zero"]), (255, 255, 0), 1)

    # Step 2 Check: Neon highlights for all detected pixels of each color
    highlight_colors = {
        "blue":   (255, 0, 0),
        "orange": (0, 128, 255),
        "yellow": (0, 255, 255),
        "gray":   (0, 255, 0),
    }
    for name, coords in color_pixels.items():
        bgr = highlight_colors[name]
        for col, row in coords:
            canvas[row, col] = bgr

    # Step 3 Check: Thin white line showing the single middle pixel chosen per column
    for name, path in middle_pixels.items():
        for col, row in path:
            canvas[row, col] = (255, 255, 255)

    window_title = "Cumulative Checks: Box (1), Highlights (2), Middle Pixel Path (3)"
    cv.imshow(window_title, canvas)
    cv.setWindowProperty(window_title, cv.WND_PROP_TOPMOST, 1)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)

import csv

def export_to_csv(extracted_data, output_path="extracted_graph_data.csv"):
    """Step 4: Aligns all curves by Time and exports a Pandas-ready CSV feature matrix."""
    # Convert each color's list of (Time, Value) tuples into a fast lookup dictionary: {time: value}
    series_lookups = {
        color: dict(points) for color, points in extracted_data.items()
    }

    # Collect every unique Time value across all four colors in sorted order
    all_times = sorted({time for points in extracted_data.values() for time, _ in points})
    colors = ["blue", "orange", "yellow", "gray"]

    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Header row formatted as clean column names for Pandas
        writer.writerow(["Time"] + colors)

        # Write one row per Time step; leave cell blank (which Pandas reads as NaN) if a curve was hidden
        for t in all_times:
            row = [t] + [series_lookups[c].get(t, "") for c in colors]
            writer.writerow(row)

    # Self-check: verify the file was created and has rows
    assert len(all_times) > 0, "Check Failed: No data rows written to CSV!"
    print(f"[CHECK 4 PASSED] Saved {len(all_times)} aligned Time rows to '{output_path}'")

if __name__ == "__main__":
    original_img = cv.imread("graph1.jpg")
    assert original_img is not None, "Check Failed: Could not open image!"

    plot_bounds = detect_gridlines(original_img)
    detected_pixels = find_colored_pixels(original_img, plot_bounds)
    data_by_color, single_pixel_paths = pixels_to_data(plot_bounds, detected_pixels, x_max=1000.0, y_max=1.2)
    
    # Step 4: Export the Pandas-ready feature matrix
    export_to_csv(data_by_color, "extracted_graph_data.csv")

    run_visual_checks(original_img, plot_bounds, detected_pixels, single_pixel_paths)

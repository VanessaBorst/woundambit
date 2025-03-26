
import os

import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

"""
V3: Sweeps a perpendicular line along the longest diagonal and measure distances between its intersection points with
the contour. The maximum of these distances will be the shortest diagonal. 
1) Finds the longest diagonal as a reference.
2) Computes the perpendicular vector.
3) Moves a perpendicular line in small steps along the longest diagonal.
4) Finds where this line intersects the contour at each step.
5) Measures the intersection distance and tracks the max.
6) Returns the two points with the largest perpendicular distance. 
"""

# Known real-world distances between specific marker corners (in mm)
REFERENCE_DISTANCES_MM = {
    (1, 3): 67,  # Top-left ID1 to top-right ID3
    (1, 2): 28,  # Top-left ID1 to bottom-left ID2
    (2, 4): 67,  # Bottom-left ID2 to bottom-right ID4
    (3, 4): 28,  # Top-right ID3 to bottom-right ID4
}

MARKER_SIZE_MM = 12  # Single marker size (fallback solution)

IMAGE_SIZE = (512, 512)  # Resize all images to 512x512


def resize_image(image, target_size=(512, 512)):
    """Resize an image to a target size while keeping aspect ratio."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def calculate_pixel_to_mm_ratio(image, image_name=""):
    """
    Calculate pixel-to-mm ratio using ArUco markers.
    Falls back to single marker size if only one is detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, detector_params)

    # corners: A list containing the (x, y)-coordinates of our detected ArUco markers;
    # ids: The ArUco IDs of the detected markers
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        raise ValueError(f"No ArUco markers detected in {image_name}.")

    # Convert detected markers to dictionary for easier access
    marker_corners = {id_: corner[0] for id_, corner in zip(ids.flatten(), corners)}

    used_distances = []
    pixel_distances = []

    # Compute pixel distances for known marker pairs
    for (marker1, marker2), real_dist in REFERENCE_DISTANCES_MM.items():
        if marker1 in marker_corners and marker2 in marker_corners:
            # Extract the (x, y)-coordinates of the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            (topLeft_m1, topRight_m1, bottomRight_m1, bottomLeft_m1) = marker_corners[marker1]
            (topLeft_m2, topRight_m2, bottomRight_m2, bottomLeft_m2) = marker_corners[marker2]

            # Select the correct corner points depending on the marker ID pair
            if marker1 == 1 and marker2 == 3:
                corner1, corner2 = topLeft_m1, topRight_m2
            elif marker1 == 1 and marker2 == 2:
                corner1, corner2 = topLeft_m1, bottomLeft_m2
            elif marker1 == 2 and marker2 == 4:
                corner1, corner2 = bottomLeft_m1, bottomRight_m2
            elif marker1 == 3 and marker2 == 4:
                corner1, corner2 = topRight_m1, bottomRight_m2
            else:
                raise ValueError(f"Invalid marker pair: {marker1}, {marker2}")

            pixel_dist = np.linalg.norm(corner2 - corner1)
            used_distances.append(real_dist)
            pixel_distances.append(pixel_dist)

    # If multiple markers are found, use the best estimation
    if used_distances:
        return np.mean([real / pixel for real, pixel in zip(used_distances, pixel_distances)])

    # Fallback: Single marker detected
    if len(marker_corners) == 1:
        marker_id = list(marker_corners.keys())[0]
        marker_corners_px = marker_corners[marker_id]
        width_px = np.linalg.norm(marker_corners_px[0] - marker_corners_px[1])
        height_px = np.linalg.norm(marker_corners_px[0] - marker_corners_px[3])
        avg_px_size = (width_px + height_px) / 2  # Average size for robustness
        return MARKER_SIZE_MM / avg_px_size

    raise ValueError(f"Could not determine pixel-to-mm ratio for {image_name}.")


def line_intersection(p1, p2, p3, p4):
    """
    Finds the intersection of two line segments: (p1, p2) and (p3, p4).
    Returns the intersection point if it exists, otherwise None.
    """
    x1, y1, x2, y2 = *p1, *p2
    x3, y3, x4, y4 = *p3, *p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Lines are parallel or coincident

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    # Check if intersection is within segment bounds
    if (min(x1, x2) <= px <= max(x1, x2)) and (min(y1, y2) <= py <= max(y1, y2)) and \
            (min(x3, x4) <= px <= max(x3, x4)) and (min(y3, y4) <= py <= max(y3, y4)):
        return np.array([px, py])

    return None


def interpolate_contour(contour, num_points):
    """
    Interpolates the contour by adding additional points.
    This function interpolates the contour linearly between consecutive points.
    """
    new_contour = []
    for i in range(len(contour) - 1):
        p1 = contour[i]
        p2 = contour[i + 1]
        for j in range(num_points):
            # Interpolate points between p1 and p2
            alpha = j / num_points
            new_point = (1 - alpha) * p1 + alpha * p2
            new_contour.append(new_point)
    # Adding last point
    new_contour.append(contour[-1])
    return np.array(new_contour)


def find_best_shortest_diagonal(wound_contour, step_size=1):
    """
    Finds the longest diagonal and the most perpendicular shortest diagonal
    by sweeping a perpendicular axis and detecting contour intersections.
    """
    points = wound_contour[:, 0, :]  # Extract (x, y) coordinates

    # Compute pairwise distances
    dist_matrix = squareform(pdist(points))

    # Find longest diagonal
    max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    longest_diag_pts = points[max_idx[0]], points[max_idx[1]]

    # Compute midpoint of longest diagonal
    mid_x = (longest_diag_pts[0][0] + longest_diag_pts[1][0]) / 2
    mid_y = (longest_diag_pts[0][1] + longest_diag_pts[1][1]) / 2
    midpoint = np.array([mid_x, mid_y])

    # Compute perpendicular direction
    dx = longest_diag_pts[1][0] - longest_diag_pts[0][0]
    dy = longest_diag_pts[1][1] - longest_diag_pts[0][1]
    perp_vector = np.array([-dy, dx])  # Rotate by 90 degrees
    perp_vector = perp_vector / np.linalg.norm(perp_vector)  # Normalize

    # Create range of steps along the longest diagonal
    num_steps = int(np.linalg.norm(longest_diag_pts[1] - longest_diag_pts[0]) / step_size)
    step_points = np.linspace(longest_diag_pts[0], longest_diag_pts[1], num_steps)

    # Search for max perpendicular intersection distance
    max_shortest_distance = 0
    best_shortest_pts = None

    for step_point in step_points:
        # Define perpendicular line at current step position
        p1 = step_point + 1000 * perp_vector  # Extend line far in one direction
        p2 = step_point - 1000 * perp_vector  # Extend far in the opposite direction

        # Find intersection points of this perpendicular line with the contour
        intersections = []
        for i in range(len(points)):
            pA, pB = points[i - 1], points[i]  # Line segment from contour
            inter = line_intersection(p1, p2, pA, pB)
            if inter is not None:
                intersections.append(inter)

        # If exactly two intersection points found, compute the distance
        if len(intersections) == 2:
            d = np.linalg.norm(intersections[0] - intersections[1])
            if d > max_shortest_distance:
                max_shortest_distance = d
                best_shortest_pts = (intersections[0], intersections[1])

    return (np.array(longest_diag_pts[0], dtype=int),
            np.array(longest_diag_pts[1], dtype=int)), \
           (np.array(best_shortest_pts[0], dtype=int),
            np.array(best_shortest_pts[1], dtype=int))


def calculate_wound_dimensions(wound_mask, pixel_to_mm_ratio, image=None, multi_wound_mode="largest"):
    assert multi_wound_mode in ["largest", "sum"], f"The requestest multi-wound mode {multi_wound_mode} is not supported"

    # Find contours of the wound area
    contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0, 0, 0, 0, image

    if multi_wound_mode == "sum":
        # Area across all
        wound_area_px = sum(cv2.contourArea(cnt) for cnt in contours)
    else:
        # Area of the largest wound
        largest_contour = max(contours, key=cv2.contourArea)
        wound_area_px = cv2.contourArea(largest_contour)

    wound_area_mm2 = wound_area_px * (pixel_to_mm_ratio ** 2)

    diags = []
    diag_points = []
    for wound_contour in contours:
        if len(wound_contour) >= 7:
            longest, shortest = find_best_shortest_diagonal(wound_contour)

            # # Visualization
            # plt.figure(figsize=(6, 6))
            # plt.scatter(wound_contour[:, 0, 0], wound_contour[:, 0, 1], color='blue', label="Contour Points")
            # plt.plot([longest[0][0], longest[1][0]], [longest[0][1], longest[1][1]], 'r-', label="Longest Diagonal")
            # plt.plot([shortest[0][0], shortest[1][0]], [shortest[0][1], shortest[1][1]], 'm-',
            #          label="Best Shortest Diagonal")
            #
            # plt.xlabel("X")
            # plt.ylabel("Y")
            # plt.title("Sweeping Perpendicular Line for Shortest Diagonal")
            # plt.legend()
            # plt.grid()
            # plt.show()

            longest_diag_mm = np.linalg.norm(longest[0] - longest[1]) * pixel_to_mm_ratio
            shortest_diag_mm = np.linalg.norm(shortest[0] - shortest[1]) * pixel_to_mm_ratio

            diags.append((longest_diag_mm, shortest_diag_mm))
            diag_points.append((longest, shortest))

    if len(diags) == 1:
        wound_width_mm, wound_height_mm = diags[0]
    elif len(diags) > 1:
        if multi_wound_mode=="sum":
            wound_width_mm = sum(width for width, height in diags)
            wound_height_mm = sum(height for width, height in diags)
        else:
            # Find the index of the largest contour
            largest_index = np.argmax([cv2.contourArea(cnt) for cnt in contours])
            # Extract width and height from the corresponding diagonals
            wound_width_mm, wound_height_mm = diags[largest_index]

    else:
        wound_width_mm, wound_height_mm = 0, 0  # No wounds

    bounding_box_area_mm2 = wound_width_mm * wound_height_mm

    if image is not None:
        annotated_image = image.copy()
        cv2.drawContours(annotated_image, contours, -1, (0, 255, 0), 2)
        for diag_points_pair in diag_points:
            longest_diag_pts, shortest_diag_pts = diag_points_pair
            cv2.line(annotated_image, tuple(longest_diag_pts[0]), tuple(longest_diag_pts[1]), (255, 0, 255), 2)
            cv2.line(annotated_image, tuple(shortest_diag_pts[0]), tuple(shortest_diag_pts[1]), (255, 0, 255), 2)

        # Detect and annotate ArUco markers
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        detector_params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, detector_params)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            for corner, marker_id in zip(corners, ids):
                (topLeft, topRight, bottomRight, bottomLeft) = corner[0]

                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # draw the bounding box of the ArUCo detection
                # colors = [(0, 0, 255),(0, 255, 0),(255, 0, 0),(0, 255, 255)]      Red, Green, Blue, Yellow
                cv2.line(annotated_image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(annotated_image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(annotated_image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(annotated_image, bottomLeft, topLeft, (0, 255, 0), 2)

                center = np.mean(corner[0], axis=0).astype(int)
                # Shift the annotation slightly above and slightly left of the center
                # text_position = (center[0] - 30, center[1] - 5)

                cv2.putText(annotated_image, str(marker_id[0]), tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return wound_area_mm2, wound_width_mm, wound_height_mm, bounding_box_area_mm2, annotated_image


def process_image(image_path, mask_paths, output_dir, multi_wound_mode):
    """Process image and its masks to extract wound characteristics."""
    image = cv2.imread(image_path)
    image = resize_image(image)  # Resize image to 512x512
    image_name = os.path.basename(image_path)
    pixel_to_mm_ratio = calculate_pixel_to_mm_ratio(image, image_name)

    results = []
    for mask_path in mask_paths:
        print(mask_path)
        wound_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        wound_mask = resize_image(wound_mask)  # Resize mask to 512x512
        _, wound_mask = cv2.threshold(wound_mask, 127, 255, cv2.THRESH_BINARY)

        wound_area_mm2, wound_width_mm, wound_height_mm, bounding_box_area_mm2, annotated_image = (
            calculate_wound_dimensions(wound_mask, pixel_to_mm_ratio, image, multi_wound_mode)
        )

        annotated_name = os.path.basename(mask_path).replace(".png", "_annotated.png")
        cv2.imwrite(os.path.join(output_dir, annotated_name), annotated_image)

        results.append((os.path.basename(image_path).split(".")[0], os.path.basename(mask_path).split("_")[2],
                        wound_area_mm2, wound_width_mm, wound_height_mm, bounding_box_area_mm2))

    return results


def main(data_dir, output_csv, output_dir, multi_wound_mode):
    """Process all images and masks in a single-level directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Identify unique image filenames (e.g., "Bild_01.png", "Bild_02.png")
    image_files = sorted({f for f in os.listdir(data_dir) if f.startswith("Bild_")
                          and f.endswith(".png") and "_mask" not in f})

    data = []

    for image_file in image_files:
        image_path = os.path.join(data_dir, image_file)

        # Find all corresponding mask files (e.g., "Bild_01_modelX_mask.png")
        mask_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                      if f.startswith(image_file[:-4]) and f.endswith("_mask.png")]

        if not mask_files:
            print(f"Skipping {image_file}: No mask files found")
            continue

        data.extend(process_image(image_path, mask_files, output_dir, multi_wound_mode))

    df = pd.DataFrame(data, columns=["image_id", "model_name", "area_mm2", "width_mm", "height_mm", "area_BB_mm2"])
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# Run script
if __name__ == "__main__":
    DATA_DIR = "image_mask_data"
    MULTI_WOUND_MDOE = "largest"
    OUTPUT_CSV = f"out/wound_analysis_results_{MULTI_WOUND_MDOE}.csv"
    OUTPUT_DIR = "out/annotated_images"

    main(DATA_DIR, OUTPUT_CSV, OUTPUT_DIR, MULTI_WOUND_MDOE)

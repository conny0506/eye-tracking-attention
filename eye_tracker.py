import numpy as np
import cv2

# MediaPipe Face Mesh landmark indices
LEFT_EYE_CONTOUR  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

LEFT_EAR_POINTS  = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_POINTS = [33, 160, 158, 133, 153, 144]

LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Gözlük analizi için göz çevresi bölgesi
GLASSES_REGION_INDICES = LEFT_EYE_CONTOUR + RIGHT_EYE_CONTOUR


def eye_aspect_ratio(landmarks, eye_points, img_w, img_h):
    coords = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_points]
    v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    h  = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    return (v1 + v2) / (2.0 * h + 1e-6)


def get_iris_position(landmarks, iris_points, eye_points, img_w, img_h):
    iris_coords = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in iris_points]
    iris_center = np.mean(iris_coords, axis=0)

    eye_coords = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_points]
    eye_left   = np.array(eye_coords[0])
    eye_right  = np.array(eye_coords[3])
    eye_top    = np.array(eye_coords[1])
    eye_bottom = np.array(eye_coords[5])

    eye_width  = np.linalg.norm(eye_right - eye_left) + 1e-6
    eye_height = np.linalg.norm(eye_bottom - eye_top) + 1e-6

    h_ratio = (iris_center[0] - eye_left[0]) / eye_width * 2 - 1
    v_ratio = (iris_center[1] - eye_top[1]) / eye_height * 2 - 1
    return h_ratio, v_ratio, tuple(iris_center.astype(int))


def detect_glasses(frame, landmarks, img_w, img_h):
    """Göz çevresi kenar yoğunluğuna bakarak gözlük tespiti yapar."""
    xs = [int(landmarks[i].x * img_w) for i in GLASSES_REGION_INDICES]
    ys = [int(landmarks[i].y * img_h) for i in GLASSES_REGION_INDICES]

    pad = 18
    x1 = max(0, min(xs) - pad)
    x2 = min(img_w, max(xs) + pad)
    y1 = max(0, min(ys) - pad)
    y2 = min(img_h, max(ys) + pad)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return False, 0.0

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Gözlük camı yansıma → Laplacian varyans
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Yatay kenarlar (gözlük çerçevesi yatay baskın)
    edges = cv2.Canny(gray, 40, 120)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    h_ratio = np.sum(np.abs(sobely)) / (np.sum(np.abs(sobelx)) + 1e-6)
    edge_density = np.sum(edges > 0) / (edges.size + 1e-6)

    # Gözlük: yüksek kenar yoğunluğu + yatay ağırlık + laplacian varyans
    has_glasses = (edge_density > 0.12) and (lap_var > 80) and (h_ratio > 0.8)
    return has_glasses, edge_density


def check_eye_obstacle(left_ear, right_ear, open_threshold):
    """Bir göz normalken diğeri anormal derecede düşükse engel var demektir."""
    diff = abs(left_ear - right_ear)
    one_very_low = (left_ear < open_threshold * 0.45) or (right_ear < open_threshold * 0.45)
    other_normal = (left_ear > open_threshold * 0.75) or (right_ear > open_threshold * 0.75)
    return diff > 0.10 and one_very_low and other_normal


def analyze_eyes(frame, landmarks, img_w, img_h):
    left_ear  = eye_aspect_ratio(landmarks, LEFT_EAR_POINTS, img_w, img_h)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EAR_POINTS, img_w, img_h)
    avg_ear   = (left_ear + right_ear) / 2.0

    left_h, left_v, left_iris_pos   = get_iris_position(landmarks, LEFT_IRIS,  LEFT_EAR_POINTS,  img_w, img_h)
    right_h, right_v, right_iris_pos = get_iris_position(landmarks, RIGHT_IRIS, RIGHT_EAR_POINTS, img_w, img_h)

    gaze_h = (left_h + right_h) / 2.0
    gaze_v = (left_v + right_v) / 2.0

    has_glasses, edge_density = detect_glasses(frame, landmarks, img_w, img_h)

    return {
        "left_ear":       left_ear,
        "right_ear":      right_ear,
        "avg_ear":        avg_ear,
        "gaze_h":         gaze_h,
        "gaze_v":         gaze_v,
        "left_iris_pos":  left_iris_pos,
        "right_iris_pos": right_iris_pos,
        "has_glasses":    has_glasses,
        "edge_density":   edge_density,
    }

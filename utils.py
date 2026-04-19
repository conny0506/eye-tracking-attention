import cv2

COLORS = {
    "green":  (0, 220, 0),
    "yellow": (0, 200, 220),
    "red":    (0, 0, 220),
    "white":  (255, 255, 255),
    "black":  (0, 0, 0),
    "blue":   (220, 120, 0),
    "cyan":   (220, 200, 0),
    "bg":     (20, 20, 20),
}


def draw_panel(frame, results: dict, eye_data: dict):
    h, w = frame.shape[:2]

    # Kalibrasyon ekranı
    if results.get("calibrating"):
        _draw_calibration(frame, results, w, h)
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (290, h), COLORS["bg"], -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    score      = results["attention_score"]
    alert      = results["alert_level"]
    has_glasses= results.get("has_glasses", False)
    obstacle   = results.get("obstacle", False)

    score_color = COLORS["green"] if score >= 70 else (COLORS["yellow"] if score >= 40 else COLORS["red"])

    # Başlık
    cv2.putText(frame, "GOZ TAKIP SISTEMI", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["white"], 1)
    cv2.line(frame, (10, 35), (280, 35), COLORS["white"], 1)

    # Gözlük göstergesi
    glasses_text  = "Gozluk: VAR" if has_glasses else "Gozluk: YOK"
    glasses_color = COLORS["cyan"] if has_glasses else COLORS["white"]
    cv2.putText(frame, glasses_text, (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, glasses_color, 1)

    # Bireysel göz durumu
    left_status  = "KAPALI" if results.get("left_eye_closed")  else "ACIK"
    right_status = "KAPALI" if results.get("right_eye_closed") else "ACIK"
    l_col = COLORS["red"]   if results.get("left_eye_closed")  else COLORS["green"]
    r_col = COLORS["red"]   if results.get("right_eye_closed") else COLORS["green"]

    cv2.putText(frame, f"Sol Goz : {left_status}",  (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.46, l_col, 1)
    cv2.putText(frame, f"Sag Goz : {right_status}", (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.46, r_col, 1)

    cv2.line(frame, (10, 106), (280, 106), (60, 60, 60), 1)

    lines = [
        (f"Dikkat Skoru: {score}/100",              score_color),
        (f"Ort. Dikkat : {results['avg_attention']}/100", score_color),
        (f"Blink Sayisi: {results['blink_count']}",       COLORS["white"]),
        (f"Blink/Dk    : {results['blinks_per_minute']:.1f}", COLORS["white"]),
        (f"Bakis Yonu  : {results['gaze_direction']}",    COLORS["yellow"]),
        (f"Sol EAR     : {eye_data['left_ear']:.3f}",     COLORS["white"]),
        (f"Sag EAR     : {eye_data['right_ear']:.3f}",    COLORS["white"]),
        (f"EAR Esigi   : {results['ear_threshold']:.3f}", COLORS["white"]),
        (f"Sure        : {results['elapsed_seconds']}s",  COLORS["white"]),
    ]

    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (10, 124 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

    # Dikkat skoru bar
    bar_y = 340
    cv2.putText(frame, "Dikkat:", (10, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["white"], 1)
    cv2.rectangle(frame, (10, bar_y), (280, bar_y + 14), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, bar_y), (10 + int(270 * score / 100), bar_y + 14), score_color, -1)

    # Engel uyarısı (özel renk)
    if obstacle:
        _draw_alert_box(frame, "!! GOZLERDE ENGEL VAR !!", COLORS["red"], w, h, offset=50)

    # Genel alert
    if alert > 0 and not obstacle:
        color = COLORS["red"] if alert == 2 else COLORS["yellow"]
        _draw_alert_box(frame, results["alert_message"], color, w, h)

    # Iris noktaları
    for pos in [eye_data["left_iris_pos"], eye_data["right_iris_pos"]]:
        cv2.circle(frame, pos, 3, COLORS["blue"], -1)
        cv2.circle(frame, pos, 6, COLORS["blue"], 1)


def _draw_calibration(frame, results, w, h):
    remaining = results.get("calib_remaining", 0)
    total     = 60
    done      = total - remaining
    progress  = done / total

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cx, cy = w // 2, h // 2

    cv2.putText(frame, "KALIBRASYON", (cx - 130, cy - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 220), 2)
    cv2.putText(frame, "Gozlerinizi kameraya dogru bakin", (cx - 220, cy - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "ve dogal sekilde goz kirpin.", (cx - 180, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Progress bar
    bar_w, bar_h = 400, 22
    bx = cx - bar_w // 2
    by = cy + 20
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bar_w * progress), by + bar_h), (0, 200, 0), -1)
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (150, 150, 150), 1)

    cv2.putText(frame, f"{int(progress * 100)}%", (cx - 18, by + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if results.get("has_glasses"):
        cv2.putText(frame, "Gozluk tespit edildi", (cx - 110, cy + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 220), 1)


def _draw_alert_box(frame, message, color, w, h, offset=0):
    box_w, box_h = 500, 46
    x = (w - box_w) // 2
    y = h - 70 - offset
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), color, -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 2)
    cv2.putText(frame, message, (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

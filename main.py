import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os

from eye_tracker import analyze_eyes
from attention_analyzer import AttentionAnalyzer
from utils import draw_panel

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera acilamadi!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.7,
        min_face_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    analyzer = AttentionAnalyzer()

    with mp_vision.FaceLandmarker.create_from_options(options) as detector:
        print("Goz Takip Sistemi baslatildi.")
        print("Kalibrasyon: Gozlerinizi kameraya dogru bakin ve dogal sekilde goz kirpin.")
        print("Cikis icin 'q' tusuna basin.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                eye_data = analyze_eyes(frame, landmarks, w, h)
                results  = analyzer.update(eye_data)
                draw_panel(frame, results, eye_data)
            else:
                cv2.putText(frame, "Yuz tespit edilemedi", (w // 2 - 150, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

            cv2.imshow("Goz Takip & Dikkat Analizi", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

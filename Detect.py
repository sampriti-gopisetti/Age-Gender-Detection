import argparse
from pathlib import Path
import sys

import cv2
import dlib
import numpy as np

# Try TensorFlow Keras first (more common); fallback to standalone Keras
try:
    from tensorflow.keras.models import load_model  # type: ignore
except Exception:  # pragma: no cover - fallback path
    from keras.models import load_model  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time Age and Gender Detection from webcam feed"
    )
    default_root = Path(__file__).resolve().parent
    parser.add_argument(
        "--model",
        type=Path,
        default=default_root / "Age_Gender_Model.h5",
        help="Path to the Keras/TensorFlow .h5 model file",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=default_root / "labels.txt",
        help="Path to labels.txt containing class labels",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (0 is default webcam)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Output frame width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Output frame height",
    )
    return parser.parse_args()


def main() -> int:
    np.set_printoptions(suppress=True)

    args = parse_args()

    # Resolve and validate paths
    model_path: Path = args.model if isinstance(args.model, Path) else Path(args.model)
    labels_path: Path = args.labels if isinstance(args.labels, Path) else Path(args.labels)

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return 1
    if not labels_path.exists():
        print(f"[ERROR] Labels file not found: {labels_path}")
        return 1

    # Load model and labels
    try:
        model = load_model(str(model_path), compile=False)
    except Exception as e:
        print(f"[ERROR] Failed to load model '{model_path}': {e}")
        return 1

    try:
        class_names = labels_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        print(f"[ERROR] Failed to read labels '{labels_path}': {e}")
        return 1

    face_detector = dlib.get_frontal_face_detector()

    # On Windows, CAP_DSHOW can reduce latency for webcams
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        print(f"[ERROR] Could not open camera index {args.camera}")
        return 1

    try:
        while True:
            ret, image = camera.read()
            if not ret or image is None:
                print("[WARN] Failed to read frame from camera")
                continue

            image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_AREA)
            faces = face_detector(image)

            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()

                # Clamp coordinates to image bounds
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(image.shape[1], x + w)
                y1 = min(image.shape[0], y + h)

                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
                face_roi = image[y0:y1, x0:x1]
                if face_roi.size == 0:
                    continue

                face_resized = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_AREA)
                face_input = np.asarray(face_resized, dtype=np.float32).reshape(1, 224, 224, 3)
                face_input = (face_input / 127.5) - 1

                prediction = model.predict(face_input, verbose=0)

                # Gender is implied by label text (Female/Male); age is index within first 11 outputs
                idx = int(np.argmax(prediction))
                class_name = class_names[idx] if 0 <= idx < len(class_names) else str(idx)

                age_labels = [
                    "0-10",
                    "10-19",
                    "20-29",
                    "30-39",
                    "40-49",
                    "50-59",
                    "60-69",
                    "70-79",
                    "80-89",
                    "90-99",
                    "100-116",
                ]
                # Defensive checks on prediction shape
                try:
                    age_index = int(np.argmax(prediction[0][:11]))
                except Exception:
                    age_index = 0
                age_label = age_labels[age_index] if 0 <= age_index < len(age_labels) else "N/A"

                # Original format: f"{class_name[2:8]},{age_label}" -> keep similar
                gender_snippet = class_name[2:8] if len(class_name) >= 8 else class_name
                text = f"{gender_snippet},{age_label}"
                cv2.putText(
                    image,
                    text,
                    (x0, max(15, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Webcam Image", image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())

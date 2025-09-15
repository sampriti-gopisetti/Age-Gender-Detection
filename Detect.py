import cv2
import dlib
import numpy as np
from keras.models import load_model
np.set_printoptions(suppress=True)
model = load_model("Age_Gender_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
face_detector = dlib.get_frontal_face_detector()
camera = cv2.VideoCapture(0)
while True:
    ret, image = camera.read()
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    faces = face_detector(image)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi = image[y:y+h, x:x+w]
        if not face_roi.size:
            continue
        face = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_AREA)
        face = np.asarray(face, dtype=np.float32).reshape(1, 224, 224, 3)
        face = (face / 127.5) - 1
        prediction = model.predict(face)
        index = np.argmax(prediction)
        class_name = class_names[index]
        age_labels = ["0-10", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99", "100-116"]
        age_index = np.argmax(prediction[0][:11])
        age_label = age_labels[age_index]
        text = f"{class_name[2:8]},{age_label}"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Webcam Image", image)
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break
camera.release()
cv2.destroyAllWindows()

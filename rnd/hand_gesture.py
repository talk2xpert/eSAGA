import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os

base_path = r"C:\Users\Rinki\PycharmProjects\New folder\eSAGA"
file_path = os.path.join(base_path, "rnd", "gesture_recognizer.task")
print(file_path)

model_file = open(file_path, "rb")
model_data = model_file.read()
model_file.close()
print("Before base optiond\s")
base_options = python.BaseOptions(model_asset_buffer=model_data)
print("After model_data")
options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2)
print("BASE OPTIONSSSSSSSSS")
recognizer = vision.GestureRecognizer.create_from_options(options)
print(recognizer)


img_files=[]
img_file1 = "C:\\dataset\\pointing_up.jpg"
img_files.append(img_file1)
img_file2 = "C:\\dataset\\victory.jpg"
img_files.append(img_file2)
img_file3 = "C:\\dataset\\thumbs_up.jpg"
img_files.append(img_file3)
img_file3 = "C:\\dataset\\waving.jpg"
img_files.append(img_file3)

for img_file in img_files:

    img_to_process = cv2.imread(img_file)
    print(img_to_process)
    rgb_format_img = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_format_img)

    hand_landmarks_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_landmarks_protocol = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_protocol.landmark.extend([
                landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) for landmark in hand_landmarks.landmark
            ])
            hand_landmarks_list.append(hand_landmarks_protocol)

    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    if hand_landmarks_list:
        copied_image = img_to_process.copy()

        for landmark in hand_landmarks_list:
            mp_drawing.draw_landmarks(
                copied_image,
                landmark,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        image = mp.Image.create_from_file(img_file)
        print(image)
        recognition_result = recognizer.recognize(image)

        print(recognition_result)

        top_gesture = recognition_result.gestures[0][0]

        gesture_prediction = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
        cv2.putText(copied_image, gesture_prediction, (5, copied_image.shape[0] - 20) , cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Guess the gesture!", copied_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No hands were detected!")
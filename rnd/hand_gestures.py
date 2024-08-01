import cv2
from PIL import Image
import numpy as np
import mediapipe as mp
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


base_path = r"C:\Users\Rinki\PycharmProjects\New folder\eSAGA"
file_path = os.path.join(base_path, "rnd", "gesture_recognizer.task")
print(file_path)
hands = mp.solutions.hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
model_file = open(file_path, "rb")
model_data = model_file.read()
model_file.close()
print("Before base optiond\s")
base_options = python.BaseOptions(model_asset_buffer=model_data)
print("After model_data")
options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2)
print("BASE OPTIONSSSSSSSSS")
recognizer = vision.GestureRecognizer.create_from_options(options)
# Initialize the gesture recognizer
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load the GIF
gif_path = "C:\\dataset\\demo.gif"
gif = Image.open(gif_path)

# Function to convert PIL image to OpenCV format
def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    return open_cv_image[:, :, ::-1]  # Convert RGB to BGR

# Initialize a list to store processed frames
processed_frames = []

# Process each frame of the GIF
try:
    while True:
        # Convert the current frame to RGB and then to OpenCV format
        frame = gif.convert('RGB')
        frame_cv2 = pil_to_cv2(frame)

        # Process the frame for hand landmarks
        rgb_frame = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hand_landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_landmarks_protocol = mp.solutions.hands.HandLandmark()
                hand_landmarks_protocol.landmark.extend([
                    mp.solutions.hands.HandLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks.landmark
                ])
                hand_landmarks_list.append(hand_landmarks_protocol)

        # Draw landmarks on the frame if detected
        if hand_landmarks_list:
            copied_image = frame_cv2.copy()
            for landmark in hand_landmarks_list:
                mp_drawing.draw_landmarks(
                    copied_image,
                    landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            image = mp.Image.create_from_file(frame)
            print(image)
            recognition_result = recognizer.recognize(image)

            print(recognition_result)

            top_gesture = recognition_result.gestures[0][0]

            gesture_prediction = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
            cv2.putText(copied_image, gesture_prediction, (10, copied_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            cv2.imshow("Guess the gesture!", copied_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            processed_frames.append(copied_image)
        else:
            processed_frames.append(frame_cv2)

        # Move to the next frame
        gif.seek(gif.tell() + 1)
except EOFError:
    pass  # End of sequence

# Display processed frames using OpenCV
for i, frame in enumerate(processed_frames):
    cv2.imshow(f'Frame {i}', frame)
    cv2.waitKey(500)  # Display each frame for 500 ms

# Close all OpenCV windows
cv2.destroyAllWindows()

import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np

# Load face detector and predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\\San\\IIM\\SEM3\\GROUP-BYOP\\source\\esagav2\\eSAGA\\shape_predictor_68_face_landmarks.dat")


# Function to compute eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Constants for eye blink detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

# Initialize counters for blink detection
COUNTER = 0
TOTAL = 0

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)
        points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 48)]
        left_eye = points[0:6]
        right_eye = points[6:12]

        # Compute eye aspect ratio for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the eye aspect ratio
        ear = (left_ear + right_ear) / 2.0

        # Check if the eye aspect ratio is below the threshold
        if ear < EYE_AR_THRESH:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>NO BLINK DETECTED>>>>>>>>>>>")
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                print(">>>>>>>>>>>>>>>> BLINK DETECTED>>>>>>>>>>>")
                TOTAL += 1
            COUNTER = 0

        # Draw eyes on the frame
        cv2.drawContours(frame, [cv2.convexHull(np.array(left_eye))], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(np.array(right_eye))], -1, (0, 255, 0), 1)
        

    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' key is pressed
    if key == ord("q"):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
import cv2
import datetime
import dlib
import numpy as np

# Load face detection model and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\\San\\IIM\\SEM3\\GROUP-BYOP\\source\\esagav2\\eSAGA\\shape_predictor_68_face_landmarks.dat")

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


# Open video capture device (webcam)
video_capture = cv2.VideoCapture(0)

frame_count = 0  # Counter to track the frames

while frame_count <= 5:
    # Read a frame from the video stream
    ret, frame = video_capture.read()

    if not ret:
        break

    # Increment frame counter
    frame_count += 1

    print("Frame Count is ",frame_count)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop through each detected face
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Draw a rectangle around the face (optional)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Loop through each landmark point
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # Draw a circle around each landmark point
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
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

    # Wait for a key press
    key = cv2.waitKey(100) & 0xFF
    if key== ord('q'):
        break

    print("Frame count:", frame_count)

# Release the video capture device and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

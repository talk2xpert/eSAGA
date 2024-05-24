import cv2
import datetime
import dlib

# Load face detection model and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\\San\\IIM\\SEM3\\GROUP-BYOP\\source\\esagav2\\eSAGA\\shape_predictor_68_face_landmarks.dat")



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

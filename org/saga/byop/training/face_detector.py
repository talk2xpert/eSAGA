import dlib
import cv2
import matplotlib.pyplot as plt
# Open video capture device (webcam)
video_capture = cv2.VideoCapture(0)

frame_count = 0  # Counter to track the frames

while frame_count <= 20:
    # Read a frame from the video stream
    ret, frame = video_capture.read()

    if not ret:
        break

    # Increment frame counter
    frame_count += 1

    print("Frame Count is ",frame_count)



    # Break the loop if 'q' key is pressed
    if key == ord("q"):
        break
import cv2
import numpy as np
import time


def main():
    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Get video frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the width for the left section (60-70% of the frame width)
    left_section_width = int(frame_width * 0.65)

    # Define fonts and initial message
    font = cv2.FONT_HERSHEY_SIMPLEX
    initial_message = "Processing/Runtime Messages"

    # Start time for timestamp
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Create an empty image for the right section
        right_section_width = frame_width - left_section_width
        right_section = np.ones((frame_height, right_section_width, 3), dtype=np.uint8) * 255  # White background

        # Calculate elapsed time
        elapsed_time = int(time.time() - start_time)

        # Add the initial message and timestamp to the right section
        y0, dy = 30, 30
        lines = initial_message.split('\n') + [f"Timestamp: {elapsed_time}s"]
        for i, line in enumerate(lines):
            y = y0 + i * dy
            cv2.putText(right_section, line, (10, y), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Combine the left and right sections
        combined_frame = np.hstack((frame[:, :left_section_width], right_section))

        # Display the combined frame
        cv2.imshow("Video Frame with Messages", combined_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create a named window to get window size information
    cv2.namedWindow("Video Frame with Messages", cv2.WINDOW_NORMAL)
    main()

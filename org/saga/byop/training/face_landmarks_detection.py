import cv2
import datetime
import dlib
from org.saga.byop.exception.FaceNotFoundException import FaceNotFoundException

class face_landmarks_detection:

    def __init__(self, face_detection_model_path):
        # Load face detector and predictor from dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_detection_model_path)

    def face_landmark_detector(self,frame,draw_rectangle=True):

        # converting frame to gray scale to
        # pass to detector
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detecting the faces
        faces = self.detector(img_gray)

        if len(faces) == 0:
            raise FaceNotFoundException("No face found in the frame")

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            # Draw a rectangle around the face (optional)

            if(draw_rectangle):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                # cv2.putText(frame, "Face #{}".format(1), (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Detect facial landmarks
            landmarks = self.predictor(img_gray, face)
            # Loop through each landmark point
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

                # Draw a circle around each landmark point
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                frame
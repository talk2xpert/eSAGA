# Importing the required dependencies 
import cv2 # for video rendering 
import dlib # for face and landmark detection 
import imutils
# for calculating dist b/w the eye landmarks 
from scipy.spatial import distance as dist 
# to get the landmark ids of the left and right eyes 
# you can do this manually too 
from imutils import face_utils
import config_manager
import numpy as np
from org.saga.byop.exception.FaceNotFoundException import FaceNotFoundException


# from imutils import


class blink_detection:
	def __init__(self, model_path):
		# Variables
		#This constant value will act as a threshold value to detect the blink.
		self.blink_thresh = 0.45
		self.succ_frame =2
		#This constant value is the threshold value for the number of consecutive frames.
		self.count_frame = 0
		# This value will denote the total number of consecutive frames that will have the threshold value less than the EYE ASPECT RATIO constant.
		self.blink_count=0

		# Eye landmarks
		(self.L_start, self.L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(self.R_start, self.R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

		# Initializing the Models for Landmark and
		# face Detection
		self.detector = dlib.get_frontal_face_detector()
		self.landmark_predict = dlib.shape_predictor(model_path)

	def blink_detector(self,frame):

		result=False
		global shape, count_frame
		# converting frame to gray scale to
		# pass to detector
		img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# detecting the faces
		faces = self.detector(img_gray)

		if len(faces) == 0:
			raise FaceNotFoundException("No face found in the frame")

		for face in faces:

			# landmark detection
			shape = self.landmark_predict(img_gray, face)

			# converting the shape class directly
			# to a list of (x,y) coordinates
			shape = face_utils.shape_to_np(shape)

			# parsing the landmarks list to extract
			# lefteye and righteye landmarks--#
			lefteye = shape[self.L_start: self.L_end]
			righteye = shape[self.R_start:self.R_end]

			# Calculate the EAR
			left_EAR = calculate_EAR(lefteye)
			right_EAR = calculate_EAR(righteye)
			print(self.count_frame)
			# Avg of left and right eye EAR
			avg = (left_EAR + right_EAR) / 2
			if avg < self.blink_thresh and self.count_frame >= self.succ_frame:
				self.count_frame = 0
				cv2.putText(frame, 'Blink Detected', (30, 30),
							cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
				return True  # Return True if blink detected
			elif avg < self.blink_thresh:
				self.count_frame += 1  # Incrementing the frame count


			cv2.drawContours(frame, [cv2.convexHull(np.array(lefteye))], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [cv2.convexHull(np.array(righteye))], -1, (0, 255, 0), 1)
		return result


# defining a function to calculate the EAR
def calculate_EAR(eye):

	# calculate the vertical distances
	y1 = dist.euclidean(eye[1], eye[5])
	y2 = dist.euclidean(eye[2], eye[4])

	# calculate the horizontal distance
	x1 = dist.euclidean(eye[0], eye[3])

	# calculate the EAR
	EAR = (y1+y2) / x1
	return EAR
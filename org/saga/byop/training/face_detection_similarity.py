import dlib
import numpy as np
import cv2
import config_manager

def getFace(img):
  face_detector = dlib.get_frontal_face_detector()
  return face_detector(img, 1)[0]

def encodeFace(image):
  face_location = getFace(image)
  pose_predictor = dlib.shape_predictor(config_manager.get_face_detection_model_path())
  face_landmarks = pose_predictor(image, face_location)
  face_encoder = dlib.face_recognition_model_v1('C:\\Users\\CE00087717\\Downloads\\dlib-models-master (1)\\dlib-models-master\\dlib_face_recognition_resnet_model_v1.dat\\dlib_face_recognition_resnet_model_v1.dat\\dlib_face_recognition_resnet_model_v1.dat')
  face = dlib.get_face_chip(image, face_landmarks)
  encodings = np.array(face_encoder.compute_face_descriptor(face))
  return encodings

def getSimilarity(image1, image2):
  face1_embeddings = encodeFace(image1)
  face2_embeddings = encodeFace(image2)
  return np.linalg.norm(face1_embeddings-face2_embeddings)

img1 = cv2.imread('D:\\San\\IIM\\SEM3\\GROUP-BYOP\\test\\vot_repo\sanjay\\Capture_6.jpg')
img2 = cv2.imread('D:\\San\\IIM\\SEM3\\GROUP-BYOP\\test\\vot_repo\sanjay\\Capture_5.jpg')

distance = getSimilarity(img1,img2)
print(distance)
if distance < .6:
  print("Faces are of the same person.")
else:
  print("Faces are of different people.")
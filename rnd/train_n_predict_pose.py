import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
import joblib

# Save the trained model to a file
joblib_file = "C:\\dataset\\svc_model.pkl"
data = pd.read_csv("C:\\dataset\\dataset3.csv")
loaded_model=None;
def createModel():
        global model
        X, Y = data.iloc[:, :132], data['target']
        print(X)
        print(Y)
        model = SVC(kernel='poly')
        model.fit(X, Y)
        joblib.dump(model, joblib_file)
        print(f"Model saved to {joblib_file}")
        print("***************MODEL IS CREATED ************************")


def loadModel():
        loaded_model = joblib.load(joblib_file)
        print("*****************Model is Loaded************")
        return loaded_model

createModel()
loaded_model=loadModel()

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
path = "C:\\dataset\\DATASET\\TRAIN\\plank\\00000218.jpg"
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(img)
print("RESULTS....",results.pose_landmarks)
if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        print("*******************************")
        #print(landmarks)
        temp = []
        for j in landmarks:
                temp = temp + [j.x, j.y, j.z, j.visibility]
                #print(temp)
        y = loaded_model.predict([temp])
        print("Prdiction of this image..........",y)
        if y == 0:
            asan = "plank"
        else:
            asan = "goddess"
        print(asan)
        cv2.putText(img, asan, (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)
        cv2.imshow("image",img)
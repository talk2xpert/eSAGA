import dlib
import cv2
import os
import imutils
import matplotlib.pyplot as plt
from .blink_detection import blink_detection
from  .face_landmarks_detection import  face_landmarks_detection
import config_manager
import time
from org.saga.byop.exception.FaceNotFoundException import FaceNotFoundException
from org.saga.byop.production.utility.helper import helper

class  face_detection:

    def face_detector(self,f_count,candidate_image_directory_path,blink_detection_status=True,landmark_detection_status=False):
        # print("Cndidate Directory",candidate_image_directory_path)
        if (blink_detection_status):
         b = blink_detection(config_manager.get_face_detection_model_path())
        if (landmark_detection_status):
            face = face_landmarks_detection(config_manager.get_face_detection_model_path())
        blink_results=[]
        cam = cv2.VideoCapture(0)
        frame_count=0
        try:
            while(frame_count<=f_count):

                _, frame = cam.read()
                frame_count =frame_count+1
                if(frame_count<=10):
                    filename = f'Capture_{frame_count}.jpg'
                    frame_name = os.path.join(candidate_image_directory_path, filename)
                    print(frame_name)
                    cv2.imwrite(frame_name, frame)

                frame = imutils.resize(frame, width=640)
                if(blink_detection_status):
                    result,ear_values=b.blink_detector(frame)
                    if(result):
                        print(" **  BLINK DETECTED    **")
                        blink_results.append("Blink Detected")
                    else:
                        print(" ........NO BLINK DETECTED ")
                        blink_results.append("No Blink Detected")
                if (landmark_detection_status):
                    face.face_landmark_detector(frame)
                    #blink_results.append(result)

                cv2.imshow("Video", frame)
                # print("Frame Count",frame_count)
                #time.sleep(.50)
                if cv2.waitKey(config_manager.get_wait_time()) & 0xFF == ord('q'):
                    break

        except FaceNotFoundException as e:
            print(e)
        except ValueError as ve:
            print(ve)

        finally:
            cam.release()
            cv2.destroyAllWindows()
        return blink_results,ear_values


#f=face_detection()
#results=f.face_detector()

#most_common_value, count = helper.likelihood_estimator(results)
#print(f"The value that occurs the most is: {most_common_value} with {count} occurrences.")
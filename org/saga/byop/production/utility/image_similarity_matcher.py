from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#The imports for DLIB
from imutils import face_utils
import numpy as np

class image_similarity_matcher:

    def image_similarity_match(image_to_verify,reference_image):
      try:
        # Define a threshold (you may need to adjust this threshold based on  requirements)
        threshold = 0.4  # Adjust threshold based on the model and metric used

        # Verify faces
        result = DeepFace.verify(img1_path=image_to_verify, img2_path=reference_image, model_name='Facenet', distance_metric='cosine')
        print("..",end="")

        #return result["verified"]
        return result
      except Exception as e:
          print("Face Not Detected")
          print("Error:", e)



    def compare_faces(image_to_verify, reference_image, threshold=0.5):
        """
        Compares facial features between two images and returns a boolean indicating similarity.
        https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/readme.md
        Args:
            image_to_verify (str): Path to the image to be verified.
            reference_image (str): Path to the reference image.
            threshold (float, optional): Similarity threshold (0.0 to 1.0). Defaults to 0.5.

        Returns:
            tuple: (bool, str) - A tuple containing a boolean indicating similarity (`True` for matching, `False` for non-matching) and a message explaining the outcome.
        """

        # Load images and convert to grayscale
        img1 = cv2.imread(image_to_verify)
        img2 = cv2.imread(reference_image)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Detect faces using dlib's face detector
        detector = cv2.dnn.readNetFromCaffe("D:\\San\\IIM\\SEM3\\GROUP-BYOP\\source\\esagav2\\eSAGA\\model\\computer_vision-master\\CAFFE_DNN\\deploy.prototxt.txt", "D:\\San\\IIM\\SEM3\\GROUP-BYOP\\source\\esagav2\\eSAGA\\model\\computer_vision-master\\CAFFE_DNN\\res10_300x300_ssd_iter_140000.caffemodel")
        (h, w) = img1.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img1, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        detector.setInput(blob)
        detections = detector.forward()

        # Check if faces are detected in both images
        if len(detections) > 0 and len(cv2.dnn.detectObjects(detector, img2)) > 0:
            # Extract the first face from the bounding boxes for both images
            i = 0
            box1 = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX1, startY1, endX1, endY1) = box1.astype(int)
            face_img1 = gray1[startY1:endY1, startX1:endX1]

            i = 0
            box2 = cv2.dnn.detectObjects(detector, img2)[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX2, startY2, endX2, endY2) = box2.astype(int)
            face_img2 = gray2[startY2:endY2, startX2:endX2]

            # Detect facial landmarks
            predictor = cv2.dnn.readNetFromCaffe("shape_predictor_68_face_landmarks.caffemodel", "deploy.prototxt.txt")
            blob1 = cv2.dnn.blobFromImage(cv2.resize(face_img1, (227, 227)), 1.0, (227, 227),
                                          (78.4263377603, 87.7689143744, 114.895847746, 81.58252930))
            predictor.setInput(blob1)
            shapes1 = predictor.forward()
            shapes1 = shapes1[0].reshape(-1, 2).astype(int)
            facial_landmarks1 = shapes1 + (startX1, startY1)

            blob2 = cv2.dnn.blobFromImage(cv2.resize(face_img2, (227, 227)), 1.0, (227, 227),
                                          (78.4263377603, 87.7689143744, 114.895847746, 81.58252930))
            predictor.setInput(blob2)
            shapes2 = predictor

    def ensemble_verify(reference_image_path, verification_image_path):
        models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace"]

        # Load the images
        reference_image = cv2.imread(reference_image_path)
        verification_image = cv2.imread(verification_image_path)

        # Convert images to RGB as DeepFace uses RGB format
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        verification_image = cv2.cvtColor(verification_image, cv2.COLOR_BGR2RGB)

        # Initialize an array to store the distance results
        distances = []

        for model in models:
            # Perform face verification
            result = DeepFace.verify(
                img1_path=reference_image_path,
                img2_path=verification_image_path,
                model_name=model
            )
            distances.append(result["distance"])

        # Compute the average distance
        avg_distance = np.mean(distances)

        # Define a threshold (you may need to adjust this threshold based on your requirements)
        threshold = 0.4  # Adjust threshold based on the model and metric used

        # Check if the average distance is within the threshold
        if avg_distance <= threshold:
            print("The images are similar.")
        else:
            print("The images are not similar.")

    def generate_xai_report_summary(results, threshold=0.4):
            """
            Analyzes DeepFace results and generates a summarized XAI report.

            Args:
                results (list): List of dictionaries containing DeepFace verification results.
                threshold (float, optional): Similarity threshold for verification (default 0.4).

            Returns:
                None (prints the XAI report summary)
            """

            verified_strict = []
            verified_moderate = []
            verified_loose = []
            not_verified_strict = []
            not_verified_moderate = []
            not_verified_loose = []

            # Categorize results by threshold range
            for result in results:
                if result is None:  # Check if result is valid before appending
                    print("XAI : .... Face Not Detected ")
                elif result['verified']:
                    if result['distance'] < 0.4:
                        verified_strict.append(result)
                    elif result['distance'] < 0.5:
                        verified_moderate.append(result)
                    else:
                        verified_loose.append(result)
                else:
                    if result['distance'] >= 0.6:
                        not_verified_strict.append(result)
                    elif result['distance'] >= 0.5:
                        not_verified_moderate.append(result)
                    else:
                        not_verified_loose.append(result)

            # Generate XAI report summary
            print("\nXAI Report Summary:")

            # Verified Results
            print(f"\nVerified:")
            print(f"- Strict Threshold (0.4 - 0.5): {len(verified_strict)} results")
            print(f"  - High accuracy with low false positives, but may miss valid matches.")
            print(f"- Moderate Threshold (0.5 - 0.6): {len(verified_moderate)} results")
            print(f"  - Good balance between accuracy and error rates.")
            print(f"- Loose Threshold (0.6 - 0.7): {len(verified_loose)} results")
            print(f"  - Minimizes false negatives, but increases chance of incorrect matches.")

            # Not Verified Results
            print(f"\nNot Verified:")
            print(f"- Strict Threshold (0.6 - 0.7): {len(not_verified_strict)} results")
            print(f"  - Likely significant facial feature differences or other factors.")
            print(f"- Moderate Threshold (0.5 - 0.6): {len(not_verified_moderate)} results")
            print(f"  - Potential for variations in pose, lighting, or other factors.")
            print(f"- Loose Threshold (0.4 - 0.5): {len(not_verified_loose)} results")
            print(f"  - May include some false negatives due to stricter threshold settings.")

            # Additional Notes
            print("\nInterpretability Insights:")
            print(f"- The distance metric from DeepFace indicates the level of similarity between faces.")
            print(f"- A lower distance indicates higher similarity, potentially representing the same person.")
            print(f"- Threshold selection influences the trade-off between accuracy and error rates.")
            print(f"- Consider the specific requirements of your application when choosing a threshold.")


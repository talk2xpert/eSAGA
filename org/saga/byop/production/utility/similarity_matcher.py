from deepface import DeepFace
import cv2
import numpy as np


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


# Example usage
reference_image_path = "path/to/reference_image_without_spectacles_or_mask.jpg"
verification_image_path = "path/to/verification_image_with_spectacles_or_mask.jpg"



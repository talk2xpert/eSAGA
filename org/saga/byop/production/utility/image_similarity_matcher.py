from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class image_similarity_matcher:

    def image_similarity_match(image_to_verify,reference_image):
        # Verify faces
        result = DeepFace.verify(img1_path=image_to_verify, img2_path=reference_image, model_name='Facenet', distance_metric='cosine')
        return result["verified"]
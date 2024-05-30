import tensorflow as tf
from tensorflow.keras.models import load_model
from .spoof_evaluator import spoof_evaluator
from .helper import helper
from .image_similarity_matcher import image_similarity_matcher
import cv2

import logging
import config_manager

from tqdm import tqdm

class image_validator:

    def image_similarity_check(image_paths, reference_image):
        results = []
        verify_result=[]
        test_images=[]
        # for image_to_verify in image_paths:
        helper.plot_comparing_images(image_paths, reference_image)

        for image_to_verify in image_paths:
            img2 = cv2.imread(reference_image)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            test_images.append(img2)
            result = image_similarity_matcher.image_similarity_match(image_to_verify, reference_image)

            if result is not None:  # Check if result is valid before appending
                verify_result.append(result["verified"])
            else:
                print(" None Detected for : " , image_to_verify)


            #image_similarity_matcher.ensemble_verify(reference_image, image_to_verify)
            # # Call the comparison function
            # result, message = image_similarity_matcher.compare_faces(image_to_verify, reference_image)

            results.append(result)
        # Create the plot
        helper.plot_similarity_score(results)
        #image_similarity_matcher.generate_xai_report_summary(results)
        helper.create_pdf_report(reference_image, test_images, results, "similarity_plot.png")
        most_common_value, count = helper.likelihood_estimator(verify_result)

        return most_common_value, count

    # def image_similarity_check_27May_Issuewith arrayimag(image_paths,reference_image):
    #     results = []
    #     for image_to_verify in image_paths:
    #         print("*************IMAGE PLOT PATHS*********")
    #         print(image_to_verify)
    #         print(reference_image)
    #         helper.plot_comparing_images(image_to_verify, reference_image)
    #         result = image_similarity_matcher.image_similarity_match(image_to_verify, reference_image)
    #         print("Result of Similarity is ", result)
    #         results.append(result)
    #     most_common_value, count = helper.likelihood_estimator(results)
    #     return most_common_value, count

    def image_spoof_check(image_paths):
        # 1-load the model
        print("Loading Model");
        model_path = config_manager.get_model_path()
        model = load_model(model_path)
        print("***************Model loaded successfully**********");
        #  2-sending the model to evaluate the image
        evaluator = spoof_evaluator(model)
        print("**************Model Initialized****************")
        # image_path = '/content/drive/MyDrive/data/spoof_980.png'
        # 3-predict image for real or spoof
        label_names = evaluator.predict_images_labels(image_paths)
        most_common_value, count = helper.likelihood_estimator(label_names)
        return most_common_value, count
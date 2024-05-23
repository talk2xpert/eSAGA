import tensorflow as tf
from tensorflow.keras.models import load_model
from .spoof_evaluator import spoof_evaluator
from .helper import helper
from .image_similarity_matcher import image_similarity_matcher
import logging
import config_manager


class image_validator:

    def image_similarity_check(image_paths,reference_image):
        results = []

        for image_to_verify in image_paths:
            helper.plot_comparing_images(image_to_verify, reference_image)
            result = image_similarity_matcher.image_similarity_match(image_to_verify, reference_image)
            print("Result of Similarity is ", result)
            results.append(result)
        most_common_value, count = helper.likelihood_estimator(results)
        return most_common_value, count

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
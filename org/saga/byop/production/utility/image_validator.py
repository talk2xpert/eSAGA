import tensorflow as tf
from tensorflow.keras.models import load_model
from .spoof_evaluator import spoof_evaluator
from .helper import Helper
from .image_similarity_matcher import image_similarity_matcher
from .face_similarity import  FaceSimilarityModel
import cv2
import logging
import config_manager


class image_validator:

    def image_similarity_check(image_paths, reference_image):
        results = []
        verify_result=[]
        test_images=[]
        # for image_to_verify in image_paths:
        Helper.plot_comparing_images(image_paths, reference_image)

        for image_to_verify in image_paths:
            scores=[]
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

        f=FaceSimilarityModel()
        scores=f.get_similarity_scores(reference_image, image_paths)

        # print("Scores Received")
        # print("Scores Length " ,len(scores))
        f.plot_similarity(scores)
        print("Scores Plotted")
        # Helper.create_pdf_report_similarity(reference_image,image_paths,"similarity_plot.png",pdf_path="similarity_report.pdf")
        # lime_explanations = [f.get_lime_explanation(img) for img in image_paths]
        # #shap_values_ref, shap_values_test=f.get_shap_explanations(reference_image,image_paths)
        # print("LIME Explanation Done")
        # # Step 4: Create PDF Report
        # shap_ref_path = 'shap_explanation_reference.png'
        # shap_test_paths = [f'shap_explanation_test_{i}.png' for i in range(len(image_paths))]
        # lime_test_paths = [f'lime_explanation_test_{i}.png' for i in range(len(image_paths))]
        #
        # f.create_pdf_report(reference_image, scores, shap_ref_path, shap_test_paths,
        #                                         lime_test_paths, 'xai_report.pdf')
       # f.save_shap_explanations(shap_values_ref, shap_values_test,reference_image, image_paths)
        #print("Save Shap Explanation done")
        # Create the plot
        #helper.load_model()
        #image_paths.genrateXAIReportsSimilarity(reference_image, results, test_images)

        most_common_value, count = Helper.likelihood_estimator(verify_result)


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
        print("[STAGE 2 ] : ...........  Loading Model");
        model_path = config_manager.get_model_path()
        model = load_model(model_path)
        print("[STAGE 2 ] : ........... Model loaded successfully");
        #  2-sending the model to evaluate the image
        evaluator = spoof_evaluator(model)
        print("[STAGE 2 ] : ...........  Model Initialized for Spoof Evaluation ")
        # image_path = '/content/drive/MyDrive/data/spoof_980.png'
        # 3-predict image for real or spoof
        label_names = evaluator.predict_images_labels(image_paths)
        most_common_value, count = Helper.likelihood_estimator(label_names)
        return most_common_value, count
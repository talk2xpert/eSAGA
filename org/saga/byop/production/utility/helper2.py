# def preprocess_image(img_path):
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=(160, 160))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array
#
#
# # Use SHAP for explanations
# def get_shap_explanations(reference_image, test_images, model):
#     ref_img = preprocess_image(reference_image)
#     images = [preprocess_image(img) for img in test_images]
#     background = np.vstack(images)
#     explainer = shap.DeepExplainer(model, background)
#
#     shap_values_ref = explainer.shap_values(ref_img)
#     shap_values_test = [explainer.shap_values(preprocess_image(img)) for img in test_images]
#     return shap_values_ref, shap_values_test
#
#
# # Load your pre-trained model
# def load_model():
#     # Placeholder function to load a pre-trained model
#     # Replace with actual model loading code
#     model = DeepFace.build_model("Facenet")
#     return model
#
#
# def get_similarity_scores(results, reference_image, test_images):
#     scores = []
#     for i in enumerate(results):
#         # result = DeepFace.verify(reference_image, img, model_name="Facenet", model=model)
#         if 'distance' in results[i]:
#             scores.append((reference_image, test_images[i], results[i]['distance']))
#     return scores
#
#     # Step 4: Compile Report with Dynamic Content
#
#
# def create_pdf_report1(reference_image, similarity_scores, plot_path, shap_ref_path, shap_test_paths, output_path):
#     doc = SimpleDocTemplate(output_path, pagesize=letter)
#     styles = getSampleStyleSheet()
#     flowables = []
#
#     title = Paragraph("Deep Face Similarity XAI Report", styles['Title'])
#     flowables.append(title)
#     flowables.append(Spacer(1, 12))
#
#     description = Paragraph(
#         "This report provides an analysis of the similarity scores between the reference image and test images. "
#         "The plot below visualizes the computed similarity scores.",
#         styles['BodyText']
#     )
#     flowables.append(description)
#     flowables.append(Spacer(1, 12))
#
#     img = Image(plot_path)
#     img._restrictSize(450, 300)
#     flowables.append(img)
#     flowables.append(Spacer(1, 12))
#
#     table_data = [["Reference Image", "Test Image", "Similarity Score (Distance)"]] + [
#         [score[0], score[1], f"{score[2]:.4f}"] for score in similarity_scores
#     ]
#     table = Table(table_data)
#     flowables.append(table)
#     flowables.append(Spacer(1, 12))
#
#     explanation = Paragraph(
#         "SHAP Explanation for Reference Image", styles['BodyText']
#     )
#     flowables.append(explanation)
#     flowables.append(Spacer(1, 12))
#
#     img = Image(shap_ref_path)
#     img._restrictSize(450, 300)
#     flowables.append(img)
#     flowables.append(Spacer(1, 12))
#
#     for i, shap_path in enumerate(shap_test_paths):
#         explanation = Paragraph(
#             f"SHAP Explanation for Test Image {i + 1}", styles['BodyText']
#         )
#         flowables.append(explanation)
#         flowables.append(Spacer(1, 12))
#
#         img = Image(shap_path)
#         img._restrictSize(450, 300)
#         flowables.append(img)
#         flowables.append(Spacer(1, 12))
#
#     doc.build(flowables)
#
#
# def genrateXAIReportsSimilarity(reference_image, results, test_images):
#     scores = get_similarity_scores(results, reference_image, test_images)
#     print("Finsihed Scores", len(scores))
#     # helper.plot_similarity_score(scores)
#     # print("Plotted the scores")
#     # model = helper.load_model()
#     # print("Deep Face Model Loaded")
#     # shap_values_ref, shap_values_test = helper.get_shap_explanations(reference_image, test_images, model)
#     # print("The Shap Explanation is also loaded ")
#     # helper.save_shap_explanations(shap_values_ref, shap_values_test, reference_image, test_images)
#     # print("Saving the shap explanation")
#     # # Generate PDF report
#     # shap_ref_path = 'shap_explanation_reference.png'
#     # shap_test_paths = [f'shap_explanation_test_{i}.png' for i in range(len(test_images))]
#     # helper.create_pdf_report1(reference_image, scores, 'similarity_plot.png', shap_ref_path, shap_test_paths,
#     #                           'xai_report.pdf')
#     # print("Report is generated.....")
#     # image_similarity_matcher.generate_xai_report_summary(results)
#     # helper.create_pdf_report_similarity(reference_image, test_images, results, "similarity_plot.png")
#
#
#
#
#
#
#
# import os
# import cv2
# import matplotlib.pyplot as plt
# from collections import Counter
#
# from deepface import DeepFace
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.platypus import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
# from reportlab.lib.styles import getSampleStyleSheet
# from deepface import DeepFace
# import shap
# import tensorflow as tf
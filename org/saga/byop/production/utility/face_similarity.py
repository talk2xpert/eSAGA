import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from deepface import DeepFace


class ProxyModel:
    def __init__(self, model):
        self.model = model

    def predict(self, images):
        print("Print in Predict")
        return self.model.predict(images)
class FaceSimilarityModel:
    def __init__(self):
        self.model_name = "Facenet"
        self.model = DeepFace.build_model(self.model_name)
        self.proxy_model = ProxyModel(self.model)
        self.explainer = lime_image.LimeImageExplainer()
        print(" Lime Explainer",self.explainer)

    def preprocess_image(self, img_path):
        img = keras_image.load_img(img_path, target_size=(160, 160))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def get_similarity_scores(self, reference_image, test_images):
        scores = []
        for img_path in test_images:
            result = DeepFace.verify(reference_image, img_path, model_name=self.model_name)
            if 'distance' in result:
                scores.append((reference_image, img_path, result['distance']))
        return scores

    def get_shap_explanations(self, reference_image, test_images):
        ref_img = self.preprocess_image(reference_image)
        images = [self.preprocess_image(img) for img in test_images]
        background = np.vstack(images)
        explainer = lime_image.LimeImageExplainer()

        shap_values_ref = explainer.explain_instance(ref_img[0], self.model.predict, top_labels=5, hide_color=0, num_samples=1000)
        shap_values_test = [explainer.explain_instance(img, self.model.predict, top_labels=5, hide_color=0, num_samples=1000) for img in images]
        return shap_values_ref, shap_values_test

    def get_lime_explanation(self, image_path):
        image = keras_image.load_img(image_path, target_size=(160, 160))
        image_array = keras_image.img_to_array(image)

        explanation = self.explainer.explain_instance(image_array, self.proxy_model.predict, top_labels=5, hide_color=0,
                                                      num_samples=1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                    hide_rest=True)
        img_boundry = mark_boundaries(temp / 2 + 0.5, mask)

        return img_boundry

    def create_pdf_report(self, reference_image, similarity_scores, shap_ref_path, shap_test_paths, lime_test_paths, output_path):
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        flowables = []

        title = Paragraph("Face Similarity XAI Report", styles['Title'])
        flowables.append(title)
        flowables.append(Spacer(1, 12))

        description = Paragraph(
            "This report provides an analysis of the similarity scores between the reference image and test images, along with SHAP and LIME explanations.",
            styles['BodyText']
        )
        flowables.append(description)
        flowables.append(Spacer(1, 12))

        # Add Similarity Plot
        similarity_plot = Image('similarity_plot.png')
        similarity_plot._restrictSize(450, 300)
        flowables.append(similarity_plot)
        flowables.append(Spacer(1, 12))

        # Add SHAP Explanations
        shap_ref_img = Image(shap_ref_path)
        shap_ref_img._restrictSize(450, 300)
        flowables.append(Paragraph("SHAP Explanation for Reference Image", styles['BodyText']))
        flowables.append(shap_ref_img)
        flowables.append(Spacer(1, 12))

        for i, shap_path in enumerate(shap_test_paths):
            shap_img = Image(shap_path)
            shap_img._restrictSize(450, 300)
            flowables.append(Paragraph(f"SHAP Explanation for Test Image {i+1}", styles['BodyText']))
            flowables.append(shap_img)
            flowables.append(Spacer(1, 12))

        # Add LIME Explanations
        for i, lime_path in enumerate(lime_test_paths):
            lime_img = Image(lime_path)
            lime_img._restrictSize(450, 300)
            flowables.append(Paragraph(f"LIME Explanation for Test Image {i+1}", styles['BodyText']))
            flowables.append(lime_img)
            flowables.append(Spacer(1, 12))

        doc.build(flowables)

    def plot_similarity(self,scores):
            pairs = [f"{i[0]} vs {i[1]}" for i in scores]
            distances = [i[2] for i in scores]

            plt.figure(figsize=(10, 5))
            plt.bar(pairs, distances, color='skyblue')
            plt.xlabel('Image Pairs')
            plt.ylabel('Similarity Score (Distance)')
            plt.title('Face Similarity Scores')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig('similarity_plot.png')

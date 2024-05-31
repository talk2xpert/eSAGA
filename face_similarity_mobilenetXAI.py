import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.metrics.pairwise import cosine_similarity
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet


class MobileNetFeatureExtractor:
    def __init__(self):
        self.model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

    def extract_features(self, img_path):
        img = keras_image.load_img(img_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = self.model.predict(img_array)
        return features


class FaceSimilarityModel:
    def __init__(self):
        self.feature_extractor = MobileNetFeatureExtractor()
        self.model = self.feature_extractor.model
        self.explainer = lime_image.LimeImageExplainer()

    def compute_similarity(self, ref_features, test_features):
        return cosine_similarity(ref_features, test_features).flatten()[0]

    def get_similarity_scores(self, reference_image, test_images):
        ref_features = self.feature_extractor.extract_features(reference_image)
        scores = []
        for img_path in test_images:
            test_features = self.feature_extractor.extract_features(img_path)
            score = self.compute_similarity(ref_features, test_features)
            scores.append((reference_image, img_path, score))
        return scores

    def get_lime_explanation(self, image_path):
        image = keras_image.load_img(image_path, target_size=(224, 224))
        image_array = keras_image.img_to_array(image)

        def predict_fn(images):
            images = preprocess_input(images)
            return self.model.predict(images)

        explanation = self.explainer.explain_instance(image_array, predict_fn, top_labels=5, hide_color=0,
                                                      num_samples=1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                    hide_rest=True)
        img_boundry = mark_boundaries(temp / 2 + 0.5, mask)
        return img_boundry

    def create_pdf_report(self, reference_image, similarity_scores, lime_test_paths, output_path):
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        flowables = []

        title = Paragraph("Face Similarity XAI Report", styles['Title'])
        flowables.append(title)
        flowables.append(Spacer(1, 12))

        description = Paragraph(
            "This report provides an analysis of the similarity scores between the reference image and test images, along with LIME explanations.",
            styles['BodyText']
        )
        flowables.append(description)
        flowables.append(Spacer(1, 12))

        # Add Similarity Scores
        for ref_img, test_img, score in similarity_scores:
            text = f"Similarity score between {ref_img} and {test_img}: {score:.2f}"
            flowables.append(Paragraph(text, styles['BodyText']))
            flowables.append(Spacer(1, 12))

        # Add LIME Explanations
        for i, lime_path in enumerate(lime_test_paths):
            lime_img = Image(lime_path)
            lime_img._restrictSize(450, 300)
            flowables.append(Paragraph(f"LIME Explanation for Test Image {i + 1}", styles['BodyText']))
            flowables.append(lime_img)
            flowables.append(Spacer(1, 12))

        doc.build(flowables)


# Example usage
face_similarity_model = FaceSimilarityModel()

reference_image = "D:\\San\\IIM\\SEM3\\GROUP-BYOP\\test\\can_repo\\sanjay.jpg"
test_images = ["D:\\San\\IIM\\SEM3\\GROUP-BYOP\\test\\vot_repo\\sanjay\\Capture_1.jpg", "D:\\San\\IIM\\SEM3\\GROUP-BYOP\\test\\vot_repo\\sanjay\\Capture_2.jpg","D:\\San\\IIM\\SEM3\\GROUP-BYOP\\test\\vot_repo\\sanjay\\Capture_3.jpg"]

# Get similarity scores
similarity_scores = face_similarity_model.get_similarity_scores(reference_image, test_images)

# Generate LIME explanations
lime_explanations = []
lime_test_paths = []
for i, img in enumerate(test_images):
    explanation_img = face_similarity_model.get_lime_explanation(img)

    # Ensure image array values are in 0..1 range
    explanation_img = (explanation_img - explanation_img.min()) / (explanation_img.max() - explanation_img.min())

    lime_path = f'lime_explanation_test_{i}.png'
    plt.imsave(lime_path, explanation_img)
    lime_test_paths.append(lime_path)

# Create PDF report
face_similarity_model.create_pdf_report(reference_image, similarity_scores, lime_test_paths, 'xai_report.pdf')

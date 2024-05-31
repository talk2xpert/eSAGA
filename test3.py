import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_image
#import cv2
from skimage.segmentation import mark_boundaries
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import numpy as np

# Function to generate LIME explanations for an image
def generate_lime_explanation(image, model, num_samples=1000, top_labels=5, num_features=100):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image.astype('double'),
                                             model.predict,
                                             top_labels=top_labels,
                                             hide_color=0,
                                             num_samples=num_samples)
    return explanation

# Function to create LIME explanations for multiple images
def create_lime_explanations(images, model, output_dir, num_samples=1000, top_labels=5, num_features=100):
    lime_explanations = []
    for i, image in enumerate(images):
        explanation = generate_lime_explanation(image, model, num_samples, top_labels, num_features)
        lime_explanations.append((image, explanation))
        # Plot and save original image with LIME explanation
        plt.figure(figsize=(8, 6))
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=num_features, hide_rest=False)
        temp_normalized = (temp - temp.min()) / (temp.max() - temp.min())
        plt.imshow(mark_boundaries(temp_normalized, mask))
        plt.axis('off')
        plt.title(f"LIME Explanation for Image {i+1}")
        plt.savefig(f'{output_dir}/lime_explanation_{i+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    return lime_explanations

# Function to create PDF XAI report
def create_xai_report(model, validation_dataset, output_path, lime_images=None, num_samples=1000, top_labels=5, num_features=100):
    try:
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        title = Paragraph("eXplainable Artificial Intelligence (XAI) Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))

        # Model Summary
        model_summary = Paragraph("<b>Model Summary</b>", styles['Heading1'])
        elements.append(model_summary)
        model_info = Paragraph(str(model.summary()), styles['BodyText'])
        elements.append(model_info)
        elements.append(Spacer(1, 12))

        # Data Overview
        data_overview = Paragraph("<b>Data Overview</b>", styles['Heading1'])
        elements.append(data_overview)
        data_info = Paragraph(f"Validation samples: {len(validation_dataset)}", styles['BodyText'])
        elements.append(data_info)
        elements.append(Spacer(1, 12))

        # Evaluation Metrics
        metrics_summary = Paragraph("<b>Evaluation Metrics</b>", styles['Heading1'])
        elements.append(metrics_summary)


        # LIME Explanations
        lime_summary = Paragraph("<b>LIME Explanations</b>", styles['Heading1'])
        elements.append(lime_summary)
        if lime_images:
            lime_explanations = create_lime_explanations(lime_images, model, output_path, num_samples, top_labels, num_features)
            for i, (image, explanation) in enumerate(lime_explanations):
                lime_plot = Image(f'{output_path}/lime_explanation_{i+1}.png', width=400, height=400)
                elements.append(lime_plot)
                elements.append(Spacer(1, 12))

        # Build PDF
        doc.build(elements)
        print("XAI report generated successfully.")

    except Exception as e:
        print(f"Error creating XAI report: {e}")


model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=True)  #
# Example usage
validation_dataset ="C:\\Users\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_evaluation\\spoof"
# validation_dataset = your_validation_dataset
lime_images = ["C:\\Users\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_evaluation\\spoof\\spoof_981.png"
    , "C:\\Users\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_evaluation\\spoof\\spoof_982.png", "C:\\Users\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_evaluation\\spoof\\spoof_983.png", ...]  # List of images
output_path = "C:\\Users\\Rinki\\Desktop\\pdfs\\XAI_Report.pdf"
create_xai_report(model, validation_dataset, output_path, lime_images)
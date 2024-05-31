import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_image
from skimage.segmentation import mark_boundaries
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# Load the pre-trained MobileNet V2 model
model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=True)

def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

def predict_label(img_array):
    preds = model.predict(img_array)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]    # Predict image label

    decoded_preds1 = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]  # Get top 5 predictions
    print("Top Five " ,decoded_preds1)
    return decoded_preds[0][1]

def generate_shap_plot(img_array):
    explainer_shap = shap.GradientExplainer(model, img_array)
    shap_values, indexes = explainer_shap.shap_values(img_array, ranked_outputs=1)
    shap.image_plot(shap_values, img_array, show=False)
    plt.savefig('shap_plot.png')
    plt.close()

def generate_lime_plot(img_array):
    explainer_lime = lime.lime_image.LimeImageExplainer()
    explanation = explainer_lime.explain_instance(img_array[0], model.predict, top_labels=2, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    temp_normalized = (temp - temp.min()) / (temp.max() - temp.min())
    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(temp_normalized, mask))
    plt.axis('off')
    plt.savefig('lime_plot.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def create_pdf(image_paths):
    try:
        doc = SimpleDocTemplate("C:\\Users\\Rinki\\Desktop\\pdfs\\XAI_Report.pdf", pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        title = Paragraph("XAI Report for MobileNet Image Classification", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))

        for img_path in image_paths:
            img_array = preprocess_image(img_path)
            pred_label = predict_label(img_array)

            # Model Summary
            model_summary = Paragraph("<b>Model Summary</b>", styles['Heading1'])
            elements.append(model_summary)
            model_info = Paragraph(f"Predicted Label: {pred_label}", styles['BodyText'])
            elements.append(model_info)
            elements.append(Spacer(1, 12))

            # Generate SHAP plot
            generate_shap_plot(img_array)
            shap_summary = Paragraph("<b>SHAP Explanation</b>", styles['Heading1'])
            elements.append(shap_summary)
            shap_text = Paragraph("SHAP Explanation for the image", styles['BodyText'])
            elements.append(shap_text)
            shap_plot = Image("shap_plot.png", width=400, height=300)
            elements.append(shap_plot)
            elements.append(Spacer(1, 12))

            # Generate LIME plot
            generate_lime_plot(img_array)
            lime_summary = Paragraph("<b>LIME Explanation</b>", styles['Heading1'])
            elements.append(lime_summary)
            lime_text = Paragraph("LIME Explanation for the image", styles['BodyText'])
            elements.append(lime_text)
            lime_plot = Image("lime_plot.png", width=400, height=300)
            elements.append(lime_plot)
            elements.append(Spacer(1, 12))

        # Save the document
        doc.build(elements)

        # Print message after saving PDF
        print("PDF report has been successfully saved as 'XAI_Report.pdf'.")
    except Exception as e:
        print(f"An error occurred while creating the PDF: {e}")

# Example usage
#image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']  # List of image paths
image_paths=['C:\\Users\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_training\\real\\YOUTUBE_id99_s2_15.png',
'C:\\Users\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_evaluation\\spoof\\spoof_980.png']



create_pdf(image_paths)

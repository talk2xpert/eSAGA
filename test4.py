import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
from sklearn.metrics import confusion_matrix, classification_report
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Function to generate Lime explanations for an image
def generate_lime_explanation(image, model, num_samples=1000, top_labels=5, num_features=100):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image.astype('double'),
                                             model.predict,
                                             top_labels=top_labels,
                                             hide_color=0,
                                             num_samples=num_samples)
    return explanation

# Function to create Lime explanations for multiple images
def create_lime_explanations(img, model, num_samples=1000, top_labels=5, num_features=100):

    lime_explanations = []
    for i, image in enumerate(img):
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        explanation = generate_lime_explanation(img_array[0], model, num_samples, top_labels, num_features)
        lime_explanations.append(explanation)
    return lime_explanations

lime_images = ["C:\\Users\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_evaluation\\spoof\\spoof_981.png"
    , "C:\\Users\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_evaluation\\spoof\\spoof_982.png", "C:\\Users\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_evaluation\\spoof\\spoof_983.png", ...]  # List of images  # List of images
model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=True)  #
lime_explanations = create_lime_explanations(lime_images, model)


# Plot and save Lime explanations for each image
for i, explanation in enumerate(lime_explanations):
    # Plot and save original image with Lime explanation
    plt.figure(figsize=(8, 6))
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=100, hide_rest=False)
    temp_normalized = (temp - temp.min()) / (temp.max() - temp.min())
    plt.imshow(mark_boundaries(temp_normalized, mask))
    plt.axis('off')
    plt.title(f"Lime Explanation for Image {i+1}")
    plt.savefig(f'lime_explanation_{i+1}.png', bbox_inches='tight', pad_inches=0)
    plt.close()


# Example usage

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

train_dir = 'C:\\Users\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_training'
val_dir = 'C:\\Users\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_evaluation'
# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load training data
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation data
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Get class names
class_names = train_dataset.class_names

# Data normalization and augmentation
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# Optimize for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Load MobileNet base model (pretrained on ImageNet)
base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification head
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)  # Adjust number of classes
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=validation_dataset)

# Convert a batch of data to numpy arrays
def dataset_to_numpy(dataset, num_batches):
    images, labels = [], []
    for img_batch, label_batch in dataset.take(num_batches):
        images.append(img_batch.numpy())
        labels.append(label_batch.numpy())
    return np.concatenate(images), np.concatenate(labels)

# Get a sample of validation data for SHAP and LIME
X_val, y_val = dataset_to_numpy(validation_dataset, num_batches=5)

# Generate SHAP explanations
explainer_shap = shap.GradientExplainer(model, X_val)
shap_values = explainer_shap.shap_values(X_val[:5])  # Use first 5 images for SHAP

# Save SHAP plots
for i in range(5):
    shap.image_plot([shap_values[j][i] for j in range(len(shap_values))], X_val[i:i+1], show=False)
    plt.savefig(f'shap_plot_{i}.png')
    plt.close()

# Generate LIME explanations for the same 5 images
explainer_lime = lime_image.LimeImageExplainer()
for i in range(5):
    explanation = explainer_lime.explain_instance(X_val[i].astype('double'), model.predict, top_labels=2, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    temp_normalized = (temp - temp.min()) / (temp.max() - temp.min())
    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(temp_normalized, mask))
    plt.axis('off')
    plt.savefig(f'lime_plot_{i}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

# Predictions and performance metrics
y_pred = np.argmax(model.predict(X_val), axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)

# Classification report
class_report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)

# Function to create PDF
def create_pdf():
    try:
        doc = SimpleDocTemplate("XAI_Report.pdf", pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        title = Paragraph("XAI Report for MobileNet Image Classification", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))

        # Model Summary
        model_summary = Paragraph("<b>Model Summary</b>", styles['Heading1'])
        elements.append(model_summary)
        model_info = Paragraph(f"TensorFlow Version: {tf.__version__}", styles['BodyText'])
        elements.append(model_info)
        elements.append(Spacer(1, 12))

        # Data Overview
        data_overview = Paragraph("<b>Data Overview</b>", styles['Heading1'])
        elements.append(data_overview)
        data_info = Paragraph(f"Training samples: {len(train_dataset)}<br/>Validation samples: {len(validation_dataset)}", styles['BodyText'])
        elements.append(data_info)
        elements.append(Spacer(1, 12))

        # Evaluation Metrics
        metrics_summary = Paragraph("<b>Evaluation Metrics</b>", styles['Heading1'])
        elements.append(metrics_summary)

        # Add classification report table
        metrics_table_data = [["Class", "Precision", "Recall", "F1-Score", "Support"]]
        for label, metrics in class_report.items():
            if isinstance(metrics, dict):
                metrics_table_data.append([
                    label,
                    f"{metrics['precision']:.2f}",
                    f"{metrics['recall']:.2f}",
                    f"{metrics['f1-score']:.2f}",
                    metrics['support']
                ])
        table = Table(metrics_table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        # Confusion Matrix
        conf_matrix_summary = Paragraph("<b>Confusion Matrix</b>", styles['Heading'])

        conf_matrix_summary = Paragraph("<b>Confusion Matrix</b>", styles['Heading1'])
        elements.append(conf_matrix_summary)

        # Add confusion matrix image
        conf_matrix_img = Image("confusion_matrix.png", width=400, height=400)
        elements.append(conf_matrix_img)
        elements.append(Spacer(1, 12))

        # SHAP Explanations
        shap_summary = Paragraph("<b>SHAP Explanations</b>", styles['Heading1'])
        elements.append(shap_summary)

        # Add SHAP plots
        for i in range(5):
            shap_plot = Image(f"shap_plot_{i}.png", width=400, height=400)
        elements.append(shap_plot)
        elements.append(Spacer(1, 12))

        # LIME Explanations
        lime_summary = Paragraph("<b>LIME Explanations</b>", styles['Heading1'])
        elements.append(lime_summary)

        # Add LIME plots
        for i in range(5):
            lime_plot = Image(f"lime_plot_{i}.png", width=400, height=400)
        elements.append(lime_plot)
        elements.append(Spacer(1, 12))

        # Build PDF
        doc.build(elements)
        print("PDF report generated successfully.")

    except Exception as e:
        print(f"Error creating PDF: {e}")

    # Create PDF report
    create_pdf()

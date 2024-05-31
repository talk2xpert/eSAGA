import os
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from deepface import DeepFace
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
import numpy as np
import shap
import tensorflow as tf

class Helper:
    @staticmethod
    def load_images_from_dir(directory):
        path = []
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
                filepath = os.path.join(directory, filename)
                path.append(filepath)
        return path

    @staticmethod
    def plot_comparing_images_org(image_to_verify, reference_image):
        img1 = cv2.imread(image_to_verify)
        img2 = cv2.imread(reference_image)
        plt.imshow(img1)
        plt.axis('off')  # Hide axis
        plt.show()
        plt.imshow(img2)
        plt.axis('off')  # Hide axis
        plt.show()

    @staticmethod
    def plot_comparing_images(images_to_verify, reference_image):
        # Read and convert the reference image
        img2 = cv2.imread(reference_image)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Number of images to verify
        num_images = len(images_to_verify)

        print("Number of Images for Validation: ", num_images)

        # Calculate the number of rows needed to fit all images in a 4-column layout, adding one row for the reference image
        num_rows = (num_images + 3) // 4 + 1  # Ceiling division to account for any extra images and add one row for the reference

        # Create subplots with the calculated number of rows and 4 columns
        fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 5))

        # If there's only one row for images to verify, axes is not a list of lists, so we need to handle this case
        if num_rows == 1:
            axes = [axes]

        # Display the reference image in the first row
        axes[0][0].imshow(img2)
        axes[0][0].axis('off')
        axes[0][0].set_title('Reference Image')

        # Hide the other subplots in the first row
        for j in range(1, 4):
            axes[0][j].axis('off')

        # Plot the images to verify
        for idx, img_path in enumerate(images_to_verify):
            row = (idx // 4) + 1  # Start from the second row
            col = idx % 4
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[row][col].imshow(img)
            axes[row][col].axis('off')
            axes[row][col].set_title(f'Image {idx + 1}')

        # Hide any remaining empty subplots
        for j in range(len(images_to_verify) % 4, 4):
            axes[-1][j].axis('off')

        # Adjust layout
        plt.tight_layout()
        # Show the plot
        plt.show()

    @staticmethod
    def plot_comparing_images_working_27May(image_to_verify, reference_image):
        # Read the images using OpenCV
        img1 = cv2.imread(image_to_verify)
        img2 = cv2.imread(reference_image)

        # Convert the images from BGR to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns, adjust the size as needed

        # Display the first image
        axes[0].imshow(img1)
        axes[0].axis('off')  # Hide axis
        axes[0].set_title('Image to Verify')

        # Display the second image
        axes[1].imshow(img2)
        axes[1].axis('off')  # Hide axis
        axes[1].set_title('Reference Image')

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.show()

    @staticmethod
    def likelihood_estimator(data_array):
        # Count occurrences of each value
        counter = Counter(data_array)
        # Find the value with the maximum count
        most_common_value, count = counter.most_common(1)[0]
        return most_common_value, count

    def create_pdf_report_similarity(reference_image, test_images, similarity_results, similarity_plot_path,
                                     pdf_path="similarity_report.pdf"):
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica", 12)

        # Title
        c.drawString(100, 750, "Face Similarity Report")

        # Summary
        y = 730
        for i, result in enumerate(similarity_results):
            c.drawString(100, y,
                         f"Image {i + 1} - Similarity Score: {result['distance']:.2f}, Verified: {'Yes' if result['verified'] else 'No'}")
            y -= 20

        # Explanation text
        explanation_text = """
        This report provides the results of face similarity analysis performed using DeepFace.
        The similarity score indicates the distance between facial embeddings of the reference image and each test image.
        A lower distance indicates higher similarity. The threshold used to determine if the faces are of the same person is 0.4.
        """
        text_lines = explanation_text.split('\n')
        y -= 20
        for line in text_lines:
            c.drawString(100, y, line.strip())
            y -= 20

        # Add plot image
        c.drawImage(similarity_plot_path, 100, y - 320, width=400, height=300)

        # Add input images
        y -= 340
        c.drawString(100, y, "Reference Image:")
        y -= 20
        c.drawImage(reference_image, 100, y - 120, width=200, height=200)

        y -= 140
        c.drawString(100, y, "Test Images:")
        y -= 20
        x = 100
        for i, test_image in enumerate(test_images):
            c.drawImage(test_image, x, y - 120, width=100, height=100)
            x += 120
            if x > 450:
                x = 100
                y -= 140
                if y < 50:
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y = 750

        c.save()
        print(f"PDF report '{pdf_path}' created successfully.")

    @staticmethod
    def plot_ear_values(ear_values, output_path="ear_plot.png"):
        plt.figure(figsize=(10, 5))
        plt.plot(ear_values, label='EAR Value')
        plt.axhline(y=0.2, color='r', linestyle='--', label='Blink Threshold')
        plt.title('Eye Aspect Ratio (EAR) Over Time')
        plt.xlabel('Frame')
        plt.ylabel('EAR')
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved as '{output_path}'")

    @staticmethod
    def create_pdf_report(blink_count, ear_plot_path, ear_values, pdf_path="blink_detection_report.pdf"):
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica", 12)

        # Title
        c.drawString(100, 750, "Blink Detection Report")

        # Summary
        c.drawString(100, 730, f"Total Blinks Detected: {blink_count}")

        # Explanation text
        explanation_text = """
        This report provides the results of blink detection performed using dlib.
        Blinks were detected based on the Eye Aspect Ratio (EAR) calculated from 
        facial landmarks. The EAR threshold and the number of consecutive frames 
        were set to ensure accurate blink detection.
        """
        text_lines = explanation_text.split('\n')
        y = 700
        for line in text_lines:
            c.drawString(100, y, line.strip())
            y -= 20

        # Add plot image
        c.drawImage(ear_plot_path, 100, y - 300, width=400, height=300)

        # Add EAR values table (optional

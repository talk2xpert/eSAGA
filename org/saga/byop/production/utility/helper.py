import os
import cv2
import matplotlib.pyplot as plt
from collections import Counter

class helper:

  def load_images_from_dir(directory):
    path = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
            filepath = os.path.join(directory, filename)
            path.append(filepath)
    return path

  def plot_comparing_images_org(image_to_verify,reference_image):
      img1 = cv2.imread(image_to_verify)
      img2 = cv2.imread(reference_image)
      plt.imshow(img1)
      plt.axis('off')  # Hide axis
      plt.show()
      plt.imshow(img2)
      plt.axis('off')  # Hide axis
      plt.show()

  def plot_comparing_images(image_to_verify, reference_image):
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

  def likelihood_estimator(data_array):
      # Count occurrences of each value
      counter = Counter(data_array)
      # Compare counts
      #Find the value with the maximum count
      most_common_value, count = counter.most_common(1)[0]
      #print(f"The value that occurs the most is: {most_common_value} with {count} occurrences.")
      return  most_common_value,count

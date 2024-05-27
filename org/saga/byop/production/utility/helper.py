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

  def plot_comparing_images(images_to_verify, reference_image):
      # Read and convert the reference image
      img2 = cv2.imread(reference_image)
      img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

      # print("Image path to reference_image : ", reference_image)

      # Number of images to verify
      num_images = len(images_to_verify)

      print("Number of Images for Validation : ", num_images)

      # Calculate the number of rows needed to fit all images in a 4-column layout, adding one row for the reference image
      num_rows = (
                         num_images + 3) // 4 + 1  # Ceiling division to account for any extra images and add one row for the reference

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
          # print("Image path to read: " ,img_path)
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

  def likelihood_estimator(data_array):
      # Count occurrences of each value
      counter = Counter(data_array)
      # Compare counts
      #Find the value with the maximum count
      most_common_value, count = counter.most_common(1)[0]
      #print(f"The value that occurs the most is: {most_common_value} with {count} occurrences.")
      return  most_common_value,count

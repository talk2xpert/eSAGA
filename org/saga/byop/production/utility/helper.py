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

  def plot_comparing_images(images_to_verify, reference_image):
      # Read and convert the reference image
      img2 = cv2.imread(reference_image)
      img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

      # Number of images to verify
      num_images = len(images_to_verify)

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



  def plot_comparing_images_lastgood(images_to_verify, reference_image):
      # Read and convert the reference image
      img2 = cv2.imread(reference_image)
      img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

      # Number of images to verify
      num_images = len(images_to_verify)

      # Calculate the number of rows needed to fit all images in a 4-column layout
      num_rows = (num_images + 3) // 4  # Ceiling division to account for any extra images

      # Create subplots with the calculated number of rows and 4 columns
      fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 5))

      # If there's only one row, axes is not a list of lists, so we need to handle this case
      if num_rows == 1:
          axes = [axes]

      # Plot the images
      for i in range(num_rows):
          for j in range(4):
              # Calculate the index of the current image
              idx = i * 4 + j
              if idx < num_images:
                  # Display the reference image
                  if j == 0:
                      axes[i][j].imshow(img2)
                      axes[i][j].axis('off')
                      axes[i][j].set_title('Reference Image')
                  else:
                      # Display images to verify
                      img_path = images_to_verify[idx]
                      img = cv2.imread(img_path)
                      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                      axes[i][j].imshow(img)
                      axes[i][j].axis('off')
                      axes[i][j].set_title(f'Image {idx + 1}')
              else:
                  # Hide the extra subplot axes
                  axes[i][j].axis('off')

      # Adjust layout
      plt.tight_layout()
      # Show the plot
      plt.show()


  def plot_comparing_images_org1(images_to_verify, reference_image):
      img2 = cv2.imread(reference_image)
      img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

      # Create subplots
      num_images = len(images_to_verify)
      fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 6))

      # Display the reference image for each row
      for i in range(num_images):
          axes[i, 0].imshow(img2)
          axes[i, 0].axis('off')
          #axes[i, 0].set_title('Reference Image')

      # Display images to verify
      for i, img_path in enumerate(images_to_verify):
          img = cv2.imread(img_path)
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          axes[i, 1].imshow(img)
          axes[i, 1].axis('off')
          #axes[i, 1].set_title(f'Image {i + 1}')
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

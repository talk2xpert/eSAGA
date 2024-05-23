import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input


class spoof_evaluator:

      def __init__(self, model, target_size=(180, 180)):
          self.model = model;
          self.target_size = target_size

      def preprocess_image(self, image_path):
          img = load_img(image_path, target_size=self.target_size)
          img_array = img_to_array(img)
          img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
          img_array = img_array / 255.0  # Normalize the image
          return img_array

      # this is  a function to predict the image label
      # this is  a function to predict the image label
      def predict_image_label(self, image_path):

          image_array = self.preprocess_image(image_path)

          # Make predictions using your trained model
          predictions = self.model.predict(image_array)

          # Decode predictions
          label_index = np.argmax(predictions)

          # Depending on how you trained your model, you might have a label mapping to decode the index
          # If you have a label mapping, you can use it to get the label name
          label_name = "Your Label"  # Replace this with your actual label name
          confidence = predictions[0][label_index]  # Get confidence score for the predicted label
          confidence_percent = confidence * 100
          label_name = "REAL"
          if (confidence_percent < 50):
              label_name = "SPOOF"

          return label_name, confidence_percent

      def predict_images_labels(self, paths):
          print("Getting labels of images")
          label_names = []
          for path in paths:
              label_name = self.predict_image_label(path)
              label_names.append(label_name)
          return label_names





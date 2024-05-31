from keras.applications import mobilenet_v2
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import config_manager


class PrintCallback(keras.callbacks.Callback):
    def __init__(self, num_epochs=10):
        self.num_epochs = num_epochs

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch No. {epoch + 1}/{self.num_epochs}")

    def on_batch_begin(self, batch, logs=None):
        print(f"\tBatch No. {batch + 1}")


class fraud_dectection_model_training_mobilenet:

        def __init__(self, folder_path, target_size=(180, 180), batch_size=32):
            self.model = None
            self.target_size = target_size
            self.batch_size = batch_size
            validation_split = 0.2
            self.test_generator = None
            datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=validation_split)

            # Data augmentation
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=validation_split
            )

            self.train_generator = train_datagen.flow_from_directory(
                folder_path,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='binary',  # Binary classification
                subset='training'
            )

            self.validation_generator = datagen.flow_from_directory(
                folder_path,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='binary',  # Binary classification
                subset='validation'
            )

            print("Train DS", len(self.train_generator))
            print("Validation DS", len(self.validation_generator))
            self.class_names = list(self.train_generator.class_indices.keys())
            print(self.class_names)
            self.num_classes = len(self.train_generator.class_indices)

        def build_model(self):
            base_model = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)  # Single output neuron with sigmoid activation
            self.model = Model(inputs=base_model.input, outputs=predictions)

        def compile_model(self, optimizer='adam', loss='binary_crossentropy',
                          metrics=['accuracy']):  # Binary crossentropy loss
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            print(self.model.summary())

        def fit_model(self, checkpoint_filepath, epochs=10):
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True
            )
            print_callback = PrintCallback(num_epochs=epochs)
            history = self.model.fit(
                self.train_generator,
                epochs=epochs,
                validation_data=self.validation_generator,
                callbacks=[model_checkpoint_callback, print_callback]
            )

            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.ylabel('Accuracy')
            plt.ylim([min(plt.ylim()), 1])
            plt.title('Training and Validation Accuracy')

            plt.subplot(2, 1, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.ylabel('Cross Entropy')
            plt.ylim([0, 1.0])
            plt.title('Training and Validation Loss')
            plt.xlabel('epoch')
            plt.show()

        def run_training(self, folder_path, checkpoint_filepath, epochs=10):
            self.__init__(folder_path)
            self.build_model()
            self.compile_model()
            self.fit_model(checkpoint_filepath, epochs)



        def predict(self):
            image_batch, label_batch = self.test_generator.next()
            predictions = self.model.predict_on_batch(image_batch)

            print('Predictions:\n', predictions)
            print('Labels:\n', label_batch)

            plt.figure(figsize=(10, 10))
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(image_batch[i].astype("uint8"))
                plt.title(
                    self.class_names[int(predictions[i] > 0.5)])  # Display the correct class name based on threshold
                plt.axis("off")

        import matplotlib.pyplot as plt
        import numpy as np
        from tensorflow.keras.preprocessing.image import DirectoryIterator

        def predict_and_plot(self):
            # Iterate over the test generator
            for image_batch, label_batch in self.test_generator:
                # Make predictions on the batch
                predictions = self.model.predict_on_batch(image_batch)

                # Print predictions and labels
                print('Predictions:\n', predictions)
                print('Labels:\n', label_batch)

                # Plot the images with their predicted class names
                plt.figure(figsize=(10, 10))
                for i in range(9):
                    ax = plt.subplot(3, 3, i + 1)
                    plt.imshow(image_batch[i].astype("uint8"))
                    plt.title(self.class_names[
                                  int(predictions[i] > 0.5)])  # Display the correct class name based on threshold
                    plt.axis("off")

                # Show the plot
                plt.show()

                # Break after one batch to prevent plotting all batches in the loop
                break

        # Usage example (assuming 'self' is an instance with appropriate attributes)
        # self.predict_and_plot()

        def save_model(self, model_path):
            self.model.save("D:\\San\\IIM\\SEM3\\GROUP-BYOP\\model\\finalized-30May2024.h5")
            print(f"Model saved to {model_path}")

        def load_model(self, model_path):
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")

        def evaluate_plot(self,test_data_dir):

            test_datagen = ImageDataGenerator(rescale=1.0 / 255)
            self.test_generator = test_datagen.flow_from_directory(
                test_data_dir,
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode='binary'  # Binary classification
            )

            y_true = self.test_generator.classes
            y_pred_prob = self.model.predict(self.test_generator)
            y_pred = (y_pred_prob > 0.5).astype(int)  # Adjust threshold if needed
            # Classification report
            print('Classification Report:')
            print(classification_report(y_true, y_pred))

            # Confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            print('Confusion Matrix:')
            print(conf_matrix)

            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            roc_auc = roc_auc_score(y_true, y_pred_prob)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.show()

            precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)

            plt.figure()
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.show()

            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.show()

            # Visualize predictions with images
            class_names = list(self.test_generator.class_indices.keys())  # Get class names

            # Plot a batch of images with predictions
            for image_batch, label_batch in self.test_generator:
                predictions = self.model.predict_on_batch(image_batch)
                plt.figure(figsize=(10, 10))
                for i in range(min(9, len(image_batch))):
                    ax = plt.subplot(3, 3, i + 1)
                    plt.imshow(image_batch[i].astype("uint8"))
                    plt.title(class_names[int(predictions[i] > 0.5)])  # Adjust for binary classification
                    plt.axis("off")
                plt.show()
                break  # Remove this to plot more batches


# Example usage
if __name__ == "__main__":
    #folder_path = config_manager.get_train_dataset_path()
    folder_path="D:\\San\\IIM\\SEM3\\GROUP-BYOP\\data"
    print("*************************",folder_path)
    checkpoint_filepath = "D:\\San\\IIM\\SEM3\\GROUP-BYOP\\model\\tmp.keras"
        #config_manager.get_checkpoint_filepath())

    epochs = 10
    model_save_path = "D:\\San\\IIM\\SEM3\\GROUP-BYOP\\model\\finalized-30May2024.h5"
        #config_manager.get_training_model_path())

    classifier = fraud_dectection_model_training_mobilenet(folder_path)
    #classifier.run_training(folder_path, checkpoint_filepath, epochs)
    #classifier.save_model(model_save_path)
    classifier.load_model(model_save_path)
    #classifier.evaluate("D:\\San\\IIM\\SEM3\\GROUP-BYOP\\source\\data\\LCC_FASD\\LCC_FASD_development")
    classifier.evaluate_plot("D:\\San\\IIM\\SEM3\\GROUP-BYOP\\source\\data\\LCC_FASD\\LCC_FASD_development")
    #classifier.predict_and_plot()

    # Load the model and evaluate again to confirm it was saved and loaded correctly

    #
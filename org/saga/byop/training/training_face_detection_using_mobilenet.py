from keras.applications import mobilenet_v2
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from org.saga.byop.training import PrintCallback


class ImageClassifier:
    def __init__(self, folder_path,target_size=(180, 180),batch_size=32):
        self.model = None
        self.target_size=target_size
        self.batch_size=batch_size
        validation_split = .2
        self.test_generator=None
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
            validation_split=validation_split)

        self.train_generator = train_datagen.flow_from_directory(
            folder_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training')  # Use 'training' subset for training data

        self.validation_generator = datagen.flow_from_directory(
            folder_path,
            target_size=(180, 180),
            batch_size=32,
            class_mode='categorical',
            subset='validation')  # Use 'validation' subset for validation data

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
        predictions = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)

    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],EPOCHS=10):

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(self.model.summary())

    def fit_model(self,checkpoint_filepath,EPOCHS=10):

        #checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        print_callback = PrintCallback()
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

    def run_training(self):
        ic = ImageClassifier("D:\\San\\IIM\\SEM3\\GROUP-BYOP\\test\\model")
        ic.build_model()
        ic.compile_model()
        ic.fit_model()

    def evaluate(self, test_data_dir, batch_size=32):
        # Test data generator
        # Evaluate the model
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical')
        return self.model.evaluate(test_generator)

    def predict(self, image):

    # Retrieve a batch of images from the test set
    image_batch, label_batch = self.test_generator.as_numpy_iterator().next()
    predictions = self.model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(class_names[predictions[i]])
        plt.axis("off")



    # Define a callback function to print information during training
    class PrintCallback(tf.keras.callbacks.Callback,num_epochs=10):
        def on_epoch_begin(self, epoch, logs=None):
            print(f"Epoch No. {epoch + 1}/{num_epochs}")

        def on_batch_begin(self, batch, logs=None):
            print(f"\tBatch No. {batch + 1}")

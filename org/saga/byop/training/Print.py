import tensorflow as tf
class PrintCallback(tf.keras.callbacks.Callback, num_epochs=10):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch No. {epoch + 1}/{num_epochs}")

    def on_batch_begin(self, batch, logs=None):
        print(f"\tBatch No. {batch + 1}")
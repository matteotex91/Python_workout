import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt


class NeuralEngine:
    def __init__(self, path: str = None):
        if path is not None:
            self.model = tf.keras.models.load_model(path)
        else:
            (ds_train, ds_test), ds_info = tfds.load(
                "mnist",
                split=["train", "test"],
                shuffle_files=True,
                as_supervised=True,
                with_info=True,
            )

            self.ds_train = ds_train
            self.ds_test = ds_test

            def normalize_img(image, label):
                """Normalizes images: `uint8` -> `float32`."""
                return tf.cast(image, tf.float32) / 255.0, label

            ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
            ds_train = ds_train.cache()
            ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
            ds_train = ds_train.batch(128)
            ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

            ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
            ds_test = ds_test.batch(128)
            ds_test = ds_test.cache()
            ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

            model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Flatten(input_shape=(28, 28)),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dense(10),
                ]
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            )

            model.fit(
                ds_train,
                epochs=6,
                validation_data=ds_test,
            )

            self.model = model

            # weights = model.layers[1].get_weights()[0]
            # biases = model.layers[2].get_weights()[1]
            # for row in weights.T:
            #     plt.pcolormesh(np.reshape(row, (28, 28)))
            #     plt.show()

    def predict(self, image: np.ndarray) -> int:
        wrapped_image = np.zeros((1, *image.shape, 1))
        wrapped_image[0, :, :, 0] = image / 255.0
        prediction = self.model.predict(wrapped_image)
        return np.argmax(prediction[0])

    def prediction_translation(self, image: np.ndarray, number: int) -> np.ndarray:
        padded_image = np.pad(image, ((27, 27), (27, 27)))
        pred_acc = np.zeros((28 + 27, 28 + 27))
        for i in range(27 + 28):
            for j in range(27 + 28):
                wrap = np.zeros((1, 28, 28, 1))
                wrap[0, :, :, 0] = padded_image[i : i + 28, j : j + 28] / 255
                # pred_acc[i, j] = self.model.predict(wrap)[0, number]
                pred_acc[i, j] = np.argmax(self.model.predict(wrap)[0])
        return pred_acc


if __name__ == "__main__":
    ne = NeuralEngine()
    test_img = np.array(list(ne.ds_train)[1][0][:, :, 0])
    test_num = int(list(ne.ds_train)[1][1])
    tr_pr = ne.prediction_translation(test_img, test_num)
    plt.pcolormesh(tr_pr)
    plt.show()

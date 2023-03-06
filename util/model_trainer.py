import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Callable, Any

RANDOM_SEED = 42
DATA_LEN = 21 * 3

class ModelTrainer():
    def __init__(self, data:np.ndarray, labels:np.ndarray, classes_num:int, model_save_path:str, trained_callback: Callable[[Any], None]) -> None:
        self.data = data
        self.lables = labels
        self.classes_num = classes_num
        self.model_save_path = model_save_path
        self.trained_callback: Callable[[Any], None] = trained_callback
        print(data)
        print(labels)

    def train(self) -> None:
        data_train, data_test, label_train, label_test = train_test_split(
            self.data, self.lables, 
            train_size=0.75, 
            random_state=RANDOM_SEED
        )
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input((DATA_LEN, )),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(self.classes_num, activation='softmax')
        ])
        model.summary()

        # Model checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            self.model_save_path, verbose=1, save_weights_only=False)
        # Callback for early stopping
        es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

        # Model compilation
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        # train
        history = model.fit(
            data_train,
            label_train,
            epochs=1000,
            batch_size=128,
            validation_data=(data_test, label_test),
            callbacks=[cp_callback, es_callback]
        )
        self.trained_callback(history)
        model.save(self.model_save_path)
        print("训练完成")
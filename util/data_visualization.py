import matplotlib.pyplot as plt
from typing import List
import numpy as np

def show_dataline_img(dataset: List[List[float]]):
    for dataline in dataset:
        one_record = np.array(dataline)
        one_record = one_record * 1000
        
        x = one_record[::3]
        y = one_record[1::3]
        z = one_record[2::3]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, color="red")
        ax.set_aspect('equal')

        plt.xlabel('x')
        plt.ylabel('y')
    plt.show()


def show_history(history):
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    fig.set_size_inches(16,5)

    epoch_num = len(train_acc)
    epochs = [i for i in range(epoch_num)]

    ax[0].plot(epochs[0:epoch_num:10] , train_acc[0:epoch_num:10] , 'go-' , label = 'Training Accuracy')
    ax[0].plot(epochs[0:epoch_num:10] , val_acc[0:epoch_num:10] , 'ro-' , label = 'Testing Accuracy')
    ax[0].set_title('Training & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs[0:epoch_num:10] , train_loss[0:epoch_num:10] , 'g-o' , label = 'Training Loss')
    ax[1].plot(epochs[0:epoch_num:10] , val_loss[0:epoch_num:10] , 'r-o' , label = 'Testing Loss')
    ax[1].set_title('Testing Accuracy & Loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    plt.show()
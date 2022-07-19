import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import scipy.io as sio
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import os
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):

    IMG_SHAPE = (224, 224, 3)
    label_names_len = 102
    restnet = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    # for layer in restnet.layers:
    #     layer.trainable = False

    # option 1
    # resnet_model = Sequential()
    # resnet_model.add(restnet)
    # resnet_model.add(Flatten())
    # resnet_model.add(Dense(512, activation='relu'))
    # resnet_model.add(Dense(label_names_len, activation='softmax'))

    # option 2
    x = restnet.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(102, activation='softmax')(x)
    resnet_model = tf.keras.Model(inputs=restnet.input, outputs=predictions)

    for layer in restnet.layers:
        layer.trainable = False

    resnet_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss="CategoricalCrossentropy",
                  metrics=["accuracy"])
    return resnet_model


def run(random_seed):
    root = "./"
    imglabel_map = os.path.join(root, '/sise/home/efrco/modelim-compute/shap/ann/imagelabels.mat')
    setid_map = os.path.join(root, '/sise/home/efrco/modelim-compute/shap/ann/setid.mat')
    imagelabels = sio.loadmat(imglabel_map)['labels'][0]
    setids = sio.loadmat(setid_map)
    ids = np.concatenate([setids['trnid'][0], setids['valid'][0],setids['tstid'][0]])
    labels = []
    image_path = []
    for i in ids:
        labels.append(int(imagelabels[i-1])-1)
        image_path.append( os.path.join(root, 'jpg', 'image_{:05d}.jpg'.format(i)))


    # Creat dataframe
    df = pd.DataFrame({'imagePath':image_path, 'label':labels})

    # split df into train(50%) and valid, test (25%)
    df.label = df.label.astype(str)
    train,test = train_test_split(df, test_size=0.50, random_state=random_seed)
    test, valid = train_test_split(test, test_size=0.5, random_state=random_seed)

    train_datagen = ImageDataGenerator(
        # rescale = 1./255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )
    valid_datagen = ImageDataGenerator(
        # rescale = 1./255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(
        # rescale = 1./255,
        featurewise_center=True,
        featurewise_std_normalization=True,
    )

    train_generator=train_datagen.flow_from_dataframe(
        dataframe= train,
        directory="/sise/home/efrco/modelim-compute/shap/ann/new",
        x_col="imagePath",
        y_col="label",
        shuffle=True,
        class_mode="categorical",
        target_size=(224,224))

    test_generator=test_datagen.flow_from_dataframe(
        dataframe= test ,
        directory="/sise/home/efrco/modelim-compute/shap/ann/new",
        x_col="imagePath",
        y_col="label",
        shuffle=True,
        class_mode="categorical",
        target_size=(224,224))

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe= valid ,
        directory="/sise/home/efrco/modelim-compute/shap/ann/new",
        x_col="imagePath",
        y_col="label",
        shuffle=True,
        class_mode="categorical",
        target_size=(224,224))


    optim_2 = Adam(lr=0.0001)
    n_classes = 102
    input_shape = (224, 224, 3)
    # Re-compile the model, this time leaving the last 2 layers unfrozen for Fine-Tuning
    model = create_model(input_shape, n_classes, optim_2, fine_tune=0)


    class TestCallback(tf.keras.callbacks.Callback):
        def __init__(self, model, test_generator):
            self.model = model
            self.test_generator = test_generator
            self.loss = []
            self.acc = []

        def on_epoch_end(self, epoch, logs={}):
            loss, acc= model.evaluate(test_generator,verbose=0)
            self.loss.append(loss)
            self.acc.append(acc)
            print(" ||Test: Loss: %.2f, Accuracy: %.2f" % (loss, acc))


    callback_test = [TestCallback(model, test_generator)]
    history = model.fit_generator(train_generator, epochs=30
                                 , validation_data=valid_generator,callbacks=callback_test)
    return callback_test,history


def fig(callback1, history_pre_train1, callback2, history_pre_train2):
    loss_train1 = history_pre_train1.history["loss"]
    loss_train2 = history_pre_train2.history["loss"]
    loss_train = [(x + y) / 2 for x, y in zip(loss_train1, loss_train2)]

    loss_val1 = history_pre_train1.history["val_loss"]
    loss_val2 = history_pre_train2.history["val_loss"]
    print(loss_val1, loss_val2)
    loss_validation = [(x + y) / 2 for x, y in zip(loss_val1, loss_val2)]

    test_loss1 = callback1[0].loss[:30]
    test_loss2 = callback2[0].loss[:30]
    test_loss = [(x + y) / 2 for x, y in zip(test_loss1, test_loss2)]

    test_acc1 = [i * 100 for i in callback1[0].acc[:30]]
    test_acc2 = [i * 100 for i in callback2[0].acc[:30]]
    test_accuracy = [(x + y) / 2 for x, y in zip(test_acc1, test_acc2)]

    acc_train1 = [i * 100 for i in history_pre_train1.history["accuracy"]]
    acc_train2 = [i * 100 for i in history_pre_train2.history["accuracy"]]
    train_accuracy = [(x + y) / 2 for x, y in zip(acc_train1, acc_train2)]

    acc_val1 = [i * 100 for i in history_pre_train1.history["val_accuracy"]]
    acc_val2 = [i * 100 for i in history_pre_train2.history["val_accuracy"]]
    validation_accuracy = [(x + y) / 2 for x, y in zip(acc_val1, acc_val2)]

    sns.set()

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(loss_train)), loss_train, label="Training Loss")
    plt.plot(range(len(loss_validation)), loss_validation, label="Validation Loss")
    plt.plot(range(len(test_loss)), test_loss, label="Test Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Cross-Entropy Loss for all Epochs for ResNet model ')
    plt.savefig('/sise/home/efrco/modelim-compute/shap/model_output/loss_train_val_test_ResNet.png')

    plt.show()
    sns.set()
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_accuracy)), train_accuracy, label="Training Accuracy")
    plt.plot(range(len(validation_accuracy)), validation_accuracy, label="Validation Accuracy")
    plt.plot(range(len(test_accuracy)), test_accuracy, label="Test Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.yticks(range(0, 100, 10))
    plt.legend(loc='lower right')
    plt.title('Accuracy for all Epochs for ResNet model')
    ax = plt.gca()

    ax.set_ylim(bottom=0)
    plt.savefig('/sise/home/efrco/modelim-compute/shap/model_output/accuracy_train_val_test_ResNet.png')
    plt.show()


# generate two model from another random seed
callback_run1, history_pre_train_run1 = run(random_seed=40)
callback_run2, history_pre_train_run2 = run(random_seed=30)
# plot the results
fig(callback_run1, history_pre_train_run1, callback_run2, history_pre_train_run2)


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



def create_model():
    """
    Compiles a model integrated with VGG16 pretrained layers

    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """

    n_classes = 102
    IMG_SHAPE = (224, 224, 3)
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = tf.keras.applications.vgg16.VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=IMG_SHAPE)

    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.

    for layer in conv_base.layers:
        layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = tf.keras.layers.Flatten(name="flatten")(top_model)
    output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')(top_model)

    # Group the convolutional base and new fully-connected layers into a Model object.
    model = tf.keras.Model(inputs=conv_base.input, outputs=output_layer)

    # Compiles the model for training.
    model.compile(optimizer = tf.keras.optimizers.Adam(0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model




def run(random_seed):

    # load the data
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

    # Image Augmentation
    train_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.3
    )
    valid_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.3
    )

    test_datagen = ImageDataGenerator(
        # rescale = 1./255,
        featurewise_center=True,
        featurewise_std_normalization=True,
    )

    # load the data
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

    # Re-compile the model, this time leaving the last 2 layers unfrozen for Fine-Tuning
    model = create_model()

    # This class save the results of the test class
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
    history = model.fit(train_generator, epochs=30
                                 , validation_data=valid_generator,callbacks=callback_test)
    return callback_test, history



def fig(callback1, history_pre_train1, callback2, history_pre_train2):

    loss_train1=history_pre_train1.history["loss"]
    loss_train2=history_pre_train2.history["loss"]
    loss_train = [(x+y)/2 for x,y in zip(loss_train1, loss_train2)]

    loss_val1 = history_pre_train1.history["val_loss"]
    loss_val2 = history_pre_train2.history["val_loss"]
    print(loss_val1, loss_val2)
    loss_validation = [(x+y)/2 for x,y in zip(loss_val1, loss_val2)]

    test_loss1 = callback1[0].loss[:30]
    test_loss2 = callback2[0].loss[:30]
    test_loss = [(x+y)/2 for x,y in zip(test_loss1, test_loss2)]

    test_acc1 = [i*100 for i in callback1[0].acc[:30]]
    test_acc2 = [i*100 for i in callback2[0].acc[:30]]
    test_accuracy = [(x+y)/2 for x,y in zip(test_acc1, test_acc2)]

    acc_train1 = [i*100 for i in history_pre_train1.history["accuracy"]]
    acc_train2 = [i*100 for i in history_pre_train2.history["accuracy"]]
    train_accuracy = [(x+y)/2 for x,y in zip(acc_train1, acc_train2)]

    acc_val1 = [i*100 for i in history_pre_train1.history["val_accuracy"]]
    acc_val2= [i*100 for i in history_pre_train2.history["val_accuracy"]]
    validation_accuracy = [(x+y)/2 for x,y in zip(acc_val1, acc_val2)]

    sns.set()

    plt.figure(figsize=(10,5))
    plt.plot(range(len(loss_train)), loss_train, label="Training Loss")
    plt.plot(range(len(loss_validation)), loss_validation, label="Validation Loss")
    plt.plot(range(len(test_loss)), test_loss, label="Test Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Cross-Entropy Loss for all Epochs for VGG model ')
    plt.savefig('/sise/home/efrco/modelim-compute/shap/model_output/loss_train_val_test_VGG.png')

    plt.show()
    sns.set()
    plt.figure(figsize=(10,5))
    plt.plot(range(len(train_accuracy)), train_accuracy, label="Training Accuracy")
    plt.plot(range(len(validation_accuracy)), validation_accuracy, label="Validation Accuracy")
    plt.plot(range(len(test_accuracy)), test_accuracy, label="Test Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.yticks(range(0,100,10))
    plt.legend(loc='lower right')
    plt.title('Accuracy for all Epochs for VGG model')
    ax = plt.gca()

    ax.set_ylim(bottom=0)
    plt.savefig('/sise/home/efrco/modelim-compute/shap/model_output/accuracy_train_val_test_VGG.png')
    plt.show()


# generate two model from another random seed
callback_run1, history_pre_train_run1 = run(random_seed=40)
callback_run2, history_pre_train_run2= run(random_seed=30)
# plot the results
fig(callback_run1, history_pre_train_run1, callback_run2, history_pre_train_run2)



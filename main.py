# https://youtu.be/XyX5HNuv-xE
"""
Author: Dr. Sreenivas Bhattiprolu

Multiclass semantic segmentation using U-Net

Including segmenting large images by dividing them into smaller patches
and stiching them back

To annotate images and generate labels, you can use APEER (for free):
www.apeer.com
"""

# from simple_multi_unet_model import multi_unet_model  # Uses softmax

import os
import interpolation_images
os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm
import tensorflow as tf
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pydicom as pd
import random
from data_augmentation import lung_data_generator


def nan2zero(result):

    if np.isnan(result):
        result = 0

    return result


def generate_sample_weights(training_data, class_weights):
    # replaces values for up to 3 classes with the values from class_weights#
    sample_weights = [np.where(y == 0, class_weights[0],
                               np.where(y == 1, class_weights[1],
                                        np.where(y == 2, class_weights[2], y))) for y in training_data]
                                                 #np.where(y == 3, class_weights[3], y)))) for y in training_data]
    return np.asarray(sample_weights)


def data_split(path_original, path_mask, size):
    SIZE_X, SIZE_Y = size

    train_images = []

    for img_path in path_original:
        ds = pd.dcmread(img_path)
        img = ds.pixel_array

        rescale_intercept = int(ds.RescaleIntercept)
        img = img + rescale_intercept

        low = -1000
        high = 250
        img = np.where(img > low, img, low)
        img = np.where(img < high, img, high)

        img = img - low
        img = img.astype(np.uint16)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.resize(img_rgb, [SIZE_X, SIZE_Y])
        train_images.append(img_rgb)

    # Capture training image info as a list

    # Convert list to array for machine learning processing
    train_images = np.array(train_images)

    # Capture mask/label info as a list
    train_mask = []
    count_images = 0
    fibrosis_mask_index = []

    for mask_path in path_mask:
        ds_mask = pd.dcmread(mask_path)

        mask = ds_mask.pixel_array
        value_max = mask.max()

        if value_max == 2:
            fibrosis_mask_index.append(count_images)

        count_images += 1
        mask = cv2.resize(mask, [SIZE_X, SIZE_Y])
        mask = mask.astype(np.int8)

        train_mask.append(mask)

    print(fibrosis_mask_index)
    # Convert list to array for machine learning processing
    train_masks = np.array(train_mask)

    print("El numero de imágenes con etiqueta otros es: ", count_images)

    train_images = train_images[fibrosis_mask_index, :, :, :]
    train_masks = train_masks[fibrosis_mask_index, :, :]

    ###############################################
    # Encode labels... but multi dim array so need to flatten, encode and reshape
    from sklearn.preprocessing import LabelEncoder

    labelencoder = LabelEncoder()
    n, h, w = train_masks.shape
    train_masks_reshaped = train_masks.reshape(-1, 1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

    #################################################
    # train_images = np.expand_dims(train_images, axis=3)
    # train_images = normalize(train_images, axis=1)

    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

    return train_images, train_masks_input, train_masks_reshaped_encoded


def training_U_net(original_train_path, mask_train_path, n_classes, size, seed):

    X_train, y_train, train_mask_reshaped_encoded = data_split(original_train_path, mask_train_path, size)
    #X_test, y_test, _ = data_split(original_test_path, mask_test_path)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.30, random_state=seed)

    X_train, y_train = lung_data_generator(X_train, y_train, seed)
    #X_test, y_test = lung_data_generator(X_test, y_test, seed)

    print("El número de imágenes de entrenamiento es: ", len(X_train))
    print("El número de imágenes de prueba es: ", len(X_test))

    print("Class values in y_train are ... ", np.unique(y_train))
    print("Class values in y_test are ... ", np.unique(y_test))

    """a, b = [77, 87]
    new_image_range = [a+11, b+11]
    X_test = X_test[range(a, b), :, :, :]
    y_test = y_test[range(a, b), :, :, :]
    
    print('Siguiente set: ', new_image_range)"""
    # Further split training data t a smaller subset for quick testing of models
    #_, X_test, _, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=0)


    from tensorflow.keras.utils import to_categorical

    train_masks_cat = to_categorical(y_train, num_classes=n_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

    test_masks_cat = to_categorical(y_test, num_classes=n_classes)
    y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

    ###############################################################

    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]

    ######################################################
    # Reused parameters in all models

    activation = 'softmax'

    LR = 0.0001
    optim = tf.keras.optimizers.Adam(LR)


    from sklearn.utils import class_weight

    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(train_mask_reshaped_encoded),
                                                      y=train_mask_reshaped_encoded)

    print("Class weights are...:", class_weights)

    class_weights = {i: w for i, w in enumerate(class_weights)}
    #class_weights = {0: 0.37583954, 1: 3.10619321, 2: 57.6282969}

    sample_weights = generate_sample_weights(y_train, class_weights)

    ##0.38095213  3.16857783 16.8352328

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    dice_loss = sm.losses.DiceLoss(class_weights=[0.38095213, 3.16857783, 57.6282969])
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + 2*focal_loss

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    ########################################################################
    ###Model 1

    #BACKBONE = 'resnet34'
    #BACKBONE = 'inceptionv3'
    BACKBONE = 'vgg16'

    preprocess_input = sm.get_preprocessing(BACKBONE)

    # preprocess input
    X_train1 = preprocess_input(X_train)
    X_test1 = preprocess_input(X_test)


    #################### Parar ###########################
    # define model
    model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation=activation)

    # compile keras model with defined optimizer, loss and metrics
    model.compile(optim, total_loss, metrics=metrics)

    # model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

    print(model.summary())

    callbacks = [
        #tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint('best_result.hdf5', verbose=1, save_best_only=True)
    ]

    history = model.fit(X_train1,
                        y_train_cat,
                        batch_size=1,
                        epochs=100,
                        verbose=1,
                        validation_data=(X_test1, y_test_cat),
                        sample_weight=sample_weights,
                        callbacks=callbacks)

    model.save('inceptionv3_backbone_50epochs.hdf5')
    ############################################################

    # model.save('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')
    ############################################################
    # Evaluate the model
    # evaluate model

    ###
    # plot the training and validation accuracy and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['iou_score']
    val_acc = history.history['val_iou_score']

    plt.plot(epochs, acc, 'y', label='Training IoU')
    plt.plot(epochs, val_acc, 'r', label='Validation IoU')
    plt.title('Training and validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.show()#"""

    return X_test1, y_test


def test_U_net(X_test1, y_test, n_classes):

    from keras.models import load_model

    name_model = "best_result.hdf5"
    model = load_model(name_model, compile=False)
    #name_model = 'data_augmentation/tested_in_estudios_fibrosis/pulmon0.8838_fibrosis0.4876_weights.hdf5'
    # model.load_weights('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')

    # IOU
    y_pred = model.predict(X_test1)
    y_pred_argmax = np.argmax(y_pred, axis=3)

    ##################################################

    # Using built in keras function
    from keras.metrics import MeanIoU

    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(y_test[:, :, :, 0], y_pred_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())

    # To calculate I0U for each class...
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)
    class1_IoU = values[0, 0] / (
            values[0, 0] + values[0, 1] + values[0, 2] + values[1, 0] + values[2, 0])
    class2_IoU = values[1, 1] / (
            values[1, 1] + values[1, 0] + values[1, 2] + values[0, 1] + values[2, 1])
    class3_IoU = values[2, 2] / (
            values[2, 2] + values[2, 0] + values[2, 1] + values[0, 2] + values[1, 2])

    print("IoU for class1 is: ", class1_IoU)
    print("IoU for class2 is: ", class2_IoU)
    print("IoU for class3 is: ", class3_IoU)


def divide_input_data(path_original, mask_test_path, seed):

    folders = {}

    for directory_path in glob.glob(path_original):
        for img_path in glob.glob(os.path.join(directory_path, "*dcm")):

            file_name = img_path.strip(path_original)
            file_splited = file_name.split('_')
            folder_name = file_splited[0]
            file_number = file_splited[1]

            if folder_name not in folders.keys():
                folders[folder_name] = []

            folders[folder_name].append(file_number)

    key_folders = folders.keys()
    key_folders = list(key_folders)

    no_folders = len(key_folders)
    val_samples = round(no_folders * 0.2)
    random.Random(seed).shuffle(key_folders)
    train_paths = key_folders[:-val_samples] # se modifico, esto estaba originalmente [:-val_samples]
    val_paths = key_folders[-val_samples:]

    complete_train_paths_original = []
    complete_train_paths_mask = []

    complete_val_paths_original = {}
    complete_val_paths_mask = {}

    for each_train in train_paths:
        file_arr = folders[each_train]

        for each_file in file_arr:
            path_complete_original = path_original + each_train + '_' + each_file + 'dcm'
            path_complete_mask = mask_test_path + each_train + '_' + each_file + 'dcm'
            complete_train_paths_original.append(path_complete_original)
            complete_train_paths_mask.append(path_complete_mask)

    for each_val in val_paths:
        file_arr = folders[each_val]
        complete_val_paths_original[each_val] = []
        complete_val_paths_mask[each_val] = []

        for each_file in file_arr:
            path_complete_original = path_original + each_val + '_' + each_file + 'dcm'
            path_complete_mask = mask_test_path + each_val + '_' + each_file + 'dcm'
            complete_val_paths_original[each_val].append(path_complete_original)
            complete_val_paths_mask[each_val].append(path_complete_mask)

    return complete_train_paths_original, complete_train_paths_mask, complete_val_paths_original, complete_val_paths_mask


if __name__ == '__main__':
    # Resizing images, if needed
    SIZE_X = 256
    SIZE_Y = 256
    size = [SIZE_X, SIZE_Y]
    n_classes = 3  # Number of classes for segmentation
    seed = random.randint(0, 2000)
    seed = 631
    print('La semilla para esta corrida es: {}'.format(seed))

    original_test_path = "D:/Bases_de_datos/ILD_DB/ILD_DB_modificada/ILD_datasets/ILD_fibrosis_complement/original"
    mask_test_path = "D:/Bases_de_datos/ILD_DB/ILD_DB_modificada/ILD_datasets/ILD_fibrosis_complement/mask"

    train_paths_original, train_paths_mask, val_paths_original, val_paths_mask = divide_input_data(original_test_path, mask_test_path, seed)

    #x_test, y_test = training_U_net(train_paths_original, train_paths_mask, n_classes, size, seed)
    #test_U_net(x_test, y_test, n_classes)

    interpolation_images.interpolate_dicom(val_paths_original, val_paths_mask, size)


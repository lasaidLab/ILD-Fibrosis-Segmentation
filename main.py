
"""
    Multiclass Semantic Segmentation of Fibrosis in ILD (MSSF-ILD)

    This system uses data from the ILD database to train a U-net for fibrosis segmentation

    Functions
    ------------
    main:
        Access point to the MSSF-ILD System.

"""

__version__ = 1.0
__maintainer__ = "Natanael Hernández Vázquez"
__status__ = "Finished"
__date__ = "10-06-2024"

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
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.metrics import MeanIoU


def nan2zero(result):

    if np.isnan(result):
        result = 0

    return result


def data_preprocessing(path_original: list, path_mask: list, size: list) -> Any:

    """
    Adjusts the image files for training and test the implemented U-net.
    First, it reads and loads each CT image and mask, applies windowing to the CT images, and then normalizes
    the intensities.

    Parameters
    ----------
    path_original:
        Path of the CT image
    path_mask:
        Path of the mask
    size:
        New image size for resizing 
        
    Returns
    ----------
        train_images:
            Multidimensional array with all the preprocessed images
        train_masks_input
            Multidimensional array with all the corresponding preprocessed masks 
        train_masks_reshaped_encoded
            Multidimensional list with all the corresponding preprocessed masks

    """

    SIZE_X, SIZE_Y = size

    train_images = []

    # Image windowing by HU (Hounsfield Units): -1000 HU to 200 HU

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

    # Convert list to array for machine learning processing
    train_masks = np.array(train_mask)

    train_images = train_images[fibrosis_mask_index, :, :, :]
    train_masks = train_masks[fibrosis_mask_index, :, :]

    labelencoder = LabelEncoder()
    n, h, w = train_masks.shape
    train_masks_reshaped = train_masks.reshape(-1, 1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

    return train_images, train_masks_input, train_masks_reshaped_encoded


def training_U_net(original_train_path: list, mask_train_path: list, n_classes: int, size: list, seed: int) -> Any:

    """
    U-net construction, training and test for semantic segmentation of background, lung tissue and fibrosis.

    Parameters
    ------------
    original_train_path:
        Original training images paths
    mask_train_path:
        Original training mask paths
    n_classes:
        Number of classes used for the semantic segmentation
    size:
        Size of the images used for training and test
    seed:
        Defines the main seed used to generate the random numbers in each division.

    Returns
    ------------



    """

    X_train, y_train, train_mask_reshaped_encoded = data_preprocessing(original_train_path, mask_train_path, size)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.30, random_state=seed)
    X_train, y_train = lung_data_generator(X_train, y_train, seed)

    train_masks_cat = to_categorical(y_train, num_classes=n_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

    test_masks_cat = to_categorical(y_test, num_classes=n_classes)
    y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

    activation = 'softmax'

    lr = 0.0001
    optim = tf.keras.optimizers.Adam(lr)

    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(train_mask_reshaped_encoded),
                                                      y=train_mask_reshaped_encoded)

    print("Class weights are...:", class_weights)

    class_weights = {i: w for i, w in enumerate(class_weights)}

    # replaces values for up to 3 classes with the values from class_weights#
    sample_weights = [np.where(y == 0, class_weights[0],
                               np.where(y == 1, class_weights[1],
                                        np.where(y == 2, class_weights[2], y))) for y in y_train]

    sample_weights = np.asarray(sample_weights)

    dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + 2*focal_loss

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    backbone = 'vgg16'

    preprocess_input = sm.get_preprocessing(backbone)

    X_train = preprocess_input(X_train)
    X_test = preprocess_input(X_test)

    model = sm.Unet(backbone, encoder_weights='imagenet', classes=n_classes, activation=activation)

    model.compile(optim, total_loss, metrics=metrics)

    print(model.summary())

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('best_result.hdf5', verbose=1, save_best_only=True)
    ]

    history = model.fit(X_train,
                        y_train_cat,
                        batch_size=1,
                        epochs=100,
                        verbose=1,
                        validation_data=(X_test, y_test_cat),
                        sample_weight=sample_weights,
                        callbacks=callbacks)

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
    plt.show()

    return X_test, y_test


def test_U_net(X_test1, y_test, n_classes):

    from keras.models import load_model

    name_model = "best_result.hdf5"
    model = load_model(name_model, compile=False)

    # IOU
    y_pred = model.predict(X_test1)
    y_pred_argmax = np.argmax(y_pred, axis=3)


    # Using built in keras function


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


def divide_input_data(path_original: str, mask_test_path: str, seed: int) -> Any:
    """
    Takes the input paths, reads all the routes of the relevant files, and divides them into training and validation.
    The division takes 80% of the files for training and 20% for validation.

    Parameters
    ------------
    path_original:
        Defines the path where the raw thorax CT DICOM files are saved in studies.
    mask_test_path:
        Defines the path where the DICOM masks are saved. These masks contain the gold standard for training.
        The annotation intensities are: 0 for background, 1 for lung tissue, and 2 for fibrosis.
    seed:
        Defines the main seed used to generate the random numbers in each division.

    Return
    ------------
    list:
        complete_train_paths_original:
            Original training images paths
        complete_train_paths_mask:
            Original training mask paths
        complete_val_paths_original:
            Original validation images paths
        complete_val_paths_mask:
            Original training mask paths

    """

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
    train_paths = key_folders[:-val_samples]
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
    """
    It controls all the system flow and defines the initial parameters for the experiments.

    Main parameters:
    ------------
    size:
        Defines the image size for training, testing, and validation.

    original_test_path:
        Defines the path where the raw thorax CT DICOM files are saved in studies.

    mask_test_path:
        Defines the path where the DICOM masks are saved. These masks contain the gold standard for training.
        The annotation intensities are: 0 for background, 1 for lung tissue, and 2 for fibrosis.

    n_classes:
        Defines the classes used for semantic segmentation. In this case, it is always 3.
    """

    # Resizing images, if needed
    SIZE_X = 256
    SIZE_Y = 256

    original_test_path = "/original"
    mask_test_path = "/mask"

    size = [SIZE_X, SIZE_Y]
    n_classes = 3  # Number of classes for segmentation
    seed = random.randint(0, 2000)

    train_paths_original, train_paths_mask, val_paths_original, val_paths_mask = divide_input_data(original_test_path,
                                                                                                   mask_test_path, seed)

    x_test, y_test = training_U_net(train_paths_original, train_paths_mask, n_classes, size, seed)
    test_U_net(x_test, y_test, n_classes)

    interpolation_images.interpolate_dicom(val_paths_original, val_paths_mask, size)


from scipy.interpolate import RegularGridInterpolator
from skimage import io, img_as_ubyte
from matplotlib import pyplot as plt
import numpy as np
from tifffile import imsave, imwrite
import pydicom as pd
import cv2
from utilities import arr_modified
import re
from main import sm
from ZeroPadding import zero_padding
from sklearn.metrics import jaccard_score
from keras.metrics import MeanIoU
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder



def IoU_comparation(n_classes, y_test, y_pred_argmax, comparation_type):

    """
    Calculates the Intersection over Union (IoU) for each class in a multi-class segmentation task. It uses the gold
    standard and predicted masks to evaluate the model's performance and prints the IoU for each class. The function can
    handle both total and normalized comparisons.

    Parameters
    ------------
    n_classes:
        The number of classes in the segmentation task.
    y_test
        The gold standard masks for the test dataset.
    y_pred_argmax
        The predicted masks obtained by taking the argmax of the model's output.
    comparation_type:
        A string indicating the type of comparison being performed.

    Returns
    ------------
    iou_total:
        A list containing the IoU values for each class.

    """

    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(y_test[:, :, :, 0], y_pred_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())

    # To calculate I0U for each class...
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    class1_IoU = values[0, 0] / (
            values[0, 0] + values[0, 1] + values[0, 2] + values[1, 0] + values[2, 0])
    class2_IoU = values[1, 1] / (
            values[1, 1] + values[1, 0] + values[1, 2] + values[0, 1] + values[2, 1])
    class3_IoU = values[2, 2] / (
            values[2, 2] + values[2, 0] + values[2, 1] + values[0, 2] + values[1, 2])

    print("Comparation type: {}".format(comparation_type))
    print("IoU for class1 is: ", class1_IoU)
    print("IoU for class2 is: ", class2_IoU)
    print("IoU for class3 is: ", class3_IoU)

    jaccard_study = []

    for each_matrix in range(0, len(y_test)):
        gold_matrix = y_test[each_matrix]
        test_matrix = y_pred_argmax[each_matrix]

        IOU_FFR = MeanIoU(num_classes=n_classes)
        IOU_FFR.update_state(gold_matrix[:, :, 0], test_matrix)

        values_FFR = np.array(IOU_FFR.get_weights()).reshape(n_classes, n_classes)
        class1_IoU = values_FFR[0, 0] / (
                values_FFR[0, 0] + values_FFR[0, 1] + values_FFR[0, 2] + values_FFR[1, 0] + values_FFR[2, 0])
        class2_IoU = values_FFR[1, 1] / (
                values_FFR[1, 1] + values_FFR[1, 0] + values_FFR[1, 2] + values_FFR[0, 1] + values_FFR[2, 1])
        class3_IoU = values_FFR[2, 2] / (
                values_FFR[2, 2] + values_FFR[2, 0] + values_FFR[2, 1] + values_FFR[0, 2] + values_FFR[1, 2])

        print("Comparation type: {}".format(comparation_type))
        print("IoU for class1 is: ", class1_IoU)
        print("IoU for class2 is: ", class2_IoU)
        print("IoU for class3 is: ", class3_IoU)

    print(jaccard_study)

    iou_total = [class1_IoU, class2_IoU, class3_IoU]

    return iou_total


def test_U_net_estimation(X_test, y_test, n_classes):

    """
    Evaluates the performance of a pre-trained semantic segmentation model (based on a U-Net architecture with a VGG16
    backbone) on a set of test images. The evaluation includes predictions, calculation of Intersection over Union (IoU)
    metrics.

    Parameters
    ------------
    X_test:
        Multidimensional array with all CT images for test
    y_test:
        Multidimensional array with all masks for test
    n_classes:
        Number of classes used for the semantic segmentation

    Return
    ------------
        y_pred_argmax:
            Predicted mask, this mask contains the predicted class labels for each pixel.
        iou_total:
            The IoU (Intersection over Union) evaluation for the total comparison between the predicted masks and the
            ground truth masks. It measures the overlap between the predicted and actual masks.
        iou_normalized:
            The IoU evaluation for the normalized comparison between the predicted masks and the ground truth masks.
            This metric normalizes the predictions before calculating IoU by only using its ROIs.
        y_pred_norm:
            he normalized predicted mask, obtained by multiplying the predicted mask (y_pred_argmax) by the adjusted
            ground truth mask (y_test_new). This normalization helps in comparing only the relevant regions of the
            prediction with the ground truth.
    """

    BACKBONE = 'vgg16'

    preprocess_input = sm.get_preprocessing(BACKBONE)

    name_model = "best_result.hdf5"
    model = load_model(name_model, compile=False)

    # preprocess input
    X_test = preprocess_input(X_test)

    # IOU
    y_pred = model.predict(X_test)
    y_pred_argmax = np.argmax(y_pred, axis=3)

    y_test_new = np.maximum(y_test - 1, 0)
    y_pred_norm = y_pred_argmax * y_test_new[:, :, :, 0]

    iou_total = IoU_comparation(n_classes, y_test, y_pred_argmax, 'total comparation')
    iou_normalized = IoU_comparation(n_classes, y_test, y_pred_norm, 'normalized comparation')

    return y_pred_argmax, iou_total, iou_normalized, y_pred_norm


def validate_performance(val_paths_original, val_paths_mask, size, n_classes):

    """
    Using the file paths for validation, CT images, and masks with gold standard annotations, it loads them into memory
    and adapts them to the pre-trained segmentation models. The CT images are processed through the trained model
    obtained from "best_result.hdf5", and the resulting masks are compared with the gold standard. The performance
    results are displayed on the screen, and the obtained images are saved as .tif files.

    Parameters
    ------------
    val_paths_original:
        Defines the path where the raw thorax CT DICOM files are saved in studies.
    val_paths_mask:
        Defines the path where the DICOM masks are saved. These masks contain the gold standard.
        The annotation intensities are: 0 for background, 1 for lung tissue, and 2 for fibrosis.
    size:
        Defines the image size for validation.
    n_classes:
        Number of classes used for the semantic segmentation

    Return
    ------------
        Performance results displayed on screen (global IOU performance and global IOU delimited performance)
        Save all used images (CT images, gold-standard mask, and obtained mask) in .tif files
    """

    SIZE_X, SIZE_Y = size

    keys_folders = val_paths_original.keys()

    iou_matrix_total = []
    iou_matrix_normalized = []

    for each_folder in keys_folders:

        print('The folder for these studies is: {}'.format(each_folder))
        arr_original = []
        images_path = val_paths_original[each_folder]
        r = re.compile(r"(_\d+\.)")
        images_path.sort(key=lambda x: int(r.search(x).group(1).strip('_.')))


        for each_image in images_path:
            ds = pd.dcmread(each_image)
            img = ds.pixel_array
            img_size = img.shape

            rescale_intercept = int(ds.RescaleIntercept)

            low = -1000
            high = 250

            img = img + rescale_intercept

            img = np.where(img > low, img, low)
            img = np.where(img < high, img, high)

            img = img - low
            img = img.astype(np.uint16)

            # img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, [SIZE_X, SIZE_Y])
            arr_original.append(img)

        pixel_relation = img_size[0] / SIZE_X
        pixelsize = ds.PixelSpacing[0] * pixel_relation
        lamda = 2
        slice_thickness_old = 10
        slice_thickness_new = slice_thickness_old / lamda
        arr_original = np.array(arr_original)

        arr_interpolated = np.array(arr_interpolated)

        arr_original_new = []
        for each_matrix in arr_original:
            each_matrix = cv2.cvtColor(each_matrix, cv2.COLOR_GRAY2BGR)
            arr_original_new.append(each_matrix)

        arr_original_new = np.array(arr_original_new)

        arr_mask = []
        images_path = val_paths_mask[each_folder]
        images_path.sort(key=lambda x: int(r.search(x).group(1).strip('_.')))

        for mask_path in images_path:
            ds_mask = pd.dcmread(mask_path)

            mask = ds_mask.pixel_array
            mask = cv2.resize(mask, [SIZE_X, SIZE_Y])
            mask = mask.astype(np.int8)

            arr_mask.append(mask)

        arr_mask = np.array(arr_mask)

        labelencoder = LabelEncoder()
        n, h, w = arr_mask.shape
        train_masks_reshaped = arr_mask.reshape(-1, 1)
        train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
        train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

        train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

        mask_pred, iou_total, iou_normalized, mask_pred_delim = test_U_net_estimation(arr_original_new,
                                                                                      train_masks_input, n_classes)

        iou_matrix_total.append(iou_total)
        iou_matrix_normalized.append(iou_normalized[2])

        ct_original_image = arr_original.astype(np.uint16)
        file = 'ct_original/study_new' + each_folder.strip("\\\\") + '.tif'
        metadata = {'spacing': slice_thickness_old, 'unit': 'mm', 'axes': 'ZYX'}
        imwrite(file, ct_original_image, imagej=True, resolution=(1/pixelsize, 1/pixelsize), metadata=metadata)

        gold_standard = arr_mask.astype(np.uint16)
        file = 'gold_standard/study_' + each_folder.strip("\\\\") + '.tif'
        metadata = {'spacing': slice_thickness_old, 'unit': 'mm', 'axes': 'ZYX'}
        imwrite(file, gold_standard, imagej=True, resolution=(1/pixelsize, 1/pixelsize), metadata=metadata)

        mask_pred = mask_pred.astype(np.uint16)
        file = 'mascaras_pred/study_' + each_folder.strip("\\\\") + '.tif'
        metadata = {'spacing': slice_thickness_new, 'unit': 'mm', 'axes': 'ZYX'}
        imwrite(file, mask_pred, imagej=True, resolution=(1 / pixelsize, 1 / pixelsize), metadata=metadata)

        mask_pred_delim = mask_pred_delim.astype(np.uint16)
        file = 'mascaras_pred_delim/study_' + each_folder.strip("\\\\") + '.tif'
        metadata = {'spacing': slice_thickness_new, 'unit': 'mm', 'axes': 'ZYX'}
        imwrite(file, mask_pred_delim, imagej=True, resolution=(1 / pixelsize, 1 / pixelsize), metadata=metadata)

    print('Global results')
    print('IOU total results = {}'.format(iou_matrix_total))
    print('IOU ROI delimited results = {}'.format(iou_matrix_normalized))

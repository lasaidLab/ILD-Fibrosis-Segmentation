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


def IoU_comparation(n_classes, y_test, y_pred_argmax, comparation_type):
    # Using built in keras function
    from keras.metrics import MeanIoU

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

        """plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title('Testing Label')
        plt.imshow(gold_matrix[:, :, 0], cmap='jet', vmin=0, vmax=3)
        plt.subplot(233)
        plt.title('Prediction on test image')
        plt.imshow(test_matrix, cmap='jet', vmin=0, vmax=3)
        plt.show()"""

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
    BACKBONE = 'vgg16'

    preprocess_input = sm.get_preprocessing(BACKBONE)

    from keras.models import load_model

    name_model = "best_result.hdf5"
    model = load_model(name_model, compile=False)

    # preprocess input
    X_test1 = preprocess_input(X_test)

    # IOU
    y_pred = model.predict(X_test1)
    y_pred_argmax = np.argmax(y_pred, axis=3)

    y_test_len = len(y_test)

    if y_test_len == 0:

        lung_pixels_gold = 0
        fibrosis_pixels_gold = 0
        lung_pixels_pred = []
        fibrosis_pixels_pred = []

        for test_img_number in range(0, len(X_test)):
            test_img = X_test[test_img_number]
            # ground_truth = y_pred_norm[test_img_number]

            test_img_input = np.expand_dims(test_img, 0)

            test_img_input = preprocess_input(test_img_input)

            test_pred = model.predict(test_img_input)
            test_prediction = np.argmax(test_pred, axis=3)[0, :, :]

            # lung_pixels_gold.append(np.sum((ground_truth == 1)))
            # fibrosis_pixels_gold.append(np.sum((ground_truth == 2)))
            lung_pixels_pred.append(np.sum((test_prediction == 1)))
            fibrosis_pixels_pred.append(np.sum((test_prediction == 2)))

        # lung_pixels_gold = np.array(lung_pixels_gold)
        # fibrosis_pixels_gold = np.array(fibrosis_pixels_gold)
        lung_pixels_pred = np.array(lung_pixels_pred)
        fibrosis_pixels_pred = np.array(fibrosis_pixels_pred)

        iou_total = None
        iou_normalized = None
        y_pred_norm = None


    else:

        y_test_new = np.maximum(y_test - 1, 0)
        y_pred_norm = y_pred_argmax * y_test_new[:, :, :, 0]

        iou_total = IoU_comparation(n_classes, y_test, y_pred_argmax, 'total comparation')
        iou_normalized = IoU_comparation(n_classes, y_test, y_pred_norm, 'normalized comparation')

        lung_pixels_gold = []
        fibrosis_pixels_gold = []
        lung_pixels_pred = []
        fibrosis_pixels_pred = []

        for test_img_number in range(0, len(X_test)):
            test_img = X_test[test_img_number]
            ground_truth = y_test[test_img_number]

            test_img_input = np.expand_dims(test_img, 0)

            test_img_input = preprocess_input(test_img_input)

            test_pred = model.predict(test_img_input)
            test_prediction = np.argmax(test_pred, axis=3)[0, :, :]

            """plt.figure(figsize=(12, 8))
            plt.subplot(121)
            plt.title('Testing Label')
            plt.imshow(ground_truth, cmap='jet', vmin=0, vmax=3)
            plt.subplot(122)
            plt.title('Prediction on test image')
            plt.imshow(test_prediction, cmap='jet', vmin=0, vmax=3)
            plt.show()"""

            lung_pixels_gold.append(np.sum((ground_truth == 1)))
            fibrosis_pixels_gold.append(np.sum((ground_truth == 2)))
            lung_pixels_pred.append(np.sum((test_prediction == 1)))
            fibrosis_pixels_pred.append(np.sum((test_prediction == 2)))

        lung_pixels_gold = np.array(lung_pixels_gold)
        fibrosis_pixels_gold = np.array(fibrosis_pixels_gold)
        lung_pixels_pred = np.array(lung_pixels_pred)
        fibrosis_pixels_pred = np.array(fibrosis_pixels_pred)

    return lung_pixels_gold, fibrosis_pixels_gold, lung_pixels_pred, fibrosis_pixels_pred, y_pred_argmax, iou_total, iou_normalized, y_pred_norm


def interpolate_dicom(val_paths_original, val_paths_mask, size):
    SIZE_X, SIZE_Y = size

    keys_folders = val_paths_original.keys()

    iou_matrix_total = []
    iou_matrix_normalized = []

    for each_folder in keys_folders:

        print('El folder de estos estudios es: {}'.format(each_folder))
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

            """if each_folder == '\94':
                img = cv2.resize(img, [SIZE_X, SIZE_Y])

                low = 0
                high = 1250

                img = np.where(img > low, img, low)
                img = np.where(img < high, img, high)

                #img = img - low
                arr_original.append(img)"""

            #else:
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


        ## INTERPOLATION STAGE ##
        image_interpolated = zero_padding(arr_original, 'original', lamda)
        # arr_original = interpolation_stage(arr_original, pixelsize, slice_thickness_old, 'interpolated_original.tif')

        arr_interpolated = []
        for each_matrix in image_interpolated:
            each_matrix = cv2.cvtColor(each_matrix, cv2.COLOR_GRAY2BGR)
            arr_interpolated.append(each_matrix)

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
        # arr_mask = interpolation_stage(arr_mask, pixelsize_old, slice_thickness_old, 'interpolated_mask.tif')

        from sklearn.preprocessing import LabelEncoder

        labelencoder = LabelEncoder()
        n, h, w = arr_mask.shape
        train_masks_reshaped = arr_mask.reshape(-1, 1)
        train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
        train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

        #################################################
        # train_images = np.expand_dims(train_images, axis=3)
        # train_images = normalize(train_images, axis=1)

        train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

        # image_interpolated = zero_padding(train_masks_input, 'mascaras')

        lung_pixels_gold, fibrosis_pixels_gold, lung_pixels_pred, fibrosis_pixels_pred, mask_pred, iou_total, iou_normalized, mask_pred_delim = \
            (test_U_net_estimation(arr_original_new, train_masks_input, 3))

        iou_matrix_total.append(iou_total)
        iou_matrix_normalized.append(iou_normalized[2])

        lung_area_gold = lung_pixels_gold * pixelsize
        fibrosis_area_gold = fibrosis_pixels_gold * pixelsize
        lung_area_pred = lung_pixels_pred * pixelsize
        fibrosis_area_pred = fibrosis_pixels_pred * pixelsize

        lung_volume_gold = lung_area_gold * slice_thickness_old
        fibrosis_volume_gold = fibrosis_area_gold * slice_thickness_old
        lung_volume_pred = lung_area_pred * slice_thickness_old
        fibrosis_volume_pred = fibrosis_area_pred * slice_thickness_old

        total_lung_volume_gold = sum(lung_volume_gold)
        total_fibrosis_volume_gold = sum(fibrosis_volume_gold)
        total_lung_volume_pred = sum(lung_volume_pred)
        total_fibrosis_volume_pred = sum(fibrosis_volume_pred)

        print("El volumen total para pulmón en imágenes gold es: {} mm3".format(total_lung_volume_gold))
        print("El volumen total para fibrosis en imágenes gold es: {} mm3".format(total_fibrosis_volume_gold))
        print("El volumen total para pulmón en imágenes obtenidas es: {} mm3".format(total_lung_volume_pred))
        print("El volumen total para fibrosis en imágenes obtenidas es: {} mm3".format(total_fibrosis_volume_pred))

        _, _, lung_pixels_pred_in, fibrosis_pixels_pred_in, mask_interpolated, _, _, _ = (
            test_U_net_estimation(arr_interpolated, [], 3))

        lung_area_pred_in = lung_pixels_pred_in * pixelsize
        fibrosis_area_pred_in = fibrosis_pixels_pred_in * pixelsize

        print('Las areas de pulmón para cada imágen son: {}'.format(lung_area_pred_in))
        print('Las areas de FRF para cada imágen son: {}'.format(fibrosis_area_pred_in))

        lung_volume_pred_in = lung_area_pred_in * slice_thickness_new
        fibrosis_volume_pred_in = fibrosis_area_pred_in * slice_thickness_new

        total_lung_volume_pred_in = sum(lung_volume_pred_in)
        total_fibrosis_volume_pred = sum(fibrosis_volume_pred_in)

        print(
            "El volumen total para pulmón en imágenes interpoladas obtenidas es: {} mm3".format(total_lung_volume_pred_in))
        print("El volumen total para fibrosis en imágenes interpoladas obtenidas es: {} mm3".format(
            total_fibrosis_volume_pred))
        print('Terminado')


        ct_original_image = arr_original.astype(np.uint16)
        file = 'ct_original/study_new' + each_folder.strip("\\\\") + '.tif'
        metadata = {'spacing': slice_thickness_old, 'unit': 'mm', 'axes': 'ZYX'}
        imwrite(file, ct_original_image, imagej=True, resolution=(1/pixelsize, 1/pixelsize), metadata=metadata)


        gold_standard = arr_mask.astype(np.uint16)
        file = 'gold_standard/study_' + each_folder.strip("\\\\") + '.tif'
        metadata = {'spacing': slice_thickness_old, 'unit': 'mm', 'axes': 'ZYX'}
        imwrite(file, gold_standard, imagej=True, resolution=(1/pixelsize, 1/pixelsize), metadata=metadata)

        ct_original_interpolated = image_interpolated.astype(np.uint16)
        file = 'ct_original_interpolated/study_' + each_folder.strip("\\\\") + '.tif'
        metadata = {'spacing': slice_thickness_new, 'unit': 'mm', 'axes': 'ZYX'}
        imwrite(file, ct_original_interpolated, imagej=True, resolution=(1/pixelsize, 1/pixelsize), metadata=metadata)

        mask_pred = mask_pred.astype(np.uint16)
        file = 'mascaras_pred/study_' + each_folder.strip("\\\\") + '.tif'
        metadata = {'spacing': slice_thickness_new, 'unit': 'mm', 'axes': 'ZYX'}
        imwrite(file, mask_pred, imagej=True, resolution=(1 / pixelsize, 1 / pixelsize), metadata=metadata)

        mask_pred_delim = mask_pred_delim.astype(np.uint16)
        file = 'mascaras_pred_delim/study_' + each_folder.strip("\\\\") + '.tif'
        metadata = {'spacing': slice_thickness_new, 'unit': 'mm', 'axes': 'ZYX'}
        imwrite(file, mask_pred_delim, imagej=True, resolution=(1 / pixelsize, 1 / pixelsize), metadata=metadata)

        mask_interpolated = mask_interpolated.astype(np.uint16)
        file = 'mascaras_interpoladas/study_' + each_folder.strip("\\\\") + '.tif'
        metadata = {'spacing': slice_thickness_new, 'unit': 'mm', 'axes': 'ZYX'}
        imwrite(file, mask_interpolated, imagej=True, resolution=(1/pixelsize, 1/pixelsize), metadata=metadata)


    print('Resultados finales')
    print('Matriz total = {}'.format(iou_matrix_total))
    print('Matriz normalizada = {}'.format(iou_matrix_normalized))

import cv2
from matplotlib import pyplot as plt
from skimage import draw
import numpy as np
import os
from tifffile import imread
from sklearn.metrics import jaccard_score


"""
    This script is separated from the main system and is used by manually defining the paths where each file for 
    evaluation is located.
"""


def expert_evaluation(study_path, mask_file_path, main_path, file_comparator, file_gold):

    """
    This script processes and evaluates semantic segmentation results on medical images, comparing obtained masks
    against gold standard annotations. It calculates Jaccard similarity scores and visualizes the original image,
    gold standard mask, annotated mask (expert annotation), and the model predicted mask for each image slice.

    The annotated images are .jpg files, hand annotated by the expert, to obtain the mask for comparison the following 
    was done:
        Read and processes the original image to enhance features related to segmentation.
        Binarizes and processes the image to obtain an annotated mask using morphological operations.

    Parameters
    ------------
    study_path:
        Directory containing study files.
    mask_file_path:
        Subdirectory to save processed mask images.
    main_path:
        Base path where study and comparison files are stored.
    file_comparator:
        TIFF file containing comparison data.
    file_gold:
        TIFF file containing gold standard annotations.

    Return
    ------------
        Displays a 2x2 grid of subplots showing:
            Original image
            Gold standard mask
            Annotated mask by a specialist
            Obtained segmentation mask

        Saves processed mask images (each_image_norm) in JPEG format to a specified subdirectory (new_folder_name).

        Prints on screen IoU performance
    """

    study_path = main_path + study_path
    file_comparator = main_path + file_comparator
    file_gold = main_path + file_gold

    cuts = os.listdir(study_path)
    cuts = [file for file in cuts if file.endswith(".png")]

    mask_set = imread(file_comparator)
    mask_gold_set = imread(file_gold)

    for each_cut in cuts:

        cut_number = each_cut.strip('.png').split('_')
        cut_number = int(cut_number[1]) - 1
        or_cut_n = cut_number + 1

        mask_obtained = mask_set[cut_number]
        mask_gold = mask_gold_set[cut_number]

        ruta_completa = ruta_estudio + each_cut

        image = cv2.imread(ruta_completa)

        rojo, verde, azul = cv2.split(image)

        new_image = rojo - azul
        new_image[new_image > 0] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dst = cv2.morphologyEx(new_image, cv2.MORPH_CLOSE, kernel)
        all_contours, _ = cv2.findContours(dst, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        index_allcontours = int(len(all_contours)/2)

        main_contours = []
        for each_contour in range(0, index_allcontours):

            decrease_contour = all_contours[each_contour*2 - 1]
            main_contours.append(decrease_contour)

        image_shape = dst.shape

        blank_image = np.zeros(image_shape, dtype=np.uint8)

        for contour in main_contours:
            length_contour, deep, high_contour = contour.shape

            contour_new = contour.reshape((length_contour, high_contour))
            contour_new = np.flip(contour_new, axis=1)
            mask = draw.polygon2mask(image_shape, contour_new)

            mask = mask.astype(np.int8)

            test_mask = blank_image * mask
            max_test_mask = test_mask.max()

            if max_test_mask != 0:
                blank_image = abs(blank_image - mask)
            else:
                blank_image = blank_image + mask

        SIZE_X, SIZE_Y = mask_obtained.shape
        blank_image = cv2.resize(blank_image, [SIZE_X, SIZE_Y])

        mask_obtained[mask_obtained == 1] = 0
        mask_obtained[mask_obtained == 2] = 1

        mask_gold[mask_gold == 1] = 0
        mask_gold[mask_gold == 2] = 1

        jaccard_image = jaccard_score(blank_image, mask_obtained, average='micro')

        jaccard_gold = jaccard_score(mask_gold, mask_obtained, average='micro')

        plt.figure()

        plt.subplot(221)
        plt.title(each_cut + ' Imagen original')
        plt.imshow(image)

        plt.subplot(222)
        plt.title(each_cut + ' Gold-standard')
        plt.imshow(mask_gold)

        plt.subplot(223)
        plt.title(each_cut + ' Mascara anotada por Dr.')
        plt.imshow(blank_image)

        plt.subplot(224)
        plt.title(each_cut + ' Mascara obtenida')
        plt.imshow(mask_obtained)
        plt.show()

        print('jaccard del: ' + each_cut + ' ' + str(jaccard_image))
        print('jaccard contra el gold de la base: ' + each_cut + ' ' + str(jaccard_gold))

        new_folder_name = main_path + estudy_path + mask_file_path

        if not os.path.exists(new_folder_name):
            os.makedirs(new_folder_name)

        min_value = blank_image.min()
        max_value = blank_image.max()

        # new_file_name = new_folder_name + '/CT_' + str(index)

        each_image_norm = (blank_image / max_value * 255).astype(np.uint8)

        # each_image_norm = each_image.astype('uint8')
        image_name = os.path.join(new_folder_name, 'CT_' + str(or_cut_n) + '.jpg')
        cv2.imwrite(image_name, each_image_norm)


study_path = 'estudio_184.1/'
mask_file_path = 'mask_image/'
main_path = 'frf_expert_annotation/'
file_comparator = 'study_184.1.tif'
file_gold = 'study_184.1_gold.tif'

expert_evaluation(study_path, mask_file_path, main_path, file_comparator, file_gold)

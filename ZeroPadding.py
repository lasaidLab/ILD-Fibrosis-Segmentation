from matplotlib import pyplot as plt
import numpy as np
from tifffile import imwrite

def zero_padding(arr, file, lamda):
    images_q, rows_q, columns_q = arr.shape

    min_arr = arr.min()
    max_arr = arr.max()

    new_images = np.zeros((rows_q, columns_q, images_q), np.dtype(np.uint16))
    each_image_index = 0

    for each_image in arr:
        matrix_columns, _ = each_image.shape
        for each_column in range(0, matrix_columns):
            column = each_image[:, each_column]
            new_images[each_column, :, each_image_index] = column

        each_image_index += 1

    mask_interpolated = new_images.astype(np.uint16)
    file = 'imagenes_transformadas/study_' + file + '.tif'
    metadata = {'spacing': 10, 'unit': 'mm', 'axes': 'ZYX'}
    imwrite(file, mask_interpolated, imagej=True, resolution=(1, 1), metadata=metadata)

    images_q_new = images_q * lamda
    padding_images = np.zeros((images_q_new, rows_q, columns_q))

    column_index = 0

    for each_image_t in new_images:

        M, N = each_image_t.shape
        Mresize = M
        Nresize = round(lamda * N)

        dimX = round((Mresize - M) / 2)
        dimY = round((Nresize - N) / 2)

        matrix_for_tf2 = np.zeros((Mresize, Nresize), np.dtype(np.complex128))

        tf2_image = np.fft.fft2(each_image_t)
        tf2_image_shift = np.fft.fftshift(tf2_image)
        matrix_for_tf2[dimX: dimX + M, dimY: dimY + N] = tf2_image_shift
        new_tf2_image = np.fft.ifftshift(matrix_for_tf2)
        new_image = np.abs(np.fft.ifft2(new_tf2_image))

        shape_X, shape_Y = new_image.shape

        """if column_index == (85):
            file = 'original.tif'
            plt.subplot(231)
            imwrite(file, each_image_t)
            plt.subplot(232)
            plt.title('fft')
            file = 'fft.tif'
            plt.imshow(np.abs(tf2_image), cmap='jet')
            imwrite(file, np.abs(tf2_image))
            plt.subplot(233)
            plt.title('FFTshift')
            file = 'FFTshift.tif'
            plt.imshow(np.abs(tf2_image_shift), cmap='jet')
            imwrite(file, np.abs(tf2_image_shift))
            plt.subplot(234)
            plt.title('zero_padding')
            file = 'zero_padding.tif'
            plt.imshow(np.abs(matrix_for_tf2), cmap='jet')
            imwrite(file, np.abs(matrix_for_tf2))
            plt.subplot(235)
            plt.title('ifftwhift')
            file = 'ifftwhift.tif'
            plt.imshow(np.abs(new_tf2_image), cmap='jet')
            imwrite(file, np.abs(new_tf2_image))
            plt.subplot(236)
            plt.title('ifft2')
            file = 'ifft2.tif'
            plt.imshow(new_image, cmap='gray')
            imwrite(file, new_image)
            plt.show()"""

        for index_image in range(0, shape_Y):
            padding_images[index_image, :, column_index] = new_image[:, index_image]

        column_index += 1

    normalized_data = (padding_images - np.min(padding_images)) * (max_arr - min_arr) / (np.max(padding_images) - np.min(padding_images)) + min_arr
    normalized_data = normalized_data.astype(np.uint16)

    drop_images = lamda - 1
    normalized_data = normalized_data[:-drop_images]

    return normalized_data

import numpy as np
from matplotlib import pyplot as plt
import cv2

each_image_t = plt.imread('dog.jpg')
each_image_t = cv2.cvtColor(each_image_t, cv2.COLOR_BGR2GRAY)

min = 0
max = 254

normalized_matrix = (each_image_t - min) / (max - min)

M, N = each_image_t.shape
lamda = 2
Mresize = round(lamda * M)
Nresize = round(lamda * N)

dimX = round((Mresize - M) / 2)
dimY = round((Nresize - N) / 2)

matrix_for_tf2 = np.zeros((Mresize, Nresize), np.dtype(np.complex128))

tf2_image = np.fft.fft2(each_image_t)
tf2_image = np.fft.fftshift(tf2_image)

matrix_for_tf2[dimX: dimX + M, dimY: dimY + N] = tf2_image
new_tf2_image = np.fft.ifftshift(matrix_for_tf2)
new_image = np.abs(np.fft.ifft2(new_tf2_image))

plt.title('old image')
plt.imshow(each_image_t, cmap='gray')
plt.show()

plt.title('transformada')
plt.imshow(np.abs(matrix_for_tf2))
plt.show()

plt.title('new image')
plt.imshow(new_image, cmap='gray')
plt.show()


def interpolation_stage(input_image, pixelsize_old, slice_thickness_old, file_name):
    pixelsize_new = pixelsize_old
    slice_thickness_new = 2.5

    x_old = np.linspace(0, (input_image.shape[1]-1)*pixelsize_old, input_image.shape[1])
    y_old = np.linspace(0, (input_image.shape[2]-1)*pixelsize_old, input_image.shape[2])
    z_old = np.arange(0, (input_image.shape[0])) * slice_thickness_old

    print(x_old)
    print(z_old)

    method = 'linear'

    my_interpolating_object = RegularGridInterpolator((z_old, x_old, y_old), input_image, method=method,
                                                      bounds_error='False')

    x_new = np.round(input_image.shape[1]*pixelsize_old/pixelsize_new).astype('int')
    y_new = np.round(input_image.shape[2]*pixelsize_old/pixelsize_new).astype('int')
    z_new = np.arange(z_old[0], z_old[-1], slice_thickness_new)

    print(z_new)

    pts = np.indices((len(z_new), x_new, y_new)).transpose((1, 2, 3, 0))
    pts = pts.reshape(1, len(z_new) * x_new * y_new, 1, 3).reshape(len(z_new) * x_new * y_new, 3)
    pts = np.array(pts, dtype=float)
    pts[:, 1:3] = pts[:, 1:3] * pixelsize_new
    pts[:, 0] = pts[:, 0] * slice_thickness_new + z_new[0]

    print("Total z slices = ", pts.shape[0] / (input_image.shape[1] * input_image.shape[2]))

    # Interpolate
    interpolated_data = my_interpolating_object(pts)
    interpolated_data = interpolated_data.reshape(len(z_new), x_new, y_new)

    interpolated_data_16bit = interpolated_data.astype(np.uint16)

    metadata = {'spacing': slice_thickness_new, 'unit': 'um', 'axes': 'ZYX'}
    imwrite(file_name, interpolated_data_16bit, imagej=True,
            resolution=(1/pixelsize_new, 1/pixelsize_new), metadata=metadata)

    return interpolated_data_16bit
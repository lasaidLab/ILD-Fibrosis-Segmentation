import numpy as np
from tifffile import imread
import glob
import os
import cv2


path = 'mascaras_pred'

image_dict = {}
for directory_path in glob.glob(path):
    for img_path in glob.glob(os.path.join(directory_path, "*tif")):

        file_name = img_path.strip(path)
        file_splited = file_name.split('_')
        folder_name = file_splited[0]
        file_number = file_splited[1].strip('.tif')
        new_folder_name = path + '/' + file_number + '/'

        if not os.path.exists(new_folder_name):
            os.makedirs(new_folder_name)

        image = imread(img_path)

        min_value = image.min()
        max_value = image.max()

        index = 1
        for each_image in image:
            #new_file_name = new_folder_name + '/CT_' + str(index)
            each_image = np.array(each_image)

            each_image_norm = (each_image / max_value * 255).astype(np.uint8)

            #each_image_norm = each_image.astype('uint8')
            image_name = os.path.join(new_folder_name, 'CT_' + str(index) + '.jpg')
            cv2.imwrite(image_name, each_image_norm)
            index += 1

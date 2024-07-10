import numpy as np
from tifffile import imread
import glob
import os
import cv2
from keras.metrics import MeanIoU

path1 = 'gold_standard'
path2 = 'mascaras_pred'
n_classes = 3

image_dict = {}
for directory_path in glob.glob(path1):
    for img_path in glob.glob(os.path.join(directory_path, "*tif")):

        file_name = img_path.strip(path1)
        print(file_name)

        file_name_gold = path1 + file_name
        file_name_test = path2 + file_name

        image_gold = imread(file_name_gold)
        image_test = imread(file_name_test)

        IOU_keras = MeanIoU(num_classes=n_classes)
        IOU_keras.update_state(image_gold, image_test)
        print("Mean IoU =", IOU_keras.result().numpy())

        values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
        class1_IoU = values[0, 0] / (
                values[0, 0] + values[0, 1] + values[0, 2] + values[1, 0] + values[2, 0])
        class2_IoU = values[1, 1] / (
                values[1, 1] + values[1, 0] + values[1, 2] + values[0, 1] + values[2, 1])
        class3_IoU = values[2, 2] / (
                values[2, 2] + values[2, 0] + values[2, 1] + values[0, 2] + values[1, 2])

        print("Comparation type: {}".format('Complete'))
        print("IoU for class1 is: ", class1_IoU)
        print("IoU for class2 is: ", class2_IoU)
        print("IoU for class3 is: ", class3_IoU)



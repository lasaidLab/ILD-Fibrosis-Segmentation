from keras.preprocessing.image import ImageDataGenerator
import numpy as np



def lung_data_generator(X_train, y_train, seed):

    """
    Data augmentation of images used for training

    Parameters
    ----------
    X_train:
        Multidimensional array with all CT images for training
    y_train:
        Multidimensional array with all masks for training
    seed:
        Defines the main seed used to generate the random numbers.

    Returns
    ----------
    new_Xtrain:
        Multidimensional array with all augmented CT images for training
    new_ytrain
        Multidimensional array with all augmented masks for training
    """

    img_data_gen_args = dict(rotation_range=45,
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             shear_range=0.5,
                             zoom_range=0.3,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='reflect')

    mask_data_gen_args = dict(rotation_range=45,
                              width_shift_range=0.3,
                              height_shift_range=0.3,
                              shear_range=0.5,
                              zoom_range=0.3,
                              horizontal_flip=True,
                              vertical_flip=True,
                              fill_mode='reflect')

    image_data_generator = ImageDataGenerator(**img_data_gen_args)

    image_cuantity = 10
    index_im = 0
    batch_size_Xtrain = X_train.shape[0]

    new_Xtrain = X_train.copy()
    for batch in image_data_generator.flow(X_train, batch_size=batch_size_Xtrain, seed=seed):
        new_Xtrain = np.concatenate((new_Xtrain, batch))
        index_im += 1

        if index_im == image_cuantity:
            index_im = 0
            break

    new_ytrain = y_train.copy()
    for batch in image_data_generator.flow(y_train, batch_size=batch_size_Xtrain, seed=seed):
        batch = batch.astype(np.int8)
        new_ytrain = np.concatenate((new_ytrain, batch))
        index_im += 1

        if index_im == image_cuantity:
            break

    return new_Xtrain, new_ytrain

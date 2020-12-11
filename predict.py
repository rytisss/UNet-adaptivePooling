import glob
import os
import numpy as np
import cv2
from autoencoder import UNet4_First5x5
from losses import Loss


def gather_image_from_dir(input_dir):
    image_extensions = ['*.bmp', '*.jpg', '*.png']
    image_list = []
    for image_extension in image_extensions:
        image_list.extend(glob.glob(input_dir + image_extension))
    image_list.sort()
    return image_list


def get_file_name(path):
    file_name_with_ext = path.rsplit('\\', 1)[1]
    file_name, file_extension = os.path.splitext(file_name_with_ext)
    return file_name


##########################################
# Super-basic testing/prediction routine
##########################################

def predict():
    # Weights path
    weight_path = r'C:\Users\Rytis\Desktop\freda holes data 2020-10-14\Adaptive_pool/Adaptive_pool_320x320-003-0.7738.hdf5'

    # Choose your 'super-model'
    model = UNet4_First5x5(number_of_kernels=4,
                           input_size=(320, 320, 1),
                           loss_function=Loss.CROSSENTROPY50DICE50,
                           learning_rate=1e-3,
                           pretrained_weights=weight_path)

    # Test images directory
    test_images = r'C:\Users\Rytis\Desktop\freda holes data 2020-10-14\dataForTraining\Image_rois/'

    image_paths = gather_image_from_dir(test_images)

    print('Press key to skip image')

    # Load and predict on all images from directory
    for image_path in image_paths:
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # preprocess
        image_norm = image / 255
        image_norm = np.reshape(image_norm, image_norm.shape + (1,))
        image_norm = np.reshape(image_norm, (1,) + image_norm.shape)
        # predict
        prediction = model.predict(image_norm)
        # normalize to image
        prediction_image_norm = prediction[0, :, :, 0]
        prediction_image = prediction_image_norm * 255
        prediction_image = prediction_image.astype(np.uint8)

        # Do you want to visualize image?
        show_image = True
        if show_image:
            cv2.imshow("image", image)
            cv2.imshow("prediction", prediction_image)
            cv2.waitKey(0)


def main():
    predict()


if __name__ == '__main__':
    main()

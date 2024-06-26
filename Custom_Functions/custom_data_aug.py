import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates

import sys
sys.path.extend(['./','../','../Models/','../Data_Loader/','../Model_Helpers/'])

def elastic_transform(image, alpha=2000, sigma=40, alpha_affine=40, random_state=None):
    """
    Applies an elastic deformation to an image which is useful for data augmentation, particularly in medical imaging.

    Args:
    image (numpy.ndarray): The input image to be transformed.
    alpha (int, optional): Scaling factor that controls the intensity of the deformation in the elastic transform.
    sigma (int, optional): Standard deviation of the Gaussian filter used in the elastic deformation.
    alpha_affine (int, optional): Scaling factor that controls the intensity of the affine transformation.
    random_state (np.random.RandomState, optional): Random state for reproducibility of the transformations.

    Returns:
    numpy.ndarray: The deformed image.

    This function first applies an affine transformation to the image to simulate typical variations like rotations and shifts.
    Then, it applies a more complex elastic deformation using Gaussian filters and displacement fields.
    The result is an image that retains the same labels as the original but appears differently due to the transformations.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([
        center_square + square_size, 
        [center_square[0] + square_size, center_square[1] - square_size],
        center_square - square_size
    ])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)

    for i in range(shape[2]):
        image[:, :, i] = cv2.warpAffine(image[:, :, i], M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1), ksize=(blur_size:= int(4 * sigma) | 1, blur_size), sigmaX=sigma) * alpha 
    dy = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    def_img = np.zeros_like(image)
    for i in range(shape[2]):
        def_img[:, :, i] = map_coordinates(image[:, :, i], indices, order=1).reshape(shape_size)

    return def_img

def salt_pepper_noise(image, salt=0.2, amount=0.004):
    row, col, chan = image.shape
    num_salt = np.ceil(amount * row * salt)
    num_pepper = np.ceil(amount * row * (1.0 - salt))

    for n in range(chan // 2):  # //2 so we don't augment the mask
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
        image[tuple(coords), n] = 1

        # Add Pepper noise  
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        image[tuple(coords), n] = 0

    return image
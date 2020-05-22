# Adapted with permission from Doug Duhaime:
# https://gist.githubusercontent.com/duhaime/70825f9bbc7454be680616799143533f/raw/c7d7a3a518650e359889d22280ca900e924827f1/vectorize_image.py

from keras.preprocessing.image import save_img, img_to_array, array_to_img
from keras_preprocessing.image import utils as kutils
from keras.applications import Xception, VGG19, InceptionV3, imagenet_utils
import keras.backend as K
import numpy as np
import io

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None


# Doug recommends InceptionV3
model = InceptionV3(weights='imagenet')


def load_img(img_bytes, color_mode='rgb', target_size=None, interpolation='nearest'):
    # code below "borrowed" from keras_preprocessing.image.utils.load_img since we didn't want
    # to load an image from a file path
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `load_img` requires PIL.')

    img = pil_image.open(io.BytesIO(img_bytes))

    if color_mode == 'grayscale':
        # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
        # convert it to an 8-bit grayscale image.
        if img.mode not in ('L', 'I;16', 'I'):
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in kutils._PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(kutils._PIL_INTERPOLATION_METHODS.keys())))
            resample = kutils._PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


def vectorize(img_bytes: bytes) -> np.ndarray:
    # VGG16, VGG19, and ResNet take 224×224 images; InceptionV3 and Xception take 299×299 inputs
    img = load_img(img_bytes, target_size=(299, 299))
    arr = img_to_array(img)
    arr = imagenet_utils.preprocess_input(arr)

    # input shape must be n_images, w, h, colors
    arr = np.expand_dims(arr, axis=0)

    # complete forward pass through model
    # preds = model.predict(arr)

    # N.B. the idea is that the last layer representation will project all the
    # images in a way that preserves locality (similar stuff grouped together)

    # or extract values from the ith layer (e.g. the last layer == index position -1)
    out = K.function([model.input], [model.layers[-1].output])([arr])[0]

    return out

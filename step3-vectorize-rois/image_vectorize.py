# Used with permission from Doug Duhaime:
# https://gist.githubusercontent.com/duhaime/70825f9bbc7454be680616799143533f/raw/c7d7a3a518650e359889d22280ca900e924827f1/vectorize_image.py

from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from keras.applications import Xception, VGG19, InceptionV3, imagenet_utils
import keras.backend as K
import numpy as np

# Doug recommends InceptionV3
model = InceptionV3(weights='imagenet')

# VGG16, VGG19, and ResNet take 224×224 images; InceptionV3 and Xception take 299×299 inputs
img = load_img('l.jpg', target_size=(299,299))
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


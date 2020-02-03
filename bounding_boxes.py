# Stephen Krewson, HTRC ACS Project, January 2020
#
# bbox.py: Return list of illustrated "regions of interest" from an input
# image. Uses a Mask-RCNN model retrained with 19C labeled data from Internet
# Archive.
# 
# Simplified from Matterport's Mask-RCNN implementation
# https://github.com/matterport/Mask_RCNN/blob/master/samples/balloon/
#   inspect_balloon_model.ipynb
#

import numpy as np
import skimage 
import sys

from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn.visualize import display_images
import tensorflow as tf


# Not logically a part of configuration
ROI_WEIGHTS = "weights/mask_rcnn_bbox_weights.h5"


# override a couple of fields from the Matterport defaults
class InferenceConfig(Config):
    
    NAME = "ACS-Krewson"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BATCH_SIZE = GPU_COUNT * IMAGES_PER_GPU

    # Backround and ROI
    NUM_CLASSES = 2

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


# instantiate all the image processing settings
roi_config = InferenceConfig()

# Start the TensorFlow session
# N.B. allows for fallback to CPU if no GPU
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    
    # model_dir just controls where stuff gets logged?
    roi_model = modellib.MaskRCNN(
        mode="inference",
        model_dir="weights",
        config=roi_config
    )

# Give the model the big h5 file of weights (from COCO CompVis challenge)
print("Loading Mask-RCNN weights...")
roi_model.load_weights(ROI_WEIGHTS, by_name=True)
print("Loaded!")


# Some lightly adapted utilities
def load_image(image_id):
    """Load the specified image and return a [H,W,3] Numpy array."""
    image = skimage.io.imread(image_id)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image

# N.B. model is passed in as an argument...

def detect_rois(image_id, model):
    """Use model.detect() to slide CNN windows over image and return
    rectangular regions that meet some threshold activation.
    """
    image = load_image(image_id)
    
    image, window, scale, padding, crop = utils.resize_image(image, min_dim=IMAGE_MIN_DIM, min_scale=IMAGE_MIN_SCALE, max_dim=IMAGE_MAX_DIM, mode=IMAGE_RESIZE_MODE)

    # turn of logging
    results = model.detect([image], verbose=0)
    
    # from model.py: rois: [N, (y1, x1, y2, x2)] detection bounding boxes
    regions = results[0]['rois']

    # return array of crops; slice the array like so: y1:y2, x1:x2
    return [image[r[0]:r[2], r[1]:r[3]] for r in regions]


if __name__ == "__main__":
    print("Hello world!")
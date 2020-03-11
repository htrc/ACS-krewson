import argparse
import json
import logging
import os
# from functools import partial
from typing import List, Iterator, TextIO
from zipfile import ZipFile
# from multiprocessing import Pool

import imageio
from numpy import ndarray
from skimage import io, color

import mrcnn.model as modellib
from mrcnn.config import Config


class PageImage:
    @classmethod
    def from_json(cls, json: dict) -> 'PageImage':
        seq = json['seq']
        img_type = json['img_type']
        img_path = json['img_path']
        label = json['label']
        return cls(seq, img_type, img_path, label)

    def __init__(self, seq: str, img_type: str, img_path: str, label: str) -> None:
        self.seq = seq
        self.img_type = img_type
        self.img_path = img_path
        self.label = label

    def __repr__(self) -> str:
        return str(self.seq)

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__)


class Volume:
    @classmethod
    def from_json(cls, json: dict) -> 'Volume':
        vol_id = json['vol_id']
        zip_path = json['zip_path']
        pages = [PageImage.from_json(page_json) for page_json in json['pages']]
        return cls(vol_id, pages, zip_path)

    def __init__(self, vol_id: str, pages: List[PageImage], zip_path: str) -> None:
        self.vol_id = vol_id
        self.pages = pages
        self.zip_path = zip_path

    def __repr__(self) -> str:
        return self.vol_id

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__)


def parse_input(in_json: TextIO) -> Iterator[Volume]:
    for line in in_json:
        vol_json = json.loads(line)
        yield Volume.from_json(vol_json)


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


def dir_path(path: str) -> str:
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid path.".format(path))


# Some lightly adapted utilities
def load_image(bytes) -> ndarray:
    """Load the specified image and return a [H,W,3] Numpy array."""
    image = io.imread(bytes, plugin='imageio')
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


# Pass in connection to the model and the configuration
def detect_rois(image, model) -> List[ndarray]:
    """Use model.detect() to slide CNN windows over image and return
    rectangular regions that meet some threshold activation.
    """
    # turn off logging
    results = model.detect([image], verbose=0)

    # from model.py: rois: [N, (y1, x1, y2, x2)] detection bounding boxes
    regions = results[0]['rois']

    # return array of crops; slice the array like so: y1:y2, x1:x2
    return [image[r[0]:r[2], r[1]:r[3]] for r in regions]


def process_vol(vol: Volume, data_dir: str, model: modellib.MaskRCNN) -> Volume:
    logger = logging.getLogger("process_vol")
    logger.info("Processing %s", vol.vol_id)
    zip_path = os.path.join(data_dir, vol.zip_path)
    logger.debug("Processing %s", zip_path)
    with ZipFile(zip_path, 'a') as vol_zip:
        for page_image in vol.pages:
            img_bytes = vol_zip.read(page_image.img_path)
            image = load_image(img_bytes)
            rois = detect_rois(image, model)
            logger.debug("%s: %d regions", page_image.img_path, len(rois))
            img_dir, img_file = os.path.split(page_image.img_path)
            img_name, _ = os.path.splitext(img_file)
            for i, roi in enumerate(rois):
                roi_bytes = imageio.imwrite(uri=imageio.RETURN_BYTES, im=roi, format='jpg')
                roi_path = os.path.join(img_dir, '{}_{}.jpg'.format(img_name, str(i).zfill(2)))
                vol_zip.writestr(roi_path, roi_bytes)
    return vol


def main(args):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s', level=logging.INFO)
    logger = logging.getLogger("main")

    # instantiate all the image processing settings
    roi_config = InferenceConfig()
    roi_model = modellib.MaskRCNN(
        mode='inference',
        model_dir='weights',
        config=roi_config
    )

    logger.info("Processing started...")
    logger.info("Loading Mask-RCNN model weights from {}".format(args.weights))
    roi_model.load_weights(args.weights, by_name=True)
    logger.info("Model loaded")

    volumes = parse_input(args.input)

    # pool = Pool()
    # try:
    #     for volume in pool.imap_unordered(partial(process_vol, data_dir=args.data, model=roi_model), volumes):
    #         pass
    # except KeyboardInterrupt:
    #     exit()
    # finally:
    #     pool.terminate()
    #     pool.join()

    for volume in volumes:
        process_vol(volume, args.data, roi_model)

    args.input.close()

    logger.info("All done")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s', level=logging.INFO)
    logger = logging.getLogger("main")

    parser = argparse.ArgumentParser(description="Returns the list of illustrated 'regions of interest' "
                                                 "from an input image. Uses a Mask-RCNN model retrained "
                                                 "with 19C labeled data from Internet Archive.")
    parser.add_argument('-i', '--input', required=True, type=argparse.FileType('r', encoding='UTF-8'),
                        help="Path to input file")
    parser.add_argument('-d', '--data', required=True, type=dir_path, help="Path to the data directory")
    parser.add_argument('-w', '--weights', required=True, type=str, help="Path to the model weights file")

    args = parser.parse_args()

    main(args)

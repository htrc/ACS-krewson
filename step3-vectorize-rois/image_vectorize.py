import argparse
import json
import logging
import os
import re
import numpy as np
from typing import List, Iterator, TextIO, Optional
from zipfile import ZipFile
from vectorizer import vectorize


id_encode = lambda id: id.replace(":", "+").replace("/", "=").replace(".", ",")
roi_regexp = re.compile(r'(?P<seq>\d{8})_(?P<roi>\d{2})\.jpg$', re.IGNORECASE)


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


def save_vector(vec: np.ndarray, htid: str, seq: str, roi: str, output_dir: str) -> str:
    libid, volid = htid.split('.', 1)
    clean_volid = id_encode(volid)
    dir = os.path.join(output_dir, libid, clean_volid[::3])
    os.makedirs(dir, exist_ok=True)
    filename = "{}.{}_{}_{}.npy".format(libid, clean_volid, seq, roi)
    filename = os.path.join(dir, filename)
    np.save(filename, vec)
    return filename


def is_roi(fn: str) -> Optional[tuple]:
    m = roi_regexp.search(fn)
    return (fn,) + m.groups() if m is not None else None


def process_vol(vol: Volume, data_dir: str, output_dir: str):
    logger = logging.getLogger("process_vol")
    logger.info("Processing %s", vol.vol_id)
    zip_path = os.path.join(data_dir, vol.zip_path)
    logger.debug("Processing %s", zip_path)
    with ZipFile(zip_path) as vol_zip:
        rois = [roi for roi in map(lambda zi: is_roi(zi.filename), vol_zip.infolist()) if roi is not None]
        logger.debug('Found %d ROIs', len(rois))
        for (img_path, seq, roi) in rois:
            img_bytes = vol_zip.read(img_path)
            vec = vectorize(img_bytes)
            vec_path = save_vector(vec, vol.vol_id, seq, roi, output_dir)
            logger.debug("Saved vector %s", vec_path)


def dir_path(path: str) -> str:
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid path.".format(path))


def main(args):
    logger.info("Processing started...")
    volumes = parse_input(args.input)

    for volume in volumes:
        process_vol(volume, args.data, args.output)

    logger.info("All done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vectorizes images and saves them as numpy arrays "
                                                 "to a user-specified output directory")
    parser.add_argument('-i', '--input', required=True, type=argparse.FileType('r', encoding='UTF-8'),
                        help="Path to input file")
    parser.add_argument('-d', '--data', required=True, type=dir_path, help="Path to the data directory")
    parser.add_argument('-o', '--output', required=True, type=str, help="Path to the output directory")
    parser.add_argument('-v', '--verbose', action="store_true", help="Enable debug output")

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("main")

    main(args)

import argparse
import json
import os
import io
import shutil
import time
import logging
from fastai.vision import *
from fastai.vision.image import open_image
from torch.multiprocessing import Pool, set_start_method
from zipfile import ZipFile

from typing import List, Iterator, TextIO, Set


class PageImage:
    @classmethod
    def from_json(cls, json: dict) -> 'PageImage':
        seq = json['seq']
        img_type = json['mimeType']
        img_path = json['imgZipPath']
        return cls(seq, img_type, img_path)

    def __init__(self, seq: str, img_type: str, img_path: str) -> None:
        self.seq = seq
        self.img_type = img_type
        self.img_path = img_path

    def __repr__(self) -> str:
        return str(self.seq)

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__)


class Volume:
    @classmethod
    def from_json(cls, json: dict) -> 'Volume':
        vol_id = json['id']
        zip_path = json['zipPath']
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


def process_vol(vol: Volume, data_dir: str, out_dir: str, learner, match_labels: Set[str]) -> Optional[Volume]:
    logger = logging.getLogger("process_vol")
    logger.info("Processing %s", vol.vol_id)
    zip_path = os.path.join(data_dir, vol.zip_path)
    out_zip_path = os.path.join(out_dir, vol.zip_path)
    path, _ = os.path.split(out_zip_path)
    os.makedirs(path, exist_ok=True)
    keep = False
    logger.debug("Processing %s", zip_path)
    good_pages = []
    with ZipFile(zip_path, 'r') as vol_zip, ZipFile(out_zip_path, 'w') as out_zip:
        for page_image in vol.pages:
            img_bytes = vol_zip.read(page_image.img_path)
            image = open_image(io.BytesIO(img_bytes))
            label, _, _ = learner.predict(image)
            logger.debug("%s: %s", page_image.img_path, str(label))
            if str(label) in match_labels:
                keep = True
                page_image.label = str(label)
                good_pages.append(page_image)
                base, _ = os.path.splitext(page_image.img_path)
                txt_path = base + ".txt"
                page_ocr_txt = vol_zip.read(txt_path)
                out_zip.writestr(page_image.img_path, img_bytes)
                out_zip.writestr(txt_path, page_ocr_txt)

    if not keep:
        shutil.rmtree(path)

    vol.pages = good_pages
    return vol


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid path.".format(path))


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s', level=logging.INFO)
    logger = logging.getLogger("main")

    parser = argparse.ArgumentParser(description="Using a 12-class CNN model, runs inference on a set of page images "
                                                 "and saves only those images for which the class inline_image was "
                                                 "the highest prediction.")
    parser.add_argument('-i', '--input', required=True, type=argparse.FileType('r', encoding='UTF-8'),
                        help="Path to input file")
    parser.add_argument('-d', '--data', required=True, type=dir_path, help="Path to the data directory")
    parser.add_argument('-o', '--output', required=True, type=str,
                        help="Path to output directory where the saved images will be written")
    parser.add_argument('-j', '--job', required=False, type=str, default='matches',
                        help="Job identifier")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    logger.info("Processing started...")

    learner = load_learner(path="model")
    logger.info("Model loaded")

    match_labels = {"inline_image", "plate_image"}

    volumes = parse_input(args.input)

    with open(os.path.join(args.output, '{}.json'.format(args.job)), 'w', encoding='UTF-8') as out:
        # for vol in volumes:
        #     matched_vol = process_vol(vol, data_dir=args.data, out_dir=args.output,
        #                               learner=learner, match_labels=match_labels)
        #     logging.info("[%s] Processed %s (%d pages matched)", time.ctime(), matched_vol.vol_id, len(matched_vol.pages))
        #     if matched_vol.pages:
        #         out.write(matched_vol.to_json() + '\n')
        #         out.flush()

        set_start_method('fork')
        try:
            pool = Pool()
            for vol in pool.imap_unordered(partial(process_vol, data_dir=args.data, out_dir=args.output, learner=learner, match_labels=match_labels), volumes):
                logger.info("Processed %s (%d pages matched)", vol.vol_id, len(vol.pages))
                if vol.pages:
                    out.write(vol.to_json() + '\n')
                    out.flush()
        except Exception:
            logger.exception("Multiprocessing pool error")
        except KeyboardInterrupt:
            exit()
        finally:
            pool.terminate()
            pool.join()

    args.input.close()

    logger.info("All done")


if __name__ == '__main__':
    main()

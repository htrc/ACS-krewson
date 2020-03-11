# Stephen Krewson, August 24, 2019
#
# filter_image_csv.py
#
# Input: a CSV file where an image resource is the *first* column
# Output: the same CSV file but with all the rows removed that do not
# have `inline_image`as the top prediction for the image
#
# Example usage, with export.pkl model file in the working directory:
#
# python filter_image_csv.py --input test.csv --output .
#
# Does NOT work as-is on Windows (PyTorch multiprocessing)
# see: https://pytorch.org/docs/stable/notes/windows.html

import argparse
import csv
from fastai.vision import *
import imageio
import os
from fastai.vision.image import open_image


def parse_args():
    parser = argparse.ArgumentParser(description='Using a 12-class CNN model, runs inference on a CSV list of image assets and writes out a list of only those images for which the class inline_image was the highest prediction.')
    parser.add_argument('--input', required=True, type=file_path, help='Path to input CSV file.')
    parser.add_argument('--output', required=True, type=dir_path, help='Path to output directory for filtered CSV file. Must have write permissions.')
    return parser.parse_args()


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"dir:{path} is not a valid path.")


def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"file:{path} is not a valid file.")


### Main script ###
# Based on: https://docs.fast.ai/tutorial.inference.html
# image CSV, export.pkl need to be in same dir as script
args = parse_args()

# fastai helper to work with list of image paths in first column of CSV
# `test` argument allows inference on multiple images
# `cols` is which column of the CSV to use for the image paths

learner = load_learner(".")

# set the labels that we want to keep; get their indices
# full list is in learner.data.classes
good_labels = ["inline_image", "plate_image"]

with open(args.input) as csv_in:
    for image_path in csv_in:
        image_path = image_path.rstrip()
        image = open_image(image_path)
        label, _, _ = learner.predict(image)
        if str(label) in good_labels:
            print(image_path)

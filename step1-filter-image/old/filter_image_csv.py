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


def parse_args():
    parser = argparse.ArgumentParser(description='Using a 12-class CNN model, runs inference on a CSV list of image assets and writes out a list of only those images for which the class inline_image was the highest prediction.')
    parser.add_argument('--input', required=True, type=str, help='Path to input CSV file.')
    parser.add_argument('--output', required=True, type=dir_path, help='Path to output directory for filtered CSV file. Must have write permissions.')
    return parser.parse_args()


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"dir:{path} is not a valid path.")


### Main script ###
# Based on: https://docs.fast.ai/tutorial.inference.html
# image CSV, export.pkl need to be in same dir as script
args = parse_args()

# fastai helper to work with list of image paths in first column of CSV
# `test` argument allows inference on multiple images
# `cols` is which column of the CSV to use for the image paths

# N.B. filepaths MUST be absolute (relative to "/")
input_csv = os.path.abspath(args.input)
imgs = ImageList.from_csv("/", input_csv, cols=0)

# this is the export.pkl file, should be in CWD
learner = load_learner(".", test=imgs)

## learner.predict(IMG) --> run in for-loop and compare
# get an IDE! read docs on multiprocessing in torch
# findout good batch size ~333

# set the labels that we want to keep; get their indices
# full list is in learner.data.classes
good_labels = ["inline_image", "plate_image"]
good_idxs = [3, 8]

# we get back the raw probabilities for each class
# and ground truth labels (which don't exist for test data)
probs, _ = learner.get_preds(ds_type=DatasetType.Test)

# turn into a vector of max probabilities for all images
preds = torch.argmax(probs, dim=1)

# list of all image paths that we want to keep
out = [os.path.normpath(imgs.items[i]) for i,p in enumerate(preds) if p in good_idxs]

# print list of all 
print('\n'.join(out))

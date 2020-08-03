# Deriving Basic Illustration Metadata

This repository contains code used in the 2019-20 HathiTrust Research Council Advanced Collaborative Support Grant, "Deriving Basic Illustration Metadata." Two sample notebooks for working with this data are also included.

The metadata files produced from the project are available on Zenodo:

http://zenodo.org/record/3940528#.XyRNSZ5KjIU

The project reports can be found here:

[Midpoint report](https://wiki.htrc.illinois.edu/display/COM/A+Half-Century+of+Illustrated+Pages%3A+ACS+Lab+Notes)

[Final report]()

The model used in Stage 2 was adapted from Matterport's Mask-RCNN model:

https://github.com/matterport/Mask_RCNN

```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```

The script in Stage 3 is adapted from Doug Duhaime:

https://gist.githubusercontent.com/duhaime/70825f9bbc7454be680616799143533f/raw/c7d7a3a518650e359889d22280ca900e924827f1/vectorize_image.py
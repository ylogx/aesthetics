# Aesthetics


## Image Aesthetics Definition
Image aesthetic evaluation aims to classify photos into high quality or low quality from the perspective of human.

<img alt="Good vs Bad Image" src="https://i.imgur.com/X0aZWFd.png" height=300/>

Image Aesthetics drills down to a classification problem:
* Low Quality Image
* High Quality Image

A commonly used dataset for image aesthetics is AVA (Image Aesthetic Visual Analysis) dataset


This repo provides following tools to help in image aesthetics problem:
* [Fisher Vector implementation](https://github.com/shubhamchaudhary/aesthetics/tree/master/aesthetics/fisher)
* AVA dataset and fast multi-threaded downloader



## Fisher Vector
Fisher Vector is a technique for generating features for images, which can be used by discriminative models like SVM. You can use fisher vectors for usecases like image classification (ImageNet), image aesthetics.
<!-- Describe Patches by their deviation from Universal Generative Mixture Model. -->

### Flow
* We create local descriptors using SIFT for each image in the training set
* We fit a Gausian Mixture Model (GMM) on descriptors for all images in training set.
* Using this global GMM we generate features for each image

![Fisher Vector flow](https://i.imgur.com/S5oAnEU.png)

### Spatial Pooling
Spatial pooling is a technique to save the spatial information of the image while generating features. This is very important in image aesthetics because the look and feel of the image are highly dependent on the aspect ratio, placement of the objects in the image.

Following image should clarify the importance of spatial pooling:

![Importance of spatial pooling](https://i.imgur.com/nZ3aYkL.png)

For spatial pooling, the fisher vector paper recommends splitting the image into 4 versions:
* Full image
* 3 horizontal slices of the image

To generate the fisher vector of the image, as shown in the flowchart above, we concat the fisher vectors of the 4 individual versions of the image.

## AVA Downloader
```sh
./download.py --help
./download.py ava
```

This is a WIP


<!--# Downloaded dataset-->

<!--ECCV 2016:-->
<!--* [Dataset Link (2 GB)](dataset_link)-->
<!--* [Dataset 256x256 size cropped preview (132 MB)](dataset_preview_link)-->


<!--[dataset_link]: https://drive.google.com/open?id=0BxeylfSgpk1MN1hUNHk1bDhYRTA-->
<!--[dataset_preview_link]: https://drive.google.com/open?id=0BxeylfSgpk1MU2RsVXo3bEJWM2c-->

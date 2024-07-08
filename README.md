<div align="center">

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF8000?style=for-the-badge&logo=tensorflow&logoColor=white)](https://github.com/tensorflow/tensorflow)
[![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io/)
[![Autoencoder](https://img.shields.io/badge/Autoencoder-66B2FF.svg?style=for-the-badge&logoColor=white)](https://en.wikipedia.org/wiki/Autoencoder)
</div>


# Retinal Vessel Segmentation

## Introduction
Implementing an Autoencoder architecture neural network for segmentation of cropped patches of the grayscaled retinal image


## Autoencoder
![autoencoder](/Image/autoencoder.png)

Autoencoder, or an "hourglass" model, is a simple extension of the image classification model. The process of obtaining the compressed feature shares the same concept of abstraction path introduced in the classification model. The compressed feature is then propagated to the expansion path, which de-compresses the compressed feature back to the spatial representation of the image. In this arrangement, the compressed feature is often called the "Bottleneck" feature.


Autoencoder architecture is used to build a model that can learn to produce a segmentation mask from the input image. Intuitively, the output of the autoencoder is expected to be the predicted segmentation mask. During training, the loss will be computed between the real label mask and the predicted mask.


## Instruction
```
$ python3 main.py
```


## Result
[![Model Result](Image/result.png)](model.py)



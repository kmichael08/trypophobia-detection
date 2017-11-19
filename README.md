Detecting trypophobia trypophobia triggers - Brainhack project

This repository contains Jupyter notebook implementations of various Convolutional Nets architectures that were trained for classifying potentially trypophobic images. The project was done as a part of the 'AON - Brainhack Warsaw 2017' conference held at University of Warsaw and was supervised by Piotr Migdał.

The dataset that was used for training can be in this repo: https://github.com/cytadela8/trypophobia

These notebooks use Python 3.6 and Keras 2.0.8. The Jupyter notebooks implementations of the models used can be found here:
- first_model.ipynb - Baseline model, 82.79% accuracy on validation set
- vgg_01.ipynb - pretrained VGG, 91.61% accuracy
- resnet_large.ipynb - pretrained ResNet, 93.10% accuracy

The models were trained on the Google Computing Platform using Neptune. For the two latter models we included a trained model and weights. The results show that more complex architectures (with >10M parameters) perform better on validation set at the cost of longer training time.

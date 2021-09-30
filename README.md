# Traffic Sign Recognition

Implementation of a Convolutional Neural Network based on the ResNet architecture, trained for the recognition of traffic signs with the [GTSRB dataset](https://benchmark.ini.rub.de/gtsrb_dataset.html).



## Repository Structure

* utils.py: contains helper functions
* preprocessing.py; functions for image processing 
* loader.py: loads the images and labels into the appropriate format 
* training.py: CNN training
* _architecture.png: diagram of the implemented network

## How to Run

1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acmarkes/traffic-sign-recog/blob/c74611f3e681291b2f9ada8220cc24e706fa0096/colab_training.ipynb)
2. That's it

### OR

1. Download the training dataset [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip)
2. Clone this repo to your local machine and extract the downloaded zip into the same directory. Make sure you have the modules required installed (pip install - r requirements.txt)
3. Run loader.py
4. Run training.py


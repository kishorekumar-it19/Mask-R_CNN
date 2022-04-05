# Mask-R_CNN

### An Algorithm for Vechicles Identification

In notebook we will be working with the vechicles[dataset](/content/drive/MyDrive/Codebugged. AI) created by [Kishore Kumar M] with a help of google images.

## What is Artificial intelligence (AI) ?

[AI](https://plato.stanford.edu/entries/artificial-intelligence): Artificial Intelligence (AI) Artificial intelligence leverages computers and machines to mimic the problem-solving and decision-making capabilities of the human mind.

### What is Machine Learning (ML) ?
[ML](https://en.wikipedia.org/wiki/Machine_learning) : A Machine Learning is an subset of AI and it learns from historical data, builds the prediction models, and whenever it receives new data, predicts the output for it. The accuracy of predicted output depends upon the amount of data, as the huge amount of data helps to build a better model which predicts the output more accurately.



#### What is Deep Learning (DL) ?

[DL](https://en.wikipedia.org/wiki/Deep_learning) : Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers. These neural networks attempt to simulate the behavior of the human brain—albeit far from matching its ability—allowing it to “learn” from large amounts of data. While a neural network with a single layer can still make approximate predictions, additional hidden layers can help to optimize and refine for accuracy.





#### How does a Convolutional Neural Network function ?  

[deep.ai](https://deepai.org/machine-learning-glossary-and-terms/convolutional-neural-network): CNNs process images as volumes, receiving a color image as a rectangular box where the width and height are measure by the number of pixels associated with each dimension, and the depth is three layers deep for each color (RGB). These layers are called channels. Within each pixel of the image, the intensity of the R, G, or B is expressed by a number. That number is part of three, stacked two-dimensional matrices that make up the image volume and form the initial data that is fed to into the convolutional network. The network then begins to filter the image by grouping squares of pixels together and looking for patterns, performing what is known as a convolution. This process of pattern analysis is the foundation of CNN functions.<br><br>


![CNN](https://miro.medium.com/max/2510/1*vkQ0hXDaQv57sALXAJquxA.jpeg)

Mask R-CNN :

Mask RCNN is a deep neural network aimed to solve instance segmentation problem in machine learning or computer vision. In other words, it can separate different objects in a image or a video. You give it a image, it gives you the object bounding boxes, classes and masks.



### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

● Data collection and annotation manually

● Object Detection with TensorFlow

● Preparing the Model Configuration Parameters 

● Building the Mask R-CNN Model Architecture 

● Loading the Model Weights 

● Reading an Input Image 

● Detecting Objects 

● Visualizing the Results 

● Complete Code for Prediction 

● Downloading the Training Dataset 

● Preparing the Training Dataset 

● Preparing Model Configuration
Training Mask R-CNN in TensorFlow 
 
We first mount the folder on the google drive with the dataset. 

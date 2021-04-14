# Emotion-AI #
- - - - 
## I.	Introduction ##
- Artificial emotional intelligence or Emotion AI is a branch of AI that allows computers to understand human non-verbal cues such as body language and facial expressions.

## II.	Problem statement ##

- The aim of this key project is to classify peoples emotion based on their face images .
- In this case study I will assume that I AI /ML consultant I've been hired by a startup to build train and deploy a system that automatically monitors people emotions and expressions .
- For this purpose, the team has collected more than 20,000 facial images with their associated facial expression labels and around 2000 images with their facial key point in annotations.

## III.	Proposed solution ##

- To find a solution for the given a problem statement I have proposed model based on emotion AI model hey that takes an original image as an input and applies the algorithm to give the required output.
- To achieve this output the proposed model is divided into 2 smaller prediction model. the first model is based on the facial key point detection and the second model is based on the facial expression that is in motion detection model both the model takes original image as input model one predict the facial key points on the image while the second model predicts the emotions class.
- In the end, the 2 models are combined to get the required output.

- In the first model that is facial key point detection I create am deep learning model based convolutional neural network and residual blocks to predict facial key points. The data set consists of x and y coordinates of 15 facial key points. Input images are 96 x 96 pixels. Images consists of only one color channel (Gray-Scale images).
- Model construction is divided into several parts and tasks.
- Parts are as follows: 
    1.	Key facial point detection. 
    2.	Facial expression detection. 
    3.	Combining both facial expression and key points detection models. 
    4.	Deploying both trained models. 

- Tasks are as follows:
    1.	Understanding the problem statement in business case. 
    2.	Importing libraries and data sets. 
    3.	Perform image visualization.
    4.	Perform image of augmentation.
    5.	Perform data normalization and training data preparation.
    6.	Understand the theory and intuition behind neural networks.
    7.	Understand neural networks training process and gradient descent algorithm.
    8.	Understanding the theory and intuition behind convolutional neural networks and resnets.
    9.	Building deep residual neural networks key facial points detection model.
    10.	Compile and train key facial points detection deep learning model.
    11.	Assess trained key facial points detection model performance.
    12.	Import and explore dataset for facial expression detection.
    13.	Visualize images and plot labels.
    14.	Perform data preparation and image augmentation.
    15.	Build and train deep learning model for facial expression classification.
    16.	Understand how to assess classifier models (confusion matrix, accuracy, precision, and recall).
    17.	Assess the performance of trained facial expression classifier model.
    18.	Combine both models (1) Facial key detection and (2) Facial expression classification models.
    19.	Save the trained model for deployment.
    20.	Server the trained model using TensorFlow serving.
    21.	Make requests to model in TensorFlow serving.
 
## V.	Result ##
- The model after combining the two smaller models gave the following result in the assessing step of the model.


## VI.	Conclusion ##
- Successfully implemented and completed this project/case study. As required in the problem statement got the same output. The model which I built by combining the 2 different models i.e., model 01 – Facial key points detection model and model 02 – Facial expression classification model, just performed as required in order to get the required output for the problem statement given to me.

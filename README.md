# ECE285_CNN_FruitClassification
## Description
This is a project about classification of 60 types of fruits by team "Triple SD". In this project, we built our own CNN models using Tensorflow and also compared the results with transfer learning using VGG16 model. Given a not complex dataset that contains sufficient images, blindly using transfer learning would lead to a higher tendency of overfitting. Choosing a model with 4 hidden layers, we then carefully fine-tuned and used data augmentation and regularization to avoid overfitting. To summarize, using AdamOptimizer (learning rate=0.001), a mini-batch size of 256, our model could reach an accuracy of about 98.3\% after 30 epochs.
## Requirements
* Language: Python(Version 3)
* Frameworks: Tensorflow, Keras
## Code Organization
demo.ipynb  - - Run a demo of our CNN models with augmented data (reproduce Table 3 of our report) <br />
train.ipynb  - - Run training and test of our CNN models with original data (reproducde Figure ) <br />
train_augmented.ipynb  - - Run training and test of our CNN models with augmented data (reproducde Figure ) <br />
train_regularized.ipynb  - - Run training and test of our CNN models with regularization (reproducde Figure ) <br />
VGG16.ipynb  - - Run training and test using VGG16 model (reproduce Table  <br />
data preprocessing.py  - - Code for data preprocessing and origin data (as described in Section ) <br />
DA-Gaussian.py  - - Data augmented by adding Gaussian noise (as described in Section ) <br /> 
DA-salt&pepper.py  - - Data augmented by adding salt&pepper noise (as described in Section ) <br />
DA-perspective.py  - - Data augmented by adding perspective transformation (as described in Section ) <br />
model.dat - - Our model trained as described in Section 
## Authors
Team: Triple SD <br />
Members: Huadong Zhang, Xiao Sai, Yiyuan Xing

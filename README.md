# ECE285_CNN_FruitClassification
## Description
This is a project about classification of 60 types of fruits by team "Triple SD". In this project, we built our own CNN models using Tensorflow and also compared the results with transfer learning using VGG16 model. Given a not complex dataset that contains sufficient images, blindly using transfer learning would lead to a higher tendency of overfitting. Choosing a model with 4 hidden layers, we then carefully fine-tuned and used data augmentation and regularization to avoid overfitting. To summarize, using AdamOptimizer (learning rate=0.001), a mini-batch size of 256, our model could reach a maximun accuracy of about 98.9\% after 30 epochs.
## Requirements
* Language: Python(Version 3)
* Frameworks: Tensorflow, Keras
## Code Organization
demo.ipynb  - - Run a demo of our CNN models (reproduce Table 5 of our report, run in python2 on the server since python3 is somewhere incompitable with Keras) <br />
train.ipynb  - - Run training and test of our CNN models with original data (reproducde Table 2,3, Figure 5) <br />
train_augmented.ipynb  - - Run our CNN models with augmented data (reproduce Tabel 5) <br />
train_regularized.ipynb  - - Run of our CNN models with regularization (reproduce result in Section 4.1.5) <br />
VGG16.ipynb  - - Run training and test using VGG16 model (reproduce Table 6) <br />
data preprocessing.py  - - Code for data preprocessing and origin data (as described in Section 3.1.1) <br />
DA-Gaussian.py  - - Data augmented by adding Gaussian noise (as described in Section 3.1.2) <br /> 
DA-salt&pepper.py  - - Data augmented by adding salt&pepper noise (as described in Section 3.1.2) <br />
DA-perspective.py  - - Data augmented by adding perspective transformation (as described in Section 3.1.2) <br />
model.h5 - - Our fine-tuned model (as described in Section 3.2) <br />
test set.py - - Set for running the demo <br />
## Authors
Team: Triple SD <br />
Members: Huadong Zhang, Xiao Sai, Yiyuan Xing

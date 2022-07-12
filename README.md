# Manifold-Learning

Rs-fMRI data is highly dimensional and several feature extraction method can be perform in order to reduce the dimensionality of the data. The most common approaches are Principal component analysis (PCA) and Independent Component Analysis (ICA). However, these methods are essentially  linear transformation and cannot extract nonlinear structures. In order to overcome the nonlinear dimensionality reduction, an autoencoder can be implemented which, unlike PCA, can learn non-linear transformations with a non-linear activation function and multiple layers. In this project, we will used the stroke dataset available from previuous studies.
In addition, the extracted features are used as predictors of neurophsychological score using languages scores as behavioral domain. This task is perform by simply solving a linear regression problem with **ElasticNET** regression.

The project is organized in two main folders:


 **Feature extraction methods**: 
 - this folder is subdivide into linear transformation models:
    *  1.1 Principal Component Analysis
    * 1.2 Independent Component Analysis
    * 1.3 AE with linear activation function: it consist of one dense layer with a linear activation function. This model is mainly to compare with the PCA (which the literature stated that they should be similar). 
        
- and non linear transformation models:
     * 1.1  AE with NON linear activation function: it consist of one dense layer with a non linear activation function (LeakyReLU).
     * 1.2  Convolutional Autoencoder applied directly to the stroke dataset. 
     * 1.3  Convolutional Autoencoder applied to the augmentated data obtained by mix-up strategy from the original dataset.
     * 1.4  Transfer learning method using the Human Connectome Project (HCP).
     * 1.5 **Overcomplete** Convolutional Autoencoder with L1 regularization
     * 1.6 **Overcomplete** Convolutional Autoencoder with k-sparse regularization
     * 1.7 **Overcomplete** Transfer Learning with k-sparse regularization
     
 **Regularization**: in this folder you can find the notebook used to perform elasticnet regression on the extracted features and the behavioral scores. The parameters of the model were obtained using LOOCV and NESTED LOOCV.


In notebook utilites you can find main .py notebooks containing general function used between all the other jupyter notebooks.

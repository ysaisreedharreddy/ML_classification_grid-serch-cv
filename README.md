SVM Model Optimization with Grid Search
This repository houses a Python implementation of Support Vector Machine (SVM) classification, optimized using Grid Search to identify the best hyperparameters for predicting user behavior based on social network advertisement data.

Features
Data Preprocessing: Includes scaling features and encoding categorical variables.
SVM Classification: Utilizes the SVM classifier from scikit-learn, configured to test various hyperparameters.
Grid Search Optimization: Applies Grid Search to find the most effective SVM parameters such as C, kernel, and gamma.
Model Evaluation: Evaluates the model using accuracy metrics, confusion matrices, and cross-validation.
Visualization: Provides functions to visualize decision boundaries for training and test datasets.

Dataset
The dataset (Social_Network_Ads.csv) contains features such as Age and Estimated Salary, alongside a Purchase indicator that denotes whether a purchase was made. This data is used to demonstrate how machine learning can predict user behavior.


Results
Outputs the confusion matrix and accuracy of the SVM model on the test set.
Displays the best parameters and the accuracy obtained from Grid Search.
Visual representations of the model's decision boundaries for better understanding and presentation.

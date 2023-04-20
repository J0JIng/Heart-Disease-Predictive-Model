# SC1015: DSAI Mini Project - Predicting Heart Disease

School of Computer Science and Engineering SCSE\
Nanyang Technological University NTU\
Lab: C133 \
Group : 8 

Members: 
1. O Jing ([@J0JIng](https://github.com/J0JIng))
2. Ridhwan Hakim ([@rorodevelopment](https://github.com/rorodevelopment))
3. Vignesh Mani Senthilnathan ([@VigneshManiSenthilnathan](https://github.com/VigneshManiSenthilnathan))


---
### Description:
This repository contains the project files for the SC1015 mini-project developed by students from the SCSE at NTU. Listed here are the ipynb files used, which should be viewed in numerical order:
1. EDA and Data processing  
2. Multivariate Decision Tree model
3. Random Forest model
4. Logistic regression model
5. Support vector machine (SVM) model
---
### Table of Contents:
1. [Introduction and Problem Formulation]
2. [Exploratory Data Analysis and Data Preparation]
3. [Methodology]
4. [Experiment]
5. [Data Driven Insights and Conclusion]
---
## 1. [Introduction and Problem Formulation]

**Our Dataset:** [Heart Failure Prediction Dataset on Kaggle] \
**Problem Statement:** Discovering the key features or symptoms that determine the patient's likelihood of heart diseases

**Introduction:**
Heart disease is a major health concern that affects a significant portion of the population. In this project, we aim to identify the most important features that contribute to the prediction of heart disease in patients.

**Significance:**
By identifying the most important features that contribute to heart disease, we can develop more accurate and efficient predictive models that can be used to improve patient outcomes. Additionally, our findings could help healthcare professionals prioritise certain risk factors when diagnosing and treating heart disease.

## 2. [Exploratory Data Analysis and Data Preparation]
Exploratory Data Analysis (EDA) is a crucial step in gaining a deeper understanding of the data and making informed decisions about how to preprocess the data before modelling. EDA uncovers patterns, trends, and relationships in data and helps detect outliers and anomalies.

In this phase, we performed the following steps:

1. Explored categorical features with respect to the target variable, **'HeartDisease'**. We analysed the distribution of each category and checked whether the feature is a predictor of the target variable.
2. Explored numerical features with respect to the target variable, **'HeartDisease'**. We analysed the distribution of each feature, checked for outliers, and checked whether the feature is a predictor of the target variable.
3. Visualiaed the data using various charts and plots, such as histograms, box plots, and correlation matrix. We used these visualisations to identify patterns and trends and to detect outliers and anomalies.

For further findings and explanations, please refer to the Jupyter Notebook on EDA.

After performing EDA, we prepared the dataset by the following steps:

1. **Imputed the missing values**. We used the best imputation method that yielded the most accurate results, based on the comparison of six separate data frames. The six data frames consisted of three with different imputation methods (zero imputation, mean imputation, and median imputation) and with or without outliers. 
2. **Removed outliers**. The data frames without outliers were removed using interquartile range (IQR) method. This is important as outliers can skew the distribution of the data and affect the accuracy of the model.

We want to chose the best imputation method to ensure that the dataset has minimal missing values and accurate data, which helps improve the accuracy of the model. Removing outliers helps ensure that the distribution of the data is normal and that the model is not affected by extreme values.

To find the best imputation method, it is explained in [Methodology].

We want to chose the best imputation method to ensure that the dataset has minimal missing values and accurate data, which helps improve the accuracy of the model. Removing outliers helps ensure that the distribution of the data is normal and that the model is not affected by extreme values.

## 3. [Methodology]

To find the best imputation method, we ran the six separate dataframes on each of the four machine learning models. We compared the accuracy of each data frame within each model by measuring metrics such as precision, recall and F1 score. How these metrics are chosen are explained later in [Experiment].

### The four machine learning models used to classify heart disease are as follows: 
1. Decision Tree 
2. Random Forest
3. Logistic Regression
4. Support Vector Machine (SVM).

### Baseline Model - Decision Tree model
The Decision Tree model is a simple yet powerful algorithm that uses a tree-like structure to classify data based on a set of rules. It is often used as a baseline model for classification tasks. However, we found that the Decision Tree model yielded subpar results for our dataset, indicating that a more complex model was needed.

### Random Forest model
The Random Forest model is an ensemble learning algorithm that combines multiple decision trees to improve accuracy and generalization by reducing variance and preventing overfitting. It works by creating a random subset of features for each tree and aggregating their predictions. We chose this model to improve the performance of our baseline model and prevent overfitting.

We used GridSearchCV to tune the following hyperparameters for the Random Forest model:

* n_estimators: the number of trees in the forest
* max_depth: the maximum depth of each tree

We trained the Random Forest model using GridSearchCV with 5-fold cross-validation and selected the best hyperparameters based on the F1 score.

### Logistic Regression model
Logistic Regression is a linear model that predicts the probability of an event occurring based on a set of input features. It can achieve comparable or better performance compared to tree models, especially when the number of features is small or when the relationships between features and outcomes are linear or additive. We chose this model to compare its performance with the tree-based models and evaluate its ability to identify significant predictors of heart disease.

We used GridSearchCV to tune the following hyperparameters for the Logistic Regression model:

* C: the inverse of regularization strength
* solver: the optimization algorithm used to fit the model
We trained the Logistic Regression model using GridSearchCV with 5-fold cross-validation and selected the best hyperparameters based on the F1 score.

### Support Vector Machine (SVM) model
SVM is a powerful algorithm that can transform the data into higher-dimensional spaces to identify complex decision boundaries, making it useful for feature selection in classification problems with non-linear relationships. It can identify the most significant features that separate the data into different classes, making it a useful tool for feature selection in identifying predictors of heart disease. We chose this model to evaluate its ability to identify the most important predictors of heart disease.

We used GridSearchCV to tune the following hyperparameters for the SVM model:

* kernel: the kernel function used to transform the data into higher-dimensional spaces
* C: the inverse of regularization strength
* gamma: the kernel coefficient for 'rbf', 'poly' and 'sigmoid'
We trained the SVM model using GridSearchCV with 5-fold cross-validation and selected the best hyperparameters based on the F1 score.

### Feature pruning
After finding what is the best imputation method for that particular model.
To find the best features and model in predicting heart dieases 

We performed the following:
1. Find the best feature within each model (with the best imputation method) via feature pruning which involves selecting the most relevant features that contribute to the accuracy of the model
2. Create the best model using the best features.
More of this is explained in [Methodology].

## 4. [Experiment]

### Metrics used to evaluate the accuracy of the Model
To evaluate the performance of our model in predicting the presence or absence of a 'HeartDisease' response variable, We generated a correlation report and the confusion matrix:

| Confusion Matrix  |       |        |        |      
| :---              | :---: | :----: | :----: |         
| Actual Negative   |  (0)  |   TN   |   FP   |             
| Actual Positive   |  (1)  |   FN   |   TP   |       
|                   |       |   (0)   |   (1)   |       
|                   |       | Predicted Negative    |   Predicted Postitive  |     

The confusion matrix is a table that summarises the predictions made by the model, where the rows represent the actual classes, and the columns represent the predicted classes.

The correlation report generates the following metrics: precision, recall, and F1 score. We will be using the macro average since the classes are not imbalanced as shown in the EDA.

Performance Metrics:

* **Precision**: Measures the correctness of positive predictions. In the context of this model, precision refers to the model's ability to predict that a patient has 'HeartDisease' (positive prediction) and to make this prediction accurately. For example, if the model predicts that a patient has 'HeartDisease', how often is it correct?

* **Recall**: Measures the identification of positive instances. In the context of this model, positive instances refer to patients who have 'HeartDisease' that the model is trying to predict. For example, if a patient has 'HeartDisease', how often does the model correctly identify them as having 'HeartDisease'?

* **F1 score**: Balances precision and recall. F1 score is the harmonic mean of precision and recall. It provides a single score that takes into account both precision and recall. A high F1 score indicates that the model has high precision and high recall.

The choice of metric to use depends on the specific context and problem. If the cost of false positives and false negatives is similar, accuracy is a reasonable metric. If the cost of false positives is higher, precision is more appropriate. If the cost of false negatives is higher, recall is better. Finally, if both costs are significant, the F1 score is a good choice as it balances the two metrics.

In the case of predicting heart disease, false negatives' cost might be higher, as a patient predicted to not have 'HeartDisease' could be worst off than a patient predicted to have 'HeartDisease'. Similarly, false postives' cost might aslo be high as a patient predicted to have 'HeartDisease' could take up valuable resources that could be used on patients that actually requires it. Therefore, the **F1 score might be the most reasonable metric as it balances both precision and recall**.

### Detailed model analysis and baseline

....

### 5. [Conclusion and Data Driven Insights]

....
 

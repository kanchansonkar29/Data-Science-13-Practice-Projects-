#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
#     </a>
# </p>
# 
# <h1 align="center"><font size="5">Final Project: Classification with Python</font></h1>
# 

# <h2>Table of Contents</h2>
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     <ul>
#     <li><a href="https://#Section_1">Instructions</a></li>
#     <li><a href="https://#Section_2">About the Data</a></li>
#     <li><a href="https://#Section_3">Importing Data </a></li>
#     <li><a href="https://#Section_4">Data Preprocessing</a> </li>
#     <li><a href="https://#Section_5">One Hot Encoding </a></li>
#     <li><a href="https://#Section_6">Train and Test Data Split </a></li>
#     <li><a href="https://#Section_7">Train Logistic Regression, KNN, Decision Tree, SVM, and Linear Regression models and return their appropriate accuracy scores</a></li>
# </a></li>
# </div>
# <p>Estimated Time Needed: <strong>180 min</strong></p>
# </div>
# 
# <hr>
# 

# # Instructions
# 

# In this notebook, you will  practice all the classification algorithms that we have learned in this course.
# 
# 
# Below, is where we are going to use the classification algorithms to create a model based on our training data and evaluate our testing data using evaluation metrics learned in the course.
# 
# We will use some of the algorithms taught in the course, specifically:
# 
# 1. Linear Regression
# 2. KNN
# 3. Decision Trees
# 4. Logistic Regression
# 5. SVM
# 
# We will evaluate our models using:
# 
# 1.  Accuracy Score
# 2.  Jaccard Index
# 3.  F1-Score
# 4.  LogLoss
# 5.  Mean Absolute Error
# 6.  Mean Squared Error
# 7.  R2-Score
# 
# Finally, you will use your models to generate the report at the end. 
# 

# # About The Dataset
# 

# The original source of the data is Australian Government's Bureau of Meteorology and the latest data can be gathered from [http://www.bom.gov.au/climate/dwo/](http://www.bom.gov.au/climate/dwo/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01).
# 
# The dataset to be used has extra columns like 'RainToday' and our target is 'RainTomorrow', which was gathered from the Rattle at [https://bitbucket.org/kayontoga/rattle/src/master/data/weatherAUS.RData](https://bitbucket.org/kayontoga/rattle/src/master/data/weatherAUS.RData?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01)
# 
# 
# 

# This dataset contains observations of weather metrics for each day from 2008 to 2017. The **weatherAUS.csv** dataset includes the following fields:
# 
# | Field         | Description                                           | Unit            | Type   |
# | ------------- | ----------------------------------------------------- | --------------- | ------ |
# | Date          | Date of the Observation in YYYY-MM-DD                 | Date            | object |
# | Location      | Location of the Observation                           | Location        | object |
# | MinTemp       | Minimum temperature                                   | Celsius         | float  |
# | MaxTemp       | Maximum temperature                                   | Celsius         | float  |
# | Rainfall      | Amount of rainfall                                    | Millimeters     | float  |
# | Evaporation   | Amount of evaporation                                 | Millimeters     | float  |
# | Sunshine      | Amount of bright sunshine                             | hours           | float  |
# | WindGustDir   | Direction of the strongest gust                       | Compass Points  | object |
# | WindGustSpeed | Speed of the strongest gust                           | Kilometers/Hour | object |
# | WindDir9am    | Wind direction averaged of 10 minutes prior to 9am    | Compass Points  | object |
# | WindDir3pm    | Wind direction averaged of 10 minutes prior to 3pm    | Compass Points  | object |
# | WindSpeed9am  | Wind speed averaged of 10 minutes prior to 9am        | Kilometers/Hour | float  |
# | WindSpeed3pm  | Wind speed averaged of 10 minutes prior to 3pm        | Kilometers/Hour | float  |
# | Humidity9am   | Humidity at 9am                                       | Percent         | float  |
# | Humidity3pm   | Humidity at 3pm                                       | Percent         | float  |
# | Pressure9am   | Atmospheric pressure reduced to mean sea level at 9am | Hectopascal     | float  |
# | Pressure3pm   | Atmospheric pressure reduced to mean sea level at 3pm | Hectopascal     | float  |
# | Cloud9am      | Fraction of the sky obscured by cloud at 9am          | Eights          | float  |
# | Cloud3pm      | Fraction of the sky obscured by cloud at 3pm          | Eights          | float  |
# | Temp9am       | Temperature at 9am                                    | Celsius         | float  |
# | Temp3pm       | Temperature at 3pm                                    | Celsius         | float  |
# | RainToday     | If there was rain today                               | Yes/No          | object |
# | RainTomorrow  | If there is rain tomorrow                             | Yes/No          | float  |
# 
# Column definitions were gathered from [http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml](http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01)
# 
# 

# ## **Import the required libraries**
# 

# In[ ]:


# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented.
# !mamba install -qy pandas==1.3.4 numpy==1.21.4 seaborn==0.9.0 matplotlib==3.5.0 scikit-learn==0.20.1
# Note: If your environment doesn't support "!mamba install", use "!pip install"


# In[ ]:


# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[ ]:


#you are running the lab in your  browser, so we will install the libraries using ``piplite``
import piplite
await piplite.install(['pandas'])
await piplite.install(['numpy'])


# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics


# ### Importing the Dataset
# 

# In[ ]:


from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())


# In[ ]:


path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv'


# In[ ]:


await download(path, "Weather_Data.csv")
filename ="Weather_Data.csv"


# In[ ]:


df = pd.read_csv("Weather_Data.csv")
df.head()


# ### Data Preprocessing
# 

# #### One Hot Encoding
# 

# First, we need to perform one hot encoding to convert categorical variables to binary variables.
# 

# In[ ]:


df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])


# Next, we replace the values of the 'RainTomorrow' column changing them from a categorical column to a binary column. We do not use the `get_dummies` method because we would end up with two columns for 'RainTomorrow' and we do not want, since 'RainTomorrow' is our target.
# 

# In[ ]:


df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)


# ### Training Data and Test Data
# 

# Now, we set our 'features' or x values and our Y or target variable.
# 

# In[ ]:


df_sydney_processed.drop('Date',axis=1,inplace=True)


# In[ ]:


df_sydney_processed = df_sydney_processed.astype(float)


# In[ ]:


features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']


# ### Linear Regression
# 

# #### Q1) Use the `train_test_split` function to split the `features` and `Y` dataframes with a `test_size` of `0.2` and the `random_state` set to `10`.
# 

# In[ ]:


#Enter Your Code, Execute and take the Screenshot


# In[ ]:


x_train, x_test, y_train, y_test = 


# #### Q2) Create and train a Linear Regression model called LinearReg using the training data (`x_train`, `y_train`).
# 

# In[ ]:


#Enter Your Code, Execute and take the Screenshot


# In[ ]:


LinearReg = 


# #### Q3) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.
# 

# In[ ]:


#Enter Your Code, Execute and take the Screenshot


# In[ ]:


predictions = 


# #### Q4) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function.
# 

# In[ ]:


#Enter Your Code, Execute and take the Screenshot


# In[ ]:


LinearRegression_MAE = 
LinearRegression_MSE = 
LinearRegression_R2 = 


# #### Q5) Show the MAE, MSE, and R2 in a tabular format using data frame for the linear model.
# 

# In[ ]:


#Enter Your Code, Execute and take the Screenshot


# In[ ]:


Report = 


# ### KNN
# 

# #### Q6) Create and train a KNN model called KNN using the training data (`x_train`, `y_train`) with the `n_neighbors` parameter set to `4`.
# 

# In[ ]:


#Enter Your Code Below, Execute, and Save the Screenshot of the Final Output


# In[ ]:


KNN = 


# #### Q7) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.
# 

# In[ ]:


#Enter Your Code Below, Execute, and Save the Screenshot of the Final Output


# In[ ]:


predictions = 


# #### Q8) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function.
# 

# In[ ]:


#Enter Your Code Below, Execute, and Save the Screenshot of the Final Output


# In[ ]:


KNN_Accuracy_Score = 
KNN_JaccardIndex = 
KNN_F1_Score = 


# ### Decision Tree
# 

# #### Q9) Create and train a Decision Tree model called Tree using the training data (`x_train`, `y_train`).
# 

# In[ ]:


#Enter Your Code, Execute and take the Screenshot


# In[ ]:


Tree = 


# #### Q10) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.
# 

# In[ ]:


#Enter Your Code, Execute and take the Screenshot


# In[ ]:


predictions = 


# #### Q11) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function.
# 

# In[ ]:


#Enter Your Code, Execute and take the Screenshot


# In[ ]:


Tree_Accuracy_Score = 
Tree_JaccardIndex = 
Tree_F1_Score = 


# ### Logistic Regression
# 

# #### Q12) Use the `train_test_split` function to split the `features` and `Y` dataframes with a `test_size` of `0.2` and the `random_state` set to `1`.
# 

# In[ ]:


#Enter Your Code, Execute and take the Screenshot


# In[ ]:


x_train, x_test, y_train, y_test = 


# #### Q13) Create and train a LogisticRegression model called LR using the training data (`x_train`, `y_train`) with the `solver` parameter set to `liblinear`.
# 

# In[ ]:


#Enter Your Code, Execute and take the Screenshot


# In[ ]:


LR = 


# #### Q14) Now, use the `predict` and `predict_proba` methods on the testing data (`x_test`) and save it as 2 arrays `predictions` and `predict_proba`.
# 

# In[ ]:


#Enter Your Code, Execute and take the Screenshot


# In[ ]:


predictions = 


# In[ ]:


predict_proba = 


# #### Q15) Using the `predictions`, `predict_proba` and the `y_test` dataframe calculate the value for each metric using the appropriate function.
# 

# In[ ]:


#Enter Your Code, Execute and take the Screenshot


# In[ ]:


LR_Accuracy_Score = 
LR_JaccardIndex = 
LR_F1_Score = 
LR_Log_Loss = 


# ### SVM
# 

# #### Q16) Create and train a SVM model called SVM using the training data (`x_train`, `y_train`).
# 

# In[ ]:


#Enter Your Code Below, Execute, and Save the Screenshot of the Final Output


# In[ ]:


SVM = 


# #### Q17) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.
# 

# In[ ]:


#Enter Your Code Below, Execute, and Save the Screenshot of the Final Output


# In[ ]:


predictions = 


# #### Q18) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function.
# 

# In[ ]:


SVM_Accuracy_Score = 
SVM_JaccardIndex = 
SVM_F1_Score = 


# ### Report
# 

# #### Q19) Show the Accuracy,Jaccard Index,F1-Score and LogLoss in a tabular format using data frame for all of the above models.
# 
# \*LogLoss is only for Logistic Regression Model
# 

# In[ ]:


Report = 


# <h2 id="Section_5">  How to submit </h2>
# 
# <p>Once you complete your notebook you will have to share it. You can download the notebook by navigating to "File" and clicking on "Download" button.
# 
# <p>This will save the (.ipynb) file on your computer. Once saved, you can upload this file in the "My Submission" tab, of the "Peer-graded Assignment" section.  
# 

# <h2>About the Authors:</h2> 
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
# 
# ### Other Contributors
# 
# [Svitlana Kramar](https://www.linkedin.com/in/svitlana-kramar/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2022-01-01)
# 

# ## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By    | Change Description          |
# | ----------------- | ------- | ------------- | --------------------------- |
# | 2022-06-22        | 2.0     | Svitlana K.   | Deleted GridSearch and Mock |
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 

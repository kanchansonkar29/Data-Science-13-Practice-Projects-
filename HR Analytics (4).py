#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas.plotting import scatter_matrix

import sklearn as sk
import sklearn.tree as tree
from IPython.display import Image 
#import pydotplus


# In[2]:


from scipy.stats import zscore, boxcox
import warnings
warnings.filterwarnings('ignore')


# ### Read And Display The Data

# In[3]:


df = pd.read_csv("C:/Users/kanch/Hr Analysis/WA_Fn-UseC_-HR-Employee-Attrition.csv")
#pd.set_option('display.max_columns', None)
df.head(7)


# In[4]:


from sklearn.preprocessing import LabelEncoder


# In[5]:


lab_enc = LabelEncoder()


# In[6]:


var_cat = df.select_dtypes(include=[object])
var_cat.head()


# In[7]:


#var_cat = var_cat.columns.tolist()
var_cat = ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']

var_cat


# In[8]:


for i in var_cat:
    df[i] = lab_enc.fit_transform(df[i])

df.head()


# In[9]:


df1 = lab_enc.fit_transform(df['BusinessTravel'])


# In[10]:


pd.Series(df1)


# In[11]:


df['BusinessTravel'] = df1


# In[12]:


df.head()


# Eliminate columns that only have one data level

# In[13]:


temp = []
for col in df.columns:
    if len(df[col].unique()) == 1:
        temp.append(col)
        df.drop(col,inplace=True,axis=1)


# In[14]:


df.shape


# In[15]:


temp


# In[16]:


df.describe()


# In[17]:


df.columns


# In[18]:


df.drop(['EmployeeNumber'], axis = 1, inplace = True)


# In[19]:


f, ax = plt.subplots(figsize=(35,25))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 15, as_cmap=True),
            square=True, ax=ax, annot = True)


# From the correlation table we see that monthly income is highly correlated with job level. However, daily rate, hourly rate and monthly rate are barely correlated with anything. 
# We will be using monthly income in later analysis as a measurement of salary and get rid of other income related variables.

# In[20]:


df.drop(['DailyRate', 'MonthlyRate'], axis = 1, inplace = True)


# In[21]:


df['OverTime'] = df.OverTime == 'Yes'


# In[22]:


df['Attrition'] = df.Attrition == 'Yes'


# ### Exploratory Data Visualization

# In[23]:


sns.countplot(x='Attrition', data=df)


# This section is providing a general idea about attrition in the data set by looking at gender, marital status and department.

# Male is generally more likely to quit than female.

# In[24]:


sns.barplot(x = 'MaritalStatus', y='Attrition', data=df)


# Single people are more likely to quit compared to married and divorced people.

# In[25]:


# We Check the monthly income by age in the age
sns.factorplot(x = 'Age', y='MonthlyIncome', kind = 'bar', data=df, aspect = 3)


# In[26]:


sns.factorplot(x = 'Age', y='Attrition', kind = 'bar', data=df, aspect = 3)


# Younger people are having higher attrition rate compared to older people.

# In[27]:


sns.factorplot(x = 'TotalWorkingYears', y='MonthlyIncome', kind = 'bar', data=df, aspect = 3)


# In[28]:


sns.factorplot(x = 'JobRole', y='MonthlyIncome', kind = 'bar', data=df, aspect = 3.5)


# In[29]:


sns.factorplot(x = 'JobRole', y='TotalWorkingYears', kind = 'bar', data=df, aspect = 3.5)


# In[30]:


sns.factorplot(x = 'JobRole', y='YearsAtCompany', kind = 'bar', data=df, aspect = 3.5)


# In[31]:


sns.factorplot(x = 'JobRole', y="Attrition", kind = 'bar', data=df, aspect = 3.5)


# Data Transformation and derivation of new attributes if necessary

# In[32]:


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score,confusion_matrix, roc_curve,roc_auc_score
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[33]:


data = df.copy()


# In[34]:


from sklearn.metrics import classification_report
#print(classification_report(y_test,y_pred))


# Create dummies and interaction terms

# In[35]:


data.describe()


# In[36]:


data.dtypes


# In[37]:


data=data.drop(['BusinessTravel','JobRole'],axis=1)


# In[38]:


data=data.drop(['Department','EducationField','MaritalStatus'],axis=1)


# In[39]:


data['Attrition'] = data['Attrition'].astype('int')
data['Gender'] = data['Gender'].astype('int')
data['OverTime'] = data['OverTime'].astype('int')


# In[40]:


data.dtypes


# In[41]:


data.shape


# In[42]:


#Let's see how data is distributed for every column
plt.figure(figsize = (20,50) , facecolor='red')
plotnumber=1

for columns in data:
    if plotnumber <=24:
        ax = plt.subplot(8,3,plotnumber)
        sns.distplot(data[columns])
        plt.xlabel(columns,fontsize=20)
    plotnumber+=1
    
plt.tight_layout()


# In[43]:


df_feature = data.drop('JobLevel', axis = 1)
data.shape


# Visualize the Outliers Using Boxplot

# In[44]:


#visualize the outliers using boxplot
plt.figure(figsize =(20,50))
graph = 1

for column in df_feature:
    if graph<=24:
        plt.subplot(8,3,graph)
        ax = sns.boxplot(data=df_feature[column])
        plt.xlabel(column,fontsize=15)
    graph+=1
plt.show()


# In[45]:


data.shape


# In[46]:


# find the IQR to identify outliers

#1st Quantile

Q1 = data.quantile(0.25)

# 3rd Quantile

Q3 = data.quantile(0.78)

#IQR ( Inter Quantile Range)

IQR = Q3-Q1


# In[47]:


#validating one outlier
preg_high = (Q3.YearsAtCompany + (1.5*IQR.YearsAtCompany))
preg_high


# In[48]:


# Check the indexes whcih have higher values
index = np.where(data['YearsAtCompany']>preg_high)
index


# In[49]:


data= data.drop(data.index[index])
data.shape


# In[50]:


data.reset_index()


# In[51]:


#validating one outlier
preg_high = (Q3.TotalWorkingYears + (1.5*IQR.TotalWorkingYears))
preg_high


# In[52]:


# Check the indexes whcih have higher values
index = np.where(data['TotalWorkingYears']>preg_high)
index


# In[53]:


data= data.drop(data.index[index])
data.shape


# In[54]:


data.reset_index()


# In[55]:


preg_high = (Q3.YearsInCurrentRole + (1.5*IQR.YearsInCurrentRole))
preg_high


# In[56]:


# Check the indexes whcih have higher values
index = np.where(data['YearsInCurrentRole']>preg_high)
index


# In[57]:


data= data.drop(data.index[index])
data.shape


# In[58]:


data.reset_index()


# In[59]:


#validating one outlier
preg_high = (Q3.MonthlyIncome + (1.5*IQR.MonthlyIncome))
preg_high


# In[60]:


# Check the indexes whcih have higher values
index = np.where(data['MonthlyIncome']>preg_high)
index


# In[61]:


data= data.drop(data.index[index])
data.shape


# In[62]:


data.reset_index()


# In[63]:


#validating one outlier
preg_high = (Q3.YearsSinceLastPromotion + (1.5*IQR.YearsSinceLastPromotion))
preg_high


# In[64]:


# Check the indexes whcih have higher values
index = np.where(data['YearsSinceLastPromotion']>preg_high)
index


# In[65]:


data= data.drop(data.index[index])
data.shape


# In[66]:


data.reset_index()


# In[67]:


#Let's see how data is distributed for every column
plt.figure(figsize = (20,50) , facecolor='red')
plotnumber=1

for columns in data:
    if plotnumber <=24:
        ax = plt.subplot(8,3,plotnumber)
        sns.distplot(data[columns])
        plt.xlabel(columns,fontsize=20)
    plotnumber+=1
    
plt.tight_layout()


# Visualize the Outliers After removing Some Outliers

# In[68]:


#visualize the outliers using boxplot
plt.figure(figsize =(20,50))
graph = 1

for column in df_feature:
    if graph<=24:
        plt.subplot(8,3,graph)
        ax = sns.boxplot(data=df_feature[column])
        plt.xlabel(column,fontsize=15)
    graph+=1
plt.show()


# Data Standarization and Normalization

# In[69]:


#Finding relationship we want to keep only those feature which have relationship with labels
X = data.drop(columns = ["Attrition"])
y = data["Attrition"]


# In[70]:


plt.figure(figsize =(20,50))
graph = 1

for column in X:
    if graph<=24:
        ax =plt.subplot(8,3,graph)
        sns.stripplot(y,X[column])
        #plt.xlabel(column,fontsize=15)
    graph+=1
plt.show()


# In[71]:


# Check multicollinearity problem Find if one feature is dependent on another feature
scalar = StandardScaler()
X_scalar = scalar.fit_transform(X)


# In[72]:


x_train,x_test,y_train,y_test=train_test_split(X_scalar,y, test_size = 0.25, random_state = 355)


# In[73]:


def metric_score (clf, x_train,x_test,y_train,y_test, train = True):
    if train:
        y_pred = clf.predict(x_train)
        print("\n =================Train Result=====================")
        print(f"Accuracy Score : {accuracy_score(y_train,y_pred)*100:.2f}%")
    elif train == False:
        pred = clf.predict(x_test)
        print("\n==================Test Result=======================")
        print(f"Accuracy Score : {accuracy_score(y_test,pred)*100:.2f}%")
        
        print("\n \n Test Classification Report\n",classification_report(y_test,pred,digits=2))


# In[74]:


from sklearn.neighbors import KNeighborsClassifier


# In[75]:


# Model Confidence / Accuracy
#Initiate KNeighborsClassifier
knn = KNeighborsClassifier()


# In[76]:


# Model Training
knn.fit(x_train,y_train)


# In[77]:


#Call The Function
metric_score(knn,x_train,x_test,y_train,y_test, train = True)
metric_score(knn,x_train,x_test,y_train,y_test, train = False)


# In[78]:


from sklearn.model_selection import KFold, cross_val_score


# In[79]:


k_f = KFold(n_splits = 3)


# In[80]:


k_f


# In[81]:


#for train, test in k_f.split([1,2,3,4,5,6,7,8,9]):
  #  print("Train : ",train,'Test : ',test)


# In[82]:


#Cross Validation Score to check if the model is overfitting 
cross_val_score(knn, X_scalar,y,cv = 5)
cross_val_score(knn,X_scalar,y,cv = 5).mean()


# ##### using GridSearchCV for the bet parameter to improve the accuracy

# In[83]:


from sklearn.model_selection import GridSearchCV


# In[84]:


param_grid = {'algorithm' : ['kd_tree','brute'],
             'leaf_size' : [3,5,6,7,8],
             'n_neighbors' : [3,5,7,9,11,13]
             }


# In[85]:


gridsearch = GridSearchCV(estimator = knn,param_grid = param_grid)


# In[86]:


gridsearch.fit(x_train,y_train)


# In[87]:


gridsearch.best_params_


# In[88]:


# We will use the parameters in our K-NN algorithm and check if accuracy is increasing.
knn = KNeighborsClassifier(algorithm = 'kd_tree',leaf_size =3,n_neighbors = 7)


# In[89]:


knn.fit(x_train,y_train)


# In[90]:


# call the function and pass dataset to check train and test score
# This is for Training Score
metric_score(knn,x_train,x_test,y_train,y_test, train = True)
#This is for Testing Score
metric_score(knn,x_train,x_test,y_train,y_test, train = False)


# In[91]:


# If we want to check the confustion_matrix we can check
y_pred = knn.predict(x_test)
cfm = confusion_matrix (y_test,y_pred)
cfm


# In[92]:


X_scalar.shape[0]


# In[93]:


#Finding Variance inflation factor in each scaled columns
vif = pd.DataFrame()
vif["vif"]=[variance_inflation_factor(X_scalar,i) for i in range (X_scalar.shape[1])]
vif["features"]=X.columns
vif


# Creation of Train and Test Dataset Using Optimum Parameters

# In[94]:


#Now Split our data in test and training set
x_train,x_test,y_train,y_test=train_test_split(X_scalar,y, test_size = 0.25, random_state = 355)


# In[95]:


log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)


# In[96]:


#Let's see how well our model perform on the test data set
x_test


# In[97]:


y_pred = log_reg.predict(x_test)


# In[98]:


y_pred


# In[99]:


#Model Accuracy
accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[100]:


# Confusion Matrix
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[101]:


# Now Calculate recall, Precision, F1 Score
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[102]:


# ROC curve
fpr,tpr, thresholds = roc_curve(y_test,y_pred)
# Threshold [0] means no innstances predicted, it should be read from 0 to max
print('Threshold = ',thresholds)
print('True Positive Rate = ', tpr)
print("False Positive Rate = ", fpr)


# In[103]:


plt.plot(fpr,tpr, color = 'orange', label = 'ROC')
plt.plot([0,1],[0,1], color = 'darkblue', linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receive Operating Characteristic Curve')
plt.legend()
plt.show()


# In[104]:


# How much area it is covering (AUC) 
auc_score=roc_auc_score(y_test,y_pred)
print(auc_score)


# In[105]:


from sklearn.linear_model import LinearRegression


# In[106]:


regression = LinearRegression()
regression.fit(x_train,y_train)


# In[107]:


# Let's check how well models fit on train data
# Adjust r2 Score
regression.score(x_train,y_train)


# In[108]:


# Let's check how well models fit the last data 
regression.score(x_test,y_test)


# Let's Plot and Visualize

# In[109]:


y_pred = regression.predict(x_test)
y_pred


# In[110]:


plt.scatter(y_test,y_pred)
plt.xlabel('Actual Attrition of Data')
plt.ylabel('Predicted Attrition of Data')
plt.title("Actual VS Model Predicted")
plt.show()


# In[111]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
y_pred = regression.predict(x_test)


# In[112]:


# MAE (Mean Absolute Error)
mean_absolute_error(y_test,y_pred)


# In[113]:


# MSE (Mean Squared Error)
mean_squared_error(y_test,y_pred)


# In[114]:


# RMSE (Root Mean Squared Error)
np.sqrt(mean_squared_error(y_test,y_pred))


# In[115]:


from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV


# In[116]:


#LassoCV will return best alpha after max iteration 
#Normalize is subtracting the mean and dividing by the L2_norm

lassocv = LassoCV(alphas = None, max_iter =100,normalize= True)
lassocv.fit(x_train,y_train)


# In[117]:


# Best Alpha Parameter
alpha = lassocv.alpha_
alpha


# In[118]:


# Now that we have best parameter, Let's use Lasso regression and see how well our data has fitted before
lasso_reg = Lasso(alpha)
lasso_reg.fit(x_train,y_train)


# In[119]:


lasso_reg.score(x_test,y_test)


# In[120]:


# Ridge will return best alpha and coefficient after performing 10 cross validation

ridgecv= RidgeCV(alphas= np.arange(0.001,0.1,0.01), normalize = True)
ridgecv.fit(x_train,y_train)


# In[121]:


ridgecv.alpha_


# In[122]:


ridge_model = Ridge(alpha = ridgecv.alpha_)
ridge_model.fit(x_train,y_train)


# In[123]:


ridge_model.score(x_test,y_test)


# In[124]:


data.shape


# In[125]:


data.describe()


# In[126]:


data.drop_duplicates()


# In[127]:


data.shape


# In[128]:


#Let's see how data is distributed for every column
plt.figure(figsize = (20,50) , facecolor='red')
plotnumber=1

for columns in data:
    if plotnumber <=24:
        ax = plt.subplot(8,3,plotnumber)
        sns.distplot(data[columns])
        plt.xlabel(columns,fontsize=20)
    plotnumber+=1
    
plt.tight_layout()


# In[129]:


data.columns


# In[130]:


z_score = zscore(data[['Age','DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'Gender','JobInvolvement',
       'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
       'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']])
abs_z_score = np.abs(z_score)
filtering_entry= (abs_z_score<3).all(axis=1)
data=data[filtering_entry]


# In[131]:


data.describe()


# In[132]:


data.head()


# In[133]:


#Let's see how data is distributed for every column
plt.figure(figsize = (20,50) , facecolor='red')
plotnumber=1

for columns in data[['Age','DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'Gender','JobInvolvement',
       'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
       'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']]:
    if plotnumber <=24:
        ax = plt.subplot(8,3,plotnumber)
        sns.distplot(data[columns])
        plt.xlabel(columns,fontsize=20)
    plotnumber+=1
    
plt.tight_layout()


# In[134]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV as rsv


# In[135]:


#Now Split our data in test and training set
x_train,x_test,y_train,y_test=train_test_split(X,y, test_size = 0.30, random_state = 41)


# In[136]:


random_clf=RandomForestClassifier()


# In[137]:


random_clf.fit(x_train,y_train)


# In[138]:


# This is for Training Score
metric_score(random_clf,x_train,x_test,y_train,y_test, train = True)
#This is for Testing Score
metric_score(random_clf,x_train,x_test,y_train,y_test, train = False)


# In[139]:


params={"n_estimators":[200,400],'max_depth':[6,9],'criterion':('gini','entropy')}
grd = GridSearchCV(random_clf,param_grid=params)
grd.fit(x_train,y_train)
print("best_params => ",grd.best_params_)


# In[140]:


random_clf=grd.best_estimator_
random_clf.fit(x_train,y_train)


# In[141]:


# This is for Training Score
metric_score(random_clf,x_train,x_test,y_train,y_test, train = True)
#This is for Testing Score
metric_score(random_clf,x_train,x_test,y_train,y_test, train = False)


# In[142]:


#Plot ROC/AUC for multiple models without hyperparams tuning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import plot_roc_curve


# In[143]:


lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
kn = KNeighborsClassifier()


# In[144]:


# training with all classifiers
lr.fit(x_train,y_train)
rf.fit(x_train,y_train)
kn.fit(x_train,y_train)
dt.fit(x_train,y_train)
print("All Models Are Trained")


# In[145]:


#All Models Score Are Captured
lr.score(x_test,y_test)
kn.score(x_test,y_test)
dt.score(x_test,y_test)
rf.score(x_test,y_test)
print("All models test score are captured")


# In[146]:


# How well our models works onn training data 
disp = plot_roc_curve(dt,x_train,y_train)
plot_roc_curve(lr,x_train,y_train,ax=disp.ax_)
plot_roc_curve(kn,x_train,y_train,ax=disp.ax_)
plot_roc_curve(rf,x_train,y_train,ax=disp.ax_)
plt.legend(prop={'size':10},loc='lower right')
plt.show()


# In[147]:


# How well our models works onn test data 
disp = plot_roc_curve(dt,x_test,y_test)
plot_roc_curve(lr,x_test,y_test,ax=disp.ax_)
plot_roc_curve(kn,x_test,y_test,ax=disp.ax_)
plot_roc_curve(rf,x_test,y_test,ax=disp.ax_)
plt.legend(prop={'size':10},loc='lower right')
plt.show()


# In[148]:


rfc=RandomForestClassifier()
rfc_para={"n_estimators":[200,400],'max_depth':[6,9],'criterion':('gini','entropy')}
rfc_rsv=rsv(rfc,rfc_para,cv=30)
rfc_rsv.fit(x_train,y_train)
print(rfc_rsv)
print('\nbest score=',rfc_rsv.best_score_)
print('\nbest parameters for RFC=\n',rfc_rsv.best_params_)


# In[149]:


rfc=rfc_rsv.best_estimator_
rfc.fit(x_train,y_train)


# In[150]:


# This is for Training Score
metric_score(random_clf,x_train,x_test,y_train,y_test, train = True)
#This is for Testing Score
metric_score(random_clf,x_train,x_test,y_train,y_test, train = False)


# In[151]:


#rfc=RandomForestClassifier()"
params={"n_estimators":[200,400],'max_depth':[6,9],'criterion':('gini','entropy')}
#rfc_rsv=rsv(rfc,rfc_para,cv=30)
#rfc_rsv.fit(x_train,y_train)
grd = GridSearchCV(random_clf,param_grid=params)
grd.fit(x_train,y_train)
print("best_params => ",grd.best_params_)
#print(rfc_rsv)
#print('\nbest score=',rfc_rsv.best_score_)
#print('\nbest parameters for RFC=\n',rfc_rsv.best_params_)


# In[152]:


random_clf=grd.best_estimator_
random_clf.fit(x_train,y_train)


# In[153]:


# This is for Training Score
metric_score(random_clf,x_train,x_test,y_train,y_test, train = True)
#This is for Testing Score
metric_score(random_clf,x_train,x_test,y_train,y_test, train = False)


# In[ ]:





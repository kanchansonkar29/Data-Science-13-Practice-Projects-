#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import mode
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import zscore, boxcox
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.preprocessing import LabelEncoder
sns.set_context('paper')
from sklearn.preprocessing import OneHotEncoder


# In[3]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn import metrics
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge


# In[4]:


test = pd.read_csv(r"C:\Users\talkt\OneDrive\Desktop\BIG MART SALES\Test.csv")
train = pd.read_csv(r"C:\Users\talkt\OneDrive\Desktop\BIG MART SALES\Train.csv")


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


test.tail()


# In[8]:


train.tail()


# In[9]:


train.describe()


# In[10]:


test.describe()


# In[11]:


print(train.shape)
print(test.shape)


# In[12]:


train.mean()


# ## Data Cleaning

# In[13]:


train.Item_Fat_Content.value_counts()


# In[14]:


train.Item_Visibility.value_counts()


# In[15]:


train.Outlet_Size.value_counts()


# In[16]:


test['Item_Outlet_Sales'] = 1
test.Item_Outlet_Sales.head()


# In[17]:


combine = pd.concat([train,test])
combine.head()


# In[18]:


crosstable = pd.crosstab(train['Outlet_Size'],train['Outlet_Type'])
crosstable


# In[19]:


train.Outlet_Size.value_counts()


# In[20]:


dic = {'Grocery Store':'Small'}
s = train.Outlet_Type.map(dic)


# In[21]:


train.Outlet_Size= train.Outlet_Size.combine_first(s)
train.Outlet_Size.value_counts()


# In[22]:


train.isnull().sum(axis=0)


# In[23]:


#checking for location type
crosstable = pd.crosstab(train.Outlet_Size,train.Outlet_Location_Type)
crosstable


# #### From the above table it is evident that all the Tier 2 stores are of small types

# ##### Therefore mapping Tier 2 store and small size

# In[24]:


dic = {"Tier 2":"Small"}
s = train.Outlet_Location_Type.map(dic)
train.Outlet_Size = train.Outlet_Size.combine_first(s)
train.Outlet_Size.value_counts()


# In[25]:


train.isnull().sum(axis=0)


# In[26]:


train.Item_Identifier.value_counts().sum()


# In[27]:


#Fill missing values of weight of Item According to means of Item Identifier
train['Item_Weight']=train['Item_Weight'].fillna(train.groupby('Item_Identifier')['Item_Weight'].transform('mean'))
train.isnull().sum()


# In[28]:


train[train.Item_Weight.isnull()]


# In[29]:


# List of item types 
item_type_list = train.Item_Type.unique().tolist()
item_type_list


# In[30]:


# grouping based on item type and calculating mean of item weight
Item_Type_Means = train.groupby('Item_Type')['Item_Weight'].mean() 


# In[31]:


Item_Type_Means


# In[32]:


# Mapiing Item weight to item type mean
for i in item_type_list:
    dic = {i:Item_Type_Means[i]}
    s = train.Item_Type.map(dic)
    train.Item_Weight = train.Item_Weight.combine_first(s)
    
Item_Type_Means = train.groupby('Item_Type')['Item_Weight'].mean() 

train.isnull().sum()


# In[33]:


Item_Type_Mean = train.groupby('Item_Type')['Item_Visibility'].mean()


# In[34]:


Item_Type_Mean


# In[35]:


train.isnull().any() # no missing values


# In[36]:


train.Item_Visibility.value_counts().head() # There are 526 values with 0 Item visibility


# In[37]:


# Replacing 0's with NaN
train.Item_Visibility.replace(to_replace=0.000000,value=np.NaN,inplace=True)
# Now fill by mean of visbility based on item identifiers
train.Item_Visibility = train.Item_Visibility.fillna(train.groupby('Item_Identifier')['Item_Visibility'].transform('mean'))


# In[38]:


train.Item_Visibility.value_counts().head()


# ### Renaming Item_Fat_Content levels

# In[39]:


train.Item_Fat_Content.value_counts()


# In[40]:


train.Item_Fat_Content.replace(to_replace=["LF","low fat"],value="Low Fat",inplace=True)
train.Item_Fat_Content.replace(to_replace="reg",value="Regular",inplace=True)

train.Item_Fat_Content.value_counts()


# ### Encoding Categorical Variables

# In[41]:


var_cat = train.select_dtypes(include=[object])
var_cat.head()


# In[42]:


var_cat = var_cat.columns.tolist()
var_cat = ['Item_Fat_Content',
 'Item_Type',
 'Outlet_Size',
 'Outlet_Location_Type',
 'Outlet_Type']

var_cat


# In[43]:


train['Item_Type_New'] = train.Item_Identifier
train.Item_Type_New.head(10)


# In[44]:


train.Item_Type_New.replace(to_replace="^FD*.*",value="Food",regex=True,inplace=True)
train.Item_Type_New.replace(to_replace="^DR*.*",value="Drinks",regex=True,inplace=True)
train.Item_Type_New.replace(to_replace="^NC*.*",value="Non-Consumable",regex=True,inplace=True)

train.head()


# In[45]:


le = LabelEncoder()


# In[46]:


train['Outlet'] = le.fit_transform(train.Outlet_Identifier)
train['Item'] = le.fit_transform(train.Item_Type_New)
train.head()


# In[47]:


for i in var_cat:
    train[i] = le.fit_transform(train[i])

train.head()


# In[48]:


combine.shape


# In[49]:


data=combine.drop('Item_Identifier',axis=1)


# In[50]:


var_cat = combine.select_dtypes(include=[object])
var_cat.head()


# In[51]:


for i in var_cat:
    combine[i] = le.fit_transform(combine[i])

combine.head()


# In[52]:


combine.info()


# In[53]:


#Let's see how data is distributed for every column
plt.figure(figsize = (20,30) , facecolor='red')
plotnumber=1

for columns in combine:
    if plotnumber <=12:
        ax = plt.subplot(4,3,plotnumber)
        sns.distplot(combine[columns])
        plt.xlabel(columns,fontsize=15)
    plotnumber+=1
    
plt.tight_layout()


# In[54]:


df_feature = combine.drop('Outlet_Location_Type', axis = 1)
combine.shape


# Visualize the Outliers Using Boxplot

# In[55]:


#Visualizing Correlation
corrmat = train.corr()
corrmat


# In[56]:


f,ax = plt.subplots(figsize = (20,10))
sns.heatmap(corrmat,annot=True,ax=ax,cmap="YlGnBu",linewidths=0.1,fmt=".2f",square=True)
plt.show()


# ### Visualzing the data

# In[57]:


print(combine.columns)


# In[58]:


combine.dtypes


# In[59]:


predictors=['Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Size','Outlet_Location_Type','Outlet_Type','Outlet_Establishment_Year',
            'Outlet','Item','Item_Weight']
predictors


# In[60]:


#X = train[predictors]
#y = train.Item_Outlet_Sales


# In[61]:


combine.isnull().sum()


# In[62]:


#Fill missing values of weight of Item According to means of Item Identifier
combine['Item_Weight']=combine['Item_Weight'].fillna(combine.groupby('Item_Identifier')['Item_Weight'].transform('mean'))
combine.isnull().sum()


# In[63]:


from sklearn.metrics import classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score,confusion_matrix, roc_curve,roc_auc_score
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[64]:


X = combine.drop(columns = ["Outlet_Type"])
y = combine["Outlet_Type"]


# In[65]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 42)


# In[66]:


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


# In[67]:


from sklearn.neighbors import KNeighborsClassifier


# In[68]:


knn = KNeighborsClassifier()


# In[69]:


knn.fit(X_train,y_train)


# In[70]:


metric_score(knn,X_train,X_test,y_train,y_test, train = True)
metric_score(knn,X_train,X_test,y_train,y_test, train = False)


# In[71]:


lr = LinearRegression()
model = lr.fit(X_train,y_train)
predictions = lr.predict(X_test)


# In[72]:


train['Item_Identifier'].value_counts(normalize = True)
train['Item_Identifier'].value_counts().plot.hist()
plt.title('Different types of item available in the store')
plt.xlabel('Item Identifier')
plt.ylabel('Number of Items')
plt.legend()
plt.show()


# In[73]:



# checking the different items in Item Fat Content

train['Item_Fat_Content'].value_counts()


# In[74]:


train['Item_Fat_Content'].value_counts(normalize = True)
train['Item_Fat_Content'].value_counts().plot.bar()
plt.title('Different varieties of fats in item in the store')
plt.xlabel('Fat')
plt.ylabel('Number of Items')
plt.show()


# In[75]:


combine['Item_Type'].value_counts()


# In[76]:


combine['Item_Type'].value_counts(normalize = True)
combine['Item_Type'].value_counts().plot.bar()
plt.title('Different types of item available in the store')
plt.xlabel('Item')
plt.ylabel('Number of Items')
plt.show()


# In[77]:


combine['Outlet_Identifier'].value_counts()


# In[78]:


combine['Outlet_Identifier'].value_counts(normalize = True)
combine['Outlet_Identifier'].value_counts().plot.bar()
plt.title('Different types of outlet identifier in the store')
plt.xlabel('Item')
plt.ylabel('Number of Items')
plt.show()


# In[79]:


combine['Outlet_Size'].value_counts()


# In[80]:


combine['Outlet_Size'].value_counts(normalize = True)
combine['Outlet_Size'].value_counts().plot.bar()
plt.title('Different types of outlet sizes in the store')
plt.xlabel('Item')
plt.ylabel('Number of Items')
plt.show()


# In[81]:


# checking unique values in the columns of train dataset

combine.apply(lambda x: len(x.unique()))


# In[82]:


combine.head()


# In[83]:


from sklearn.preprocessing import StandardScaler


# In[84]:


# Check multicollinearity problem Find if one feature is dependent on another feature
scalar = StandardScaler()
X_scalar = scalar.fit_transform(X)


# In[85]:


# splitting the dataset into train and test

train = combine.iloc[:8523,:]
test = combine.iloc[8523:,:]

print(train.shape)
print(test.shape)


# In[86]:


plt.hist(combine['Item_Outlet_Sales'], bins = 25, color = 'green')
plt.title('Target Variable')
plt.xlabel('Item Outlet Sales')
plt.ylabel('count')
plt.show()


# In[87]:


plt.scatter(combine.Item_Type,combine.Item_Outlet_Sales)
plt.xlabel("Item Type")
plt.ylabel("Item Outlet Sales")
plt.title("Item Type vs Item Outlet Sales")

plt.show()


# In[88]:


plt.bar(combine.Outlet_Identifier,combine.Item_Outlet_Sales)
plt.show()


# In[89]:


sns.barplot(x='Outlet_Identifier',y='Item_Outlet_Sales', data=combine, palette='magma',capsize = 0.05,saturation = 8,errcolor = 'gray', errwidth = 2,  
            ci = 'sd')
plt.show()


# ### Checking which item type sold the most

# In[90]:


sns.barplot(x='Item_Type',y='Item_Outlet_Sales',data=combine,palette='hls',saturation=8)
plt.xticks(rotation=90)
plt.show()


# ### Using a boxplot to see the outliers in each item type

# In[91]:


sns.boxplot(x='Item_Type',y='Item_MRP',data=combine,palette='Paired',saturation=8)
plt.xticks(rotation=90)
plt.show()


# ### Predictive Modelling

# In[92]:


combine.dtypes


# In[93]:


predictors=['Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Size','Outlet_Location_Type','Outlet_Type','Outlet_Year',
            'Outlet','Item','Item_Weight']
predictors


# In[94]:


seed = 240
np.random.seed(seed)


# In[95]:


X.head()


# In[96]:


y.head()


# In[97]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 42)


# In[98]:


X_train.shape


# In[99]:


X_test.shape


# In[100]:


y_train.shape


# In[101]:


y_test.shape


# In[102]:


lr = LinearRegression()


# In[103]:


model = lr.fit(X_train,y_train)
predictions = lr.predict(X_test)


# In[104]:


predictions[:5]


# ### Plotting the model

# In[105]:


plt.scatter(y_test,predictions)
plt.show()


# In[106]:


print("Linear Regression Model Score:",model.score(X_test,y_test))


# In[107]:


#Root mean squared error
original_values = y_test
rmse = np.sqrt(metrics.mean_squared_error(original_values,predictions))
print("Linear Regression R2 score: ",metrics.r2_score(original_values,predictions))


# In[108]:


print("Linear Regression RMSE: ", rmse)


# In[109]:


# Linear Regression with statsmodels
x = sm.add_constant(X_train)
results = sm.OLS(y_train,x).fit()
results.summary()


# In[110]:


predictions = results.predict(x)
predictions.head()


# In[111]:


predictionsDF = pd.DataFrame({"Predictions":predictions})
joined = x.join(predictionsDF)
joined.head()


# In[112]:


regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)


# In[113]:


predictions = regressor.predict(X_test)
predictions[:5]


# In[114]:


results = pd.DataFrame({'Actual':y_test,'Predicted':predictions})
results.head()


# ### BIG MART SALES VISUALIZATION

# In[115]:


import seaborn as sns
sns.lineplot(data = combine['Item_Outlet_Sales'])


# In[116]:


sns.scatterplot(x = combine['Item_MRP'],y=combine['Item_Outlet_Sales'])


# In[117]:


sns.barplot(x = combine['Item_Fat_Content'], y = combine['Item_Outlet_Sales'])


# In[118]:


sns.swarmplot(x=combine['Outlet_Location_Type'],y=combine['Item_Outlet_Sales'])


# In[119]:


sns.distplot(a = combine['Item_Outlet_Sales'])


# In[120]:


sns.kdeplot(data = combine['Item_Outlet_Sales'], shade = True)


# In[121]:


combine[['Item_MRP','Item_Outlet_Sales']].reset_index()


# In[122]:


duplicates = combine['Item_Outlet_Sales'].index.duplicated()
duplicates


# In[123]:


combine = combine[~duplicates].reset_index(drop=True)
combine


# In[124]:


sns.jointplot(x = combine['Item_MRP'],y = combine['Item_Outlet_Sales'])


# In[125]:


combine.boxplot(column='Item_Weight', by='Outlet_Identifier') #, rot=10)


# In[126]:


# average weight per item
item_av_weight = combine.pivot_table(values='Item_Weight', index='Item_Identifier')
item_av_weight.head()


# In[127]:


combine['Item_Outlet_Sales'] = combine['Item_Outlet_Sales']/combine['Item_MRP']


# In[128]:


# average weight per item
item_av_weight = combine.pivot_table(values='Item_Weight', index='Item_Identifier')
item_av_weight.head()


# In[129]:


combine.pivot_table(values='Outlet_Type', index='Outlet_Identifier', aggfunc=(lambda x:mode(x).mode[0]))


# In[130]:


combine['Outlet_Size'] = combine['Outlet_Size'].fillna('unknown')
outlet_type_mode_size = combine.pivot_table(values='Outlet_Size', index='Outlet_Type', aggfunc=(lambda x:mode(x).mode[0]))
outlet_type_mode_size


# In[131]:


# check out the frequecy of each different category in each nomical value

# filter the categorical variables
categorical_columns = [x for x in combine.dtypes.index if combine.dtypes[x]=='object']

# exclude the id and source columns
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier', 'source']]

# print the frequency of categories
for col in categorical_columns:
    print('\nFrequency of Categories for variable %s'%(col))
    print(combine[col].value_counts())


# In[132]:


# sales per Outlet_Type
ax = combine.boxplot(column='Item_Outlet_Sales', by='Outlet_Type', rot=90)
ax.set_ylabel('Item_Outlet_Sales')
ax.set_title('')


# In[133]:


data = pd.concat([train,test])


# In[134]:


# sales per Outlet_Identifier
ax = combine.boxplot(column='Item_Outlet_Sales', by='Outlet_Identifier', rot=90)
ax.set_ylabel('Item_Outlet_Sales')
ax.set_title('')


# In[135]:


# print Outlet_Type of OUT010 and Out019
outlet_identifier_mode_size = combine.pivot_table(values='Outlet_Size', index='Outlet_Identifier', aggfunc=(lambda x:mode(x).mode[0]))
outlet_identifier_mode_size


# In[136]:


# replace the Outlet_Size of the Grocery Store in the pivot table with small
outlet_type_mode_size.loc['Grocery Store'] = 'Small'
outlet_type_mode_size


# In[137]:


outlet_identifier_mode_size = combine.pivot_table(values='Outlet_Size', index='Outlet_Identifier', aggfunc=(lambda x:mode(x).mode[0]))
outlet_identifier_mode_size


# In[138]:


# create a mask of the missing data in Item_Weight
null_mask_size = combine['Outlet_Size']=='unknown'

# impute values
combine.loc[null_mask_size, 'Outlet_Size'] = combine.loc[null_mask_size, 'Outlet_Type'].apply(lambda x: outlet_type_mode_size.loc[x])


# In[139]:


combine.pivot_table(values='Outlet_Type', index='Outlet_Identifier', aggfunc=(lambda x:mode(x).mode[0]))


# In[140]:


# check how many entries have 0
combine['Item_Visibility'].value_counts().head()


# In[141]:


# visual check that the 0 values in the firs 10 entries have been replaced
combine['Item_Visibility'] = train['Item_Visibility'].replace({0:np.nan})


# In[142]:


# pivot table with the mean values that will be used to replace the nan values
table = combine.pivot_table(values='Item_Visibility', index='Item_Type', columns='Outlet_Type', aggfunc='mean')
table


# Combine Low Fat, low fat and LF to Low Fat and reg and Regular to Regular

# In[143]:


combine['Item_Fat_Content'] = combine['Item_Fat_Content'].replace({'LF': 'Low Fat', 
                                                             'low fat': 'Low Fat', 
                                                             'reg': 'Regular'})
combine['Item_Fat_Content'].head(5)


# In[144]:


print('\nFrequency of Categories for variable Item_Fat_Content')
print(combine['Item_Fat_Content'].value_counts())


# Convert the Outlet_Establishment_Years into how old the establishments are

# In[145]:


combine['Outlet_Age'] = 2023 - combine['Outlet_Establishment_Year']
combine['Outlet_Age'].head(5)


# In[146]:


combine['Outlet_Age'].describe()


# In[147]:


ax = sns.distplot(combine['Item_MRP'])
x1=72
x2=138
x3=204
ax.plot([x1, x1],[0, 0.005], color='r')
ax.plot([x2, x2],[0, 0.008],color='b')
ax.plot([x3, x3],[0, 0.006],color='g')
plt.show()


# The Item_MRP clearly shows there are 4 different price categories. So we define them to be 'Low', 'Medium', 'High', 'Very High'.

# In[148]:


def price_cat(x):
    if x <= x1:
        return 'Low'
    elif (x > x1) & (x <= x2):
        return 'Medium'
    elif (x > x2) & (x <= x3):
        return 'High'
    else:
        return 'Very High'

combine['Item_MRP_Category'] = combine['Item_MRP']
combine['Item_MRP_Category'] = combine['Item_MRP_Category'].apply(price_cat)
combine['Item_MRP_Category'].value_counts()


# ## Analysis
# ## Numerical data

# In[149]:


data.describe()


# As we have seen previously, the Item_MRP is clearly divided into 4 categories. Now, let's plot the Item_MRP grouped by the Outlet_Type and Outle_Size.

# In[150]:


ax = combine.hist(column='Item_MRP' , by='Outlet_Type', bins=25, density=True)


# In[151]:


ax = combine.hist(column='Item_MRP' , by='Outlet_Size', bins=40, density=True)


# In[152]:


ax = combine.hist(column='Item_MRP' , by='Outlet_Identifier', bins=25, density=True)


# Explore how Item_MRP depends on Outlet_Type:

# In[153]:


ax = sns.boxplot(data=combine, x='Outlet_Type', y='Item_MRP')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
leg = ax.legend()
ax.legend(loc='center right', bbox_to_anchor=(1.45, 0.5))


# Item_Outlet_Sales

# Item_Outlet_Sales are very low for Grocery Stores, even though we saw above the Item_MRP is the same for all Outlet_Types.
# 
# 

# Let's Explore if this is because of the Outlet_Size.

# In[156]:


ax = sns.boxplot(data=combine, x='Item_Type', y='Item_Outlet_Sales', hue='Outlet_Type')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


# Grocery stores just sell a smaller number of everything.

# In[158]:


# Item_Outlet_Sales per Outlet_Identifier
ax = sns.boxplot(data=combine, x='Outlet_Identifier', y='Item_Outlet_Sales')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


# In[159]:


for i in combine['Outlet_Identifier'].unique():
    otype = combine[combine['Outlet_Identifier']==i]['Outlet_Type'].unique()
    osize = combine[combine['Outlet_Identifier']==i]['Outlet_Size'].unique()
    print('Outlet_Identifier: {}, Outlet_Type(s): {}, Outlet_Size: {}'.format(i, otype, osize))


# In[160]:


combine['Item_Number_Sales'] = combine['Item_Outlet_Sales']/combine['Item_MRP']


# In[161]:


ax = sns.boxplot(data=combine, x='Item_Type', y='Item_Number_Sales', hue='Outlet_Type')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
leg = ax.legend()
ax.legend(loc='center right', bbox_to_anchor=(1.45, 0.5))


# In[164]:


# Item_Outlet_Sales per Outlet_Identifier
ax = sns.boxplot(data=combine, x='Outlet_Identifier', y='Item_Number_Sales')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


# ### Item_outlet_sales and Item_MRP vs Item_Visibility

# In[165]:


sns.pairplot(data=combine, x_vars='Item_MRP', y_vars='Item_Number_Sales', hue='Outlet_Type', size=5)


# In[166]:


sns.pairplot(data=combine, x_vars='Item_MRP', y_vars='Item_Outlet_Sales', hue='Outlet_Type', size=5)


# In[167]:


cor1 = combine['Item_MRP'].corr(combine['Item_Outlet_Sales'])
cor2 = combine['Item_MRP'].corr(combine['Item_Number_Sales'])
print('Correlation between Item_MRP and Item_Outlet_Sales: {}'.format(cor1))
print('Correlation between Item_MRP and Item_Number_Sales: {}'.format(cor2))


# In[168]:


sns.pairplot(data=combine, x_vars='Item_Visibility', y_vars='Item_Outlet_Sales', hue='Outlet_Type', size=5)


# In[170]:


sns.pairplot(data=combine, x_vars='Item_Visibility', y_vars='Item_Number_Sales', hue='Outlet_Type', size=5)


# In[180]:


# check out the frequecy of each different category in each nomical value

# filter the categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']

# exclude the id and source columns
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier', 'source']]

# print the frequency of categories
for col in categorical_columns:
    print('\nFrequency of Categories for variable %s'%(col))
    print(data[col].value_counts())


# In[182]:


data


# In[172]:


train['Item_Type_New'] = train.Item_Identifier
train.Item_Type_New.head(10)


# In[174]:


#visualizing corelation
corrmat = combine.corr()
corrmat


# In[ ]:





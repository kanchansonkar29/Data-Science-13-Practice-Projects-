#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
#     </a>
# </p>
# 

# # **Create visualizations using Matplotib, Seaborn and Folium** 
# 
# Estimated time needed: **40** minutes
# 
# In this assignment, you will have the opportunity to demonstrate the skills you have acquired in creating visualizations using *Matplotlib, Seaborn, Folium*.
# <br>
# <br>
# <span style="color:red">After each task you will be required to save your plots as an image or screenshot using the filenames specified.  You will be uploading these images during your final project submission so they can be evaluated by your peers. </span>
# 

# # __Table of Contents__
# 
# <ol>
#     <li><a href="#Objectives">Objectives</a></li>
#     <li>
#         <a href="#Setup">Setup</a>
#         <ol>
#             <li><a href="#Installing-Required-Libraries">Installing Required Libraries</a></li>
#             <li><a href="#Importing-Required-Libraries">Importing Required Libraries</a></li>
#             </ol>
#     </li>
#     <li>
#         <a href="#Scenario">Scenario</a>
#         <ol>
#             <li><a href="#Data Description">Data Description</a></li>
#         </ol>
#     </li>
#     <li><a href="#Importing Data">Importing data</a></li>
#     <li><a href="#Creating Visualizations for Data Analysis">Creating Visualizations for Data Analysis</a></li>
# </ol>
# 

# # Objectives
# 
# After completing this lab you will be able to:
# 
# - Create informative and visually appealing plots with Matplotlib and Seaborn.
# - Apply visualization to communicate insights from the data.
# - Analyze data through using visualizations.
# - Customize visualizations
# 

# # Setup
# 

# For this lab, we will be using the following libraries:
# 
# *   [`pandas`](https://pandas.pydata.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0187ENSkillsNetwork31430127-2021-01-01) for managing the data.
# *   [`numpy`](https://numpy.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0187ENSkillsNetwork31430127-2021-01-01) for mathematical operations.
# *   [`matplotlib`](https://matplotlib.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0187ENSkillsNetwork31430127-2021-01-01) for plotting.
# *   [`seaborn`](https://seaborn.pydata.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0187ENSkillsNetwork31430127-2021-01-01) for plotting.
# *  [`Folium`](https://python-visualization.github.io/folium/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0187ENSkillsNetwork31430127-2021-01-01) for plotting.
# 

# ### Installing Required Libraries
# 
# The following required libraries are pre-installed in the Skills Network Labs environment. However, if you run these notebook commands in a different Jupyter environment (e.g. Watson Studio or Ananconda), you will need to install these libraries by removing the `#` sign before `%pip` in the code cell below.
# 

# In[1]:


# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented.
# %pip install -qy pandas==1.3.4 numpy==1.21.4 matplotlib==3.5.0 seaborn folium
# Note: If your environment doesn't support "%pip install", use "!mamba install"


# In[2]:


get_ipython().run_line_magic('pip', 'install seaborn')
get_ipython().run_line_magic('pip', 'install folium')


# ### Importing Required Libraries
# 
# _We recommend you import all required libraries in one place (here):_
# 

# In[3]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import folium


# 
# 
# <details><summary>Click here for a hint</summary>
# 
# <p>
# You will require:-
# <br>Numpy for many scientific computing in Python
# <br>Pandas for creating and working on dataframe, also for plotting directly on dataframe/series
# <br>The inline backend to generate the plots within the browser [%matplotlib inline]
# <br>Matplotlib and its pyplot pacakge for plotting
# <br>Seaborn for plotting
# </details>
# 
# 
# 

# 
# <details><summary>Click here for python solution</summary>
# 
# ```python
#     #Import Primary Modules:
#     import numpy as np
#     import pandas as pd
#     %matplotlib inline
#     import matplotlib as mpl
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import folium
# ```
# 
# </details>
# 

# ---
# 

# # Scenario
# 
# In this assignment you will be tasked with creating plots which answer questions for analysing "historical_automobile_sales" data to understand the historical trends in automobile sales during recession periods.<br>
# recession period 1 - year 1980 <br>
# recession period 2 - year 1981 to 1982<br>
# recession period 3 - year 1991<br>
# recession period 4 - year 2000 to 2001<br>
# recession period 5 - year end 2007 to mid 2009<br>
# recession period 6 - year 2020 -Feb to April (Covid-19 Impact)<br>
# 
# # Data Description
# 
# The dataset used for this visualization assignment contains *historical_automobile_sales* data representing automobile sales and related variables during recession and non-recession period. 
# 
# The dataset includes the following variables:
# <br>1. Date: The date of the observation.
# <br>2. Recession: A binary variable indicating recession perion; 1 means it was recession, 0 means it was normal.
# <br>3. Automobile_Sales: The number of vehicles sold during the period.
# <br>4. GDP: The per capita GDP value in USD.
# <br>5. Unemployment_Rate: The monthly unemployment rate.
# <br>6. Consumer_Confidence: A synthetic index representing consumer confidence, which can impact consumer spending and automobile purchases.
# <br>7. Seasonality_Weight: The weight representing the seasonality effect on automobile sales during the period.
# <br>8. Price: The average vehicle price during the period.
# <br>9. Advertising_Expenditure: The advertising expenditure of the company.
# <br>10.Vehicle_Type: The type of vehicles sold; Supperminicar, Smallfamiliycar,                 Mediumfamilycar, Executivecar, Sports.
# <br>11.Competition: The measure of competition in the market, such as the number of competitors or market share of major manufacturers.
# <br>12.Month: Month of the observation extracted from Date..
# <br>13.Year: Year of the observation extracted from Date.
# <br>
# By examining various factors mentioned above from the dataset, you aim to gain insights into how recessions impacted automobile sales for your company.
# 

# ---
# 

# # Importing Data
# 

# #### For your convenience, we have already written code to import the data below.
# 

# In[4]:


#from js import fetch
#import io

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv"
#resp = await fetch(URL)
#text = io.BytesIO((await resp.arrayBuffer()).to_py())
#import pandas as pd
df = pd.read_csv(URL)
print('Data downloaded and read into a dataframe!')


# In[5]:


df.describe()


# In[6]:


df.columns


# In[7]:


df.dtypes


# ---
# 

# # Creating Visualizations for Data Analysis
# 

# ### TASK 1.1: Develop a *Line chart* using the functionality of pandas to show how automobile sales fluctuate from year to year
# 

# <details><summary>Click here for a hint</summary>
# 
# <p>
# You will require:-
# <br>to group the year and calculate the average on the 'Automobile Sales', as the data has years and months column
# <br>make use of .plot() with kind = 'line'
# <br>donot forget to include labels and title
# </details>
# 

# In[8]:


temp = df.groupby("Year")["Automobile_Sales"].sum()
temp.plot(x="Year", y="Automobile_Sales", kind="line")
plt.title("Automobile sold across the years")
plt.xlabel("Year")
plt.ylabel("Number of Automobiles")


# <details><summary>Click here for a solution template</summary>
# 
# ```python
#     #create data for plotting
#     df_line = df.groupby(df['Year'])['Automobile_Sales'].mean()
#     #create figure
#     plt.figure(figsize=(10, 6))
#     df_line.plot(kind = 'line')
#     plt.xlabel('Year')
#     plt.ylabel('Sales Volume')
#     plt.title('Automobile Sales over Time')
#     plt.show()
# ```
# </details>
# 

# ### Include the following on the plot
# ticks on x- axis with all the years, to identify the years of recession 
# <br>annotation for at least two years of recession
# <br>Title as Automobile Sales during Recession
# <br> 
# 

# <details><summary>Click here for a hint</summary>
#     <p>
#     You can create the list for the range 1980 till 2023 and pass that list to the plt.xticks function or you can directly pass the range to the function.
#     You might need to rotate the ticks to an angle so that they fit in well on the axis
#     You can include annotation with plt.text(x, y, 'text to display') 
#     </p>
# </details>
# 

# In[9]:


plt.figure(figsize=(10, 6))
temp.plot(x="Year", y="Automobile_Sales", kind="line")
plt.xticks(list(range(df.Year.unique().min(), df.Year.unique().max()+1)), rotation=75)
years = [1979.5, 1981.5, 1991.5, 2000.6, 2008.2, 2006.7]
texts = ["1980 Recession", 
         "1981-82 Recession", 
         "1991 Recession", 
         "2000-2001 Recession", 
         "2007-2009 Recession", 
         "2020 (Covid-19 Impact) Recession"]
positions = [(i+1)*(10**4) for i in range(len(years))]
positions = [10000, 9000, 9000, 16000, 22000, 32000]
for year, text, pos in zip(years, texts, positions):
    if year==1979.5:
        plt.text(year, pos, text, rotation=90, color="blue",
             bbox=dict(boxstyle="larrow",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8)
                      )
            )
        continue
    if year==2006.7:
        plt.text(year, pos, text, rotation=-30, color="blue",
             bbox=dict(boxstyle="rarrow",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8)
                      )
            )
        continue
    plt.text(year, pos, text, rotation=45, color="blue",
             bbox=dict(boxstyle="larrow",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8)
                      )
            )
plt.grid()
plt.title("Automobile sold across the years")
plt.xlabel("Year")
plt.ylabel("Number of Automobiles")


# <details>
#     <summary>Click here for Solution template</summary>
# 
# ```python
#     plt.figure(figsize=(10, 6))
#     df_line = ...............
#     df_line.plot(kind = 'line')
#     plt.xticks(list(range(1980,2024)), rotation = 75)
#     plt.xlabel('..............')
#     plt.ylabel('............')
#     plt.title('...................')
#     plt.text(1982, 650, '1981-82 Recession')
#     plt.text(......, ..., '..............')
#     plt.legend()
#     plt.show()
# ```
# 
# </details>
# 

# <span style="color:red">
# Save this plot as "Line_Plot_1.png"</span><br>
# *Hint: You can right lick on the plot and then click on "Save image as" option to save it on your local machine*
# 

# ---
# 

# 
# ####  TASK 1.2: Plot different lines for categories of vehicle type and analyse the trend to answer the question Is there a noticeable difference in sales trends between different vehicle types during recession periods?
# 

# <details><summary>Click here for a hint</summary>
# 
#     <p>
#     You will require:-
#     <br>create a separate dataframe where the column recession has a value of '1'
#     <br>to group the year, vehicle_type and calculate the average on the 'Automobile Sales'
#     <br>one way is to -
#     <br>use as_index as false else you will endup with multiple-indexed datafame
#     <br>later set year as index and groupby vehicle over Sales and plot
#     <br>make use of .plot() with kind = 'line'
#     <br>do not forget to include labels and title
# </details>
# 

# In[10]:


plt.figure(figsize=(10, 6))
temp = df.groupby(["Year", "Vehicle_Type"])[["Automobile_Sales"]].sum().reset_index()
temp.set_index("Year", inplace=True)
temp = temp.groupby(["Vehicle_Type"])["Automobile_Sales"]
temp.plot(kind="line")
plt.xticks(list(range(df.Year.unique().min(), df.Year.unique().max()+1)), rotation=75)
plt.grid()
plt.title("Automobile sold across the years")
plt.xlabel("Year")
plt.ylabel("Number of Automobiles")
plt.legend()


# <details>
#     <summary>Click here for Solution template</summary>
# 
# ```python
#     df_Mline = df.groupby(['Year','Vehicle_Type'], as_index=False)['Automobile_Sales'].sum()
#     df_Mline.set_index('Year', inplace=True)
#     df_Mline = df_Mline.groupby(['Vehicle_Type'])['Automobile_Sales']
#     df_Mline.plot(kind='line')
#     plt.xlabel('..............')
#     plt.ylabel('............')
#     plt.title('Sales Trend Vehicle-wise during Recession')
#     plt.legend()
#     plt.show()
# 
# ```
# </details>
# 

# #### From the above plot, what insights have you gained on the sales of various vehicle types?<br> Type in your answer below:
# 

# 

# <details>
#     <summary>Inference</summary>
# <p>
# Inference:
# From this plot, we can understand that during recession period, the sales for 'Sports type vehicles' declined because of the high cost of the vehicle.<br>while sales of the superminicar and smallfamilycar increased.<br><br>
#     </p>
#     </details>
# 

# <span style="color:red">
# Save this plot as "Line_Plot_2.png"</span><br>
# *Hint: You can right lick on the plot and then click on "Save image as" option to save it on your local machine*
# 

# ----
# 

# ### TASK 1.3: Use the functionality of **Seaborn Library** to create a visualization to compare the sales trend per vehicle type for a recession period with a non-recession period.
# 

# <details><summary>Click here for a hint</summary>
# 
# <p>
#      To visualize the average number of vehicles sold during recession and non-recession periods, you can use a bar chart
#         <br> You will need to group recession average Automobile_Sales and then plot it<br>
#     Make use of sns.barplot(x=x,y=y, data = df)</p>
# </details>
# 

# In[11]:


temp=df.groupby(["Recession"])["Automobile_Sales"].mean().reset_index()
temp["Recession"] = temp["Recession"].replace({0:"No Recession", 1:"Recession"})
# temp
plt.figure(figsize=(10, 6))
sns.barplot(data=temp, x="Recession", y="Automobile_Sales")
plt.xlabel('Period')
plt.ylabel('Average sales')
plt.title('Average Automobile Sales during Recession and Non-Recession')


# <details>
#     <summary>Click here for Solution template</summary>
# 
# ```python
#     new_df = df.groupby('Recession')['Automobile_Sales'].mean().reset_index()
# 
#     # Create the bar chart using seaborn
#     plt.figure(figsize=(.........., ............)
#     sns.barplot(x='Recession', y='Automobile_Sales', hue='Recession',  data=new_df)
#     plt.xlabel('............')
#     plt.ylabel('...............')
#     plt.title('Average Automobile Sales during Recession and Non-Recession')
#     plt.xticks(ticks=[0, 1], labels=['Non-Recession', 'Recession'])
#     plt.show()
# ```
# 
# </details>
# 

# ### Now you want to compare the sales of different vehicle types during a recession and a non-recession period
# <br>We recommend that you use the functionality of **Seaborn Library** to create this visualization
# 

# <details><summary>Click here for a hint</summary>
# 
# <p>
#      To visualize sales of different vehicles during recession and non-recession periods, you can use a bar chart
#         <br> You will need to group Recession, Vehicle_Type for average Automobile_Sales and then plot it<br>
#     Make use of sns.barplot(x=x,y=y, data = df)</p>
# </details>
# 

# In[12]:


temp=df.groupby(["Vehicle_Type", "Recession"])["Automobile_Sales"].mean().reset_index()
temp["Recession"] = temp["Recession"].replace({0:"No", 1:"Yes"})
plt.figure(figsize=(10, 6))
sns.barplot(data=temp, x="Vehicle_Type", y="Automobile_Sales", hue="Recession")
plt.xlabel('VEhicle Type')
plt.ylabel('Average sales')
plt.title('Average Automobile Sales per Vehicle Type during Recession and Non-Recession')


# <details>
# <summary>Click here for Solution template</summary>
# 
# ```python
#    # Filter the data for recessionary periods
#     recession_data = df[df['Recession'] == 1]
# 
#     dd=df.groupby(['Recession','Vehicle_Type'])['Automobile_Sales'].mean().reset_index()
# 
#     # Calculate the total sales volume by vehicle type during recessions
#     #sales_by_vehicle_type = recession_data.groupby('Vehicle_Type')['Automobile_Sales'].sum().reset_index()
# 
#     # Create the grouped bar chart using seaborn
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='Recession', y='Automobile_Sales', hue='Vehicle_Type', data=dd)
#     plt.xticks(ticks=[0, 1], labels=['Non-Recession', 'Recession'])
#     plt.xlabel('Period')
#     plt.ylabel('Average Sales')
#     plt.title('Vehicle-Wise Sales during Recession and Non-Recession Period')
# 
#     plt.show()
# 
# ```
# </details>
# 

# ### From the above chart what insights have you gained on the overall sales of automobiles during recession? <br> Type your answer below:-
# 

# 

# <details>
# <summary>Inference</summary>
# 
# From this plot, we can understand that there is a drastic decline in the overall sales of the automobiles during recession.<br>However, the most affected type of vehicle is executivecar and sports<br><br>
# </details>
# 

# <span style="color:red">
# Save this plot as "Bar_Chart.png"</span><br>
# *Hint: You can right lick on the plot and then click on "Save image as" option to save it on your local machine*
# 

# ---
# 

# ### TASK 1.4: Use sub plotting to compare the variations in GDP during recession and non-recession period by developing line plots for each period.
# <br>Now, you want to find more insights from the data to understand the reason. <br>Plot a two line charts using subplotting to answer:-
# #### How did the GDP vary over time during recession and non-recession periods? 
# <br>Make use of <u>add_subplot()</u> from Matplotlib for this comparision.
# 

# In[13]:


temp = df.groupby(["Recession","Year"])["GDP"].mean().reset_index()
no = temp[temp["Recession"] == 0]
yes = temp[temp["Recession"] == 1]

fig = plt.figure(figsize=(10, 6))
ax0 = fig.add_subplot(1,2,1)
ax1 = fig.add_subplot(1,2,2)

no.plot(x="Year", y="GDP", kind="line", ax=ax0)
ax0.set_title('GDP Variation during No Recession Period')
ax0.set_ylabel('GDP')

yes.plot(x="Year", y="GDP", kind="line", ax=ax1)
ax1.set_title('GDP Variation during Recession Period')
ax1.set_ylabel('GDP')

plt.show()


# <details>
# <summary>Click here for Solution template</summary>
# 
# ```python
#     #Create dataframes for recession and non-recession period
#     rec_data = df[df['Recession'] == 1]
#     non_rec_data = df[df['Recession'] == 0]
#     
#     #Figure
#     fig=plt.figure(figsize=(12, 6))
#     
#     #Create different axes for subploting
#     ax0 = fig.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
#     ax1 = fig.add_subplot(... ,... ,... ) # add subplot 2 (1 row, 2 columns, second plot). 
#     
#     #plt.subplot(1, 2, 1)
#     sns.lineplot(x='Year', y='GDP', data=rec_data, label='Recession', ax=ax0)
#     ax0.set_xlabel('Year')
#     ax0.set_ylabel('GDP')
#     ax0.set_title('GDP Variation during Recession Period')
#     
#     #plt.subplot(1, 2, 2)
#     sns.lineplot(x='......', y='......', data=........, label='.........',ax=...)
#     ax1.set_xlabel('.....')
#     ax1.set_ylabel('.......')
#     ax1.set_title('..........')
#     
#     plt.tight_layout()
#     plt.show()
# 
#    #------------------------------------------------Alternatively--------------
#    #Using subplot()
#     plt.figure(figsize=(............, ..........))
#     
#     #subplot 1
#     plt.subplot(1, 2, 1)
#     sns.lineplot(x='.........', y='......', data=.........., label='......')
#     plt.xlabel('.......')
#     plt.ylabel('..........')
#     plt.legend()
#     #subplot 1
#     plt.subplot(1, 2, 2)
#     sns.lineplot(x='.........', y='......', data=.........., label='......')
#     plt.xlabel('.......')
#     plt.ylabel('..........')
#     plt.legend()
#     
#     plt.tight_layout()
#     plt.show()
# ```
# </details>
# 

# ### Inference
# From this plot, it is evident that during recession, the GDP of the country was in a low range, might have afected the overall sales of the company<br><br>
# <span style="color:red">
# Save this plot as "Subplot.png"</span><br>
# *Hint: You can right lick on the plot and then click on "Save image as" option to save it on your local machine*
# 

# ---
# 

# ### TASK 1.5: Develop a Bubble plot for displaying the impact of seasonality on Automobile Sales.
# <br>How has seasonality impacted the sales, in which months the sales were high or low? Check it for non-recession years to understand the trend
# 
# ##### Develop a Bubble plot for displaying Automobile Sales for every month and use Seasonality Weight for representing the size of each bubble<br>
# Title this plot as 'Seasonality impact on Automobile Sales'
# 

# <details><summary>Click here for a hint</summary>
# 
# <p>
#      You can create Bubble Chart by calling the scatter()
#         <br>Pass the 'Month' and 'Automobile_Sales' to the functions as x and y and then use Seasonality weight for size parameter</p>
# </details>
# 

# In[14]:


temp = df.copy()
months=temp["Month"].unique().tolist()
# display(months)
temp["Month"] = temp["Month"].replace(dict(zip(months, range(0,len(months)))))
temp = temp.groupby("Month")[["Automobile_Sales", "Seasonality_Weight"]].mean().reset_index()
# display(temp)
plt.figure(figsize=(10, 6))
plt.xticks(ticks=range(0,len(months)),labels=months)
sns.scatterplot(data=temp,
                x='Month',
                y='Automobile_Sales',
                hue="Seasonality_Weight",
                size="Seasonality_Weight",  # pass in weights 
                #sizes=(20, 1500)
               )
plt.xlabel("Month")
plt.ylabel("Automobile Sales")
plt.title("Automobile Sales per Month according to Seasonality Weight")
plt.legend(loc="right")


# <details>
# <summary>Click here for Solution template</summary>
# 
# ```python
#     non_rec_data = df[df['Recession'] == 0]
#     
#     size=non_rec_data['Seasonality_Weight'] #for bubble effect
#     
#     sns.scatterplot(data=non_rec_data, x='....', y='........', size=size)
#     
#     #you can further include hue='Seasonality_Weight', legend=False)
# 
#     plt.xlabel('Month')
#     plt.ylabel('Automobile_Sales')
#     plt.title('Seasonality impact on Automobile Sales')
# 
#     plt.show()
# 
# ```
# </details>
# 

# ### Inference
# From this plot, it is evident that seasonality has not affected on the overall sales. However, there is a drastic raise in sales in the month of April<br><br>
# <span style="color:red">
# Save this plot as "Bubble.png"</span><br>
# *Hint: You can right lick on the plot and then click on "Save image as" option to save it on your local machine*
# 

# ---
# 

#  ### TASK 1.6: Use the functionality of Matplotlib to develop a scatter plot to identify the correlation between average vehicle price relate to the sales volume during recessions.
#  #### From the data, develop a scatter plot to identify if there a correlation between consumer confidence and automobile sales during recession period? 
#  <br> Title this plot as 'Consumer Confidence and Automobile Sales during Recessions'
# 

# <details><summary>Click here for a hint</summary>
# 
# <p>
#      You can create dataframe where recession is '1'.
#         <br>Pass the 'Consumer_Confidence' and 'Automobile_Sales' to the plt.scatter()</p>
# </details>
# 

# In[15]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df[df["Recession"]==1], x='Consumer_Confidence', y='Automobile_Sales')
plt.title("Consumer Confidence and Automobile Sales during Recessions")
plt.xlabel("Consumer Confidence")
plt.ylabel("Automobile Sales")
plt.show()


# <details>
# <summary>Click here for Solution template</summary>
# 
# ```python
#     #Create dataframes for recession and non-recession period
#     rec_data = df[df['Recession'] == 1]
#     plt.scatter(recession_data['Consumer_Confidence'], rec_data['Automobile_Sales'])
#     
#     plt.xlabel('.....')
#     plt.ylabel('.......')
#     plt.title('..........')
#     plt.show()
# 
# ```
# </details>
# 

# 
#  ### How does the average vehicle price relate to the sales volume during recessions?
#  <br> Plot another scatter plot and title it as 'Relationship between Average Vehicle Price and Sales during Recessions'
# 

# In[16]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df[df["Recession"]==1], x='Price', y='Automobile_Sales')
plt.title("Relationship between Average Vehicle Price and Sales during Recessions")
plt.xlabel("Price")
plt.ylabel("Automobile Sales")
plt.show()


# <details>
# <summary>Click here for Solution template</summary>
# 
# ```python
#     #Create dataframes for recession and non-recession period
#     rec_data = df[df['Recession'] == 1]
#     plt.scatter(recession_data['Price'], rec_data['Automobile_Sales'])
#     
#     plt.xlabel('.....')
#     plt.ylabel('.......')
#     plt.title('..........')
#     plt.show()
# 
# ```
# </details>
# 

# ### Inference
# There is not much relation!<br><br>
# <span style="color:red">
# Save this plot as "Scatter.png"</span><br>
# *Hint: You can right lick on the plot and then click on "Save image as" option to save it on your local machine*
# 

# ----
# 

#  ### TASK 1.7: Create a pie chart to display the portion of advertising expenditure of XYZAutomotives during recession and non-recession periods.
#  <br>How did the advertising expenditure of XYZAutomotives change during recession and non-recession periods? 
# 

# <details><summary>Click here for a hint</summary>
# 
# <p>
#      You can create two dataframe for recession and nonreccession period.
#     <br> Calculate the sum of Advertising_Expenditure for both dataframes
#     <br> Pass these total values to plt.pie(). May include labels as ['Recession', 'Non-Recession']
#         <br>Feel Free to customie the pie further
#     <br>title this plot as  - Advertising Expenditure during Recession and Non-Recession Periods</p>
# </details>
# 

# In[17]:


temp=df.groupby("Recession")["Advertising_Expenditure"].sum()/df["Advertising_Expenditure"].sum()
temp.index=["No Recession", "Recession"]
display(temp)
temp.plot(kind="pie",
          figsize=(12, 8),
          y="Advertising_Expenditure",
          #ylabel=dict(ylabel="Advertising Expenditure",fontsize=100),
          x="Recession",
          #title="HOLA",
          autopct='%1.1f%%',
          textprops={'color':"w", "size":22},
         )
plt.ylabel(ylabel="Advertising Expenditure",fontsize=20)
plt.title(label="Advertising Expenditure during\n Recession and No Recession periods",fontsize=25)
plt.legend(fontsize=15)
plt.show()


# <details>
# <summary>Click here for Solution template</summary>
# 
# ```python
#     # Filter the data 
#     Rdata = df[df['Recession'] == 1]
#     NRdata = df[df['Recession'] == 0]
# 
#     # Calculate the total advertising expenditure for both periods
#     RAtotal = Rdata['Advertising_Expenditure'].sum()
#     NRAtotal = NRdata['Advertising_Expenditure'].sum()
# 
#     # Create a pie chart for the advertising expenditure 
#     plt.figure(figsize=(8, 6))
# 
#     labels = ['Recession', 'Non-Recession']
#     sizes = [RAtotal, NRtotal]
#     plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
# 
#     plt.title('Advertising Expenditure during Recession and Non-Recession Periods')
# 
#     plt.show()
# 
# 
# ```
# </details>
# 

# #### From the above plot, what insights do you find on the advertisement expenditure during recession and non recession periods?<br> Type your answer below:-
# 

# 

# <details><summary>Inference</summary>
# It seems ABCAutomotives has been spending much more on the advertisements during non-recession periods as compared to during recession times. Fair enough!<br><br></details>
# 

# <span style="color:red">
# Save this plot as "Pie_1.png"</span><br>
# *Hint: You can right lick on the plot and then click on "Save image as" option to save it on your local machine*
# 

# ---
# 

# ### TASK 1.8: Develop a pie chart to display the total Advertisement expenditure for each vehicle type during recession period.<br>
# Can we observe the share of each vehicle type in total sales during recessions? 
# #### Create another pie plot to display the total advertisement expenditure for each vehicle type
# 

# <details><summary>Click here for a hint</summary>
# 
# <p>
#      You will be required to group vehicle type for sum of advertisement expenditure.
#     <br> the plot a pie with the data, May include relevant labels
#     <br>title this plot as  - Share of Each Vehicle Type in Total Sales during Recessions</p>
# </details>
# 

# In[18]:


temp=df[df["Recession"]==1].groupby("Vehicle_Type")["Advertising_Expenditure"].sum()
temp.plot(kind="pie",
          figsize=(12, 8),
          autopct='%1.1f%%',
          y="Advertising_Expenditure",
          textprops={'color':"w", "size":20},
         )
plt.ylabel(ylabel="Advertising Expenditure",fontsize=20)
plt.title(label="Advertising Expenditure during\n Recession period per Vehicle type",fontsize=25)
plt.legend(title="Vehicle Type", loc="best", fontsize=11)
plt.show()


# <details>
#     <summary>Click here for Solution template</summary>
# 
# ```python
#     # Filter the data 
#     Rdata = df[df['Recession'] == 1]
# 
#     # Calculate the sales volume by vehicle type during recessions
#     VTsales = Rdata.groupby('Vehicle_Type')['Advertising_Expenditure'].sum()
# 
#     # Create a pie chart for the share of each vehicle type in total sales during recessions
#     plt.figure(figsize=(8, 6))
# 
#     labels = VTsales.index
#     sizes = VTsales.values
#     plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
# 
#     plt.title('Share of Each Vehicle Type in Total Sales during Recessions')
# 
#     plt.show()
# ```
#     </details>
# 

# ### Inference
# During recession the advertisements were mostly focued on low price range vehicle. A wise decision!<br><br>
# <span style="color:red">
# Save this plot as "Pie_2.png"</span><br>
# *Hint: You can right lick on the plot and then click on "Save image as" option to save it on your local machine*
# 

# ----
# 

#  ### TASK 1.9: Develop a countplot to analyse the effect of the unemployment rate on vehicle type and sales during the Recession Period.
#  <br>Analyze the effect of the unemployment rate on vehicle type and sales during the Recession Period
#  #### You can create a countplot and title the plot as 'Effect of Unemployment Rate on Vehicle Type and Sales'
# 

# <details><summary>Click here for a hint</summary>
# 
# <p>
#     Filter out the data for recession period<br>
#      Make use of countplot() from seaborn and pass the relavent data</p>
# </details>
# 

# In[19]:


#df.columns
plt.figure(figsize=(10,6))
sns.countplot(data=df[df["Recession"]==1], x="unemployment_rate", hue="Vehicle_Type")
plt.xlabel("Unemployment Rate")
plt.ylabel("Count")
plt.title("Effect of Unemployment Rate on Vehicle Type and Sales")
plt.legend(loc="upper right", title="Vehicle Type")


# <details>
#     <summary>Click here for Solution template</summary>
# 
# ```python
#     data= df[df['Recession'] == 1]
#     
#     plt.figure(figsize=(10, 6))
#     
#     sns.countplot(data=data, x='unemployment_rate', hue='Vehicle_Type')
#     
#     plt.xlabel('Unemployment Rate')
#     plt.ylabel('Count')
#     plt.title('Effect of Unemployment Rate on Vehicle Type and Sales')
#     plt.legend(loc='upper right')
#     plt.show()
# ```
# 
# </details>
# 

# #### From the above plot, what insights have you gained on the sales of superminicar, smallfamilycar, mediumminicar?<br> Type your answer below:-
# 

# 

# <details><summary>Inference</summary>
# During recession, buying pattern changed, the sales of low range vehicle like superminicar,smallfamilycar and Mediumminicar<br><br>
# </details>
# 

# <span style="color:red">
# Save this plot as "count_plot.png"</span><br>
# *Hint: You can right lick on the plot and then click on "Save image as" option to save it on your local machine*
# 

# ---
# 

# ### OPTIONAL : TASK 1.10 Create a map on the hightest sales region/offices of the company during recession period
# 

# In[20]:


#from pyodide.http import pyfetch

#async def download(url, filename):
    #response = await pyfetch(url)
    #if response.status == 200:
         #with open(filename, "wb") as f:
            #f.write(await response.bytes())

#path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/us-states.json'
#await download(path, "us-states.json")

get_ipython().system('wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/us-states.json')
get_ipython().system('head -2 us-states.json')

filename = "us-states.json"
get_ipython().system('head -2 us-states.json')


# #### You found that the datset also contains the location/city for company offices. Now you want to show the recession impact on various offices/city sales by developing a choropleth
# 

# In[21]:


temp=pd.read_json(filename)
for row in temp["features"]:
    print(row["id"], row["properties"]["name"])


# <details><summary>Click for Solution</summary>
#     
# ```python
# 
#     # Filter the data for the recession period and specific cities
#     recession_data = data[data['Recession'] == 1]
# 
#     # Calculate the total sales by city
#     sales_by_city = recession_data.groupby('City')['Automobile_Sales'].sum().reset_index()
# 
#     # Create a base map centered on the United States
#     map1 = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
# 
#     # Create a choropleth layer using Folium
#     choropleth = folium.Choropleth(
#         geo_data= 'us-states.json',  # GeoJSON file with state boundaries
#         data=sales_by_city,
#         columns=['City', 'Automobile_Sales'],
#         key_on='feature.properties.name',
#         fill_color='YlOrRd',
#         fill_opacity=0.7,
#         line_opacity=0.2,
#         legend_name='Automobile Sales during Recession'
#     ).add_to(map1)
# 
# 
#     # Add tooltips to the choropleth layer
#     choropleth.geojson.add_child(
#         folium.features.GeoJsonTooltip(['name'], labels=True)
#     )
# 
#     # Display the map
#     map1
# 
# ```
# </details>
# 

# # Congratulations! You have completed the lab
# 

# ## Authors
# 

# [Dr. Pooja](author_link)
# 

# ## Change Log
# 

# |Date (YYYY-MM-DD)|Version|Changed By|Change Description|
# |-|-|-|-|
# 2023-06-17|0.2|Pooja|Initial Lab Creation|
# |2023-05-01|0.1|Shengkai|Create Lab Template|
# 

# Copyright © 2023 IBM Corporation. All rights reserved.
# 

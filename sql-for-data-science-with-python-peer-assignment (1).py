#!/usr/bin/env python
# coding: utf-8

# Step-By-Step Assignment Instructions
# Assignment Topic:
# In this assignment, you will download the datasets provided, load them into a database, write and execute SQL queries to answer the problems provided, and upload a screenshot showing the correct SQL query and result for review by your peers. A Jupyter notebook is provided in the preceding lesson to help you with the process.
# 
# This assignment involves 3 datasets for the city of Chicago obtained from the Chicago Data Portal:
# 
# 1. Chicago Socioeconomic Indicators
# 
# This dataset contains a selection of six socioeconomic indicators of public health significance and a hardship index, by Chicago community area, for the years 2008 – 2012.
# 
# 2. Chicago Public Schools
# 
# This dataset shows all school level performance data used to create CPS School Report Cards for the 2011-2012 school year.
# 
# 3. Chicago Crime Data
# 
# This dataset reflects reported incidents of crime (with the exception of murders where data exists for each victim) that occurred in the City of Chicago from 2001 to present, minus the most recent seven days.
# 
# Instructions:
# 1. Review the datasets
# 
# Before you begin, you will need to become familiar with the datasets. Snapshots for the three datasets in .CSV format can be downloaded from the following links:
# 
# Chicago Socioeconomic Indicators: Click here
# 
# Chicago Public Schools: Click here
# 
# Chicago Crime Data: Click here
# 
# NOTE: Ensure you have downloaded the datasets using the links above instead of directly from the Chicago Data Portal. The versions linked here are subsets of the original datasets and have some of the column names modified to be more database friendly which will make it easier to complete this assignment. The CSV file provided above for the Chicago Crime Data is a very small subset of the full dataset available from the Chicago Data Portal. The original dataset is over 1.55GB in size and contains over 6.5 million rows. For the purposes of this assignment you will use a much smaller sample with only about 500 rows.
# 
# 2. Load the datasets into a database
# 
# Perform this step using the LOAD tool in the Db2 console. You will need to create 3 tables in the database, one for each dataset, named as follows, and then load the respective .CSV file into the table:
# 
# CENSUS_DATA
# 
# CHICAGO_PUBLIC_SCHOOLS
# 
# CHICAGO_CRIME_DATA

# In[1]:


import csv, sqlite3

con = sqlite3.connect("RealWorldData.db")
cur = con.cursor()


# In[2]:


get_ipython().system('pip install -q pandas==1.1.5')


# In[3]:


get_ipython().run_line_magic('load_ext', 'sql')


# In[4]:


get_ipython().run_line_magic('sql', 'sqlite:///RealWorldData.db')


# In[5]:


import pandas
df = pandas.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoCensusData.csv")
df.to_sql("CENSUS_DATA", con, if_exists='replace', index=False,method="multi")

df = pandas.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoCrimeData.csv")
df.to_sql("CHICAGO_CRIME_DATA", con, if_exists='replace', index=False, method="multi")

df = pandas.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoPublicSchools.csv")
df.to_sql("CHICAGO_PUBLIC_SCHOOLS_DATA", con, if_exists='replace', index=False, method="multi")


# Query the database system catalog to retrieve table metadata

# In[6]:


get_ipython().run_line_magic('sql', "SELECT name FROM sqlite_master WHERE type='table'")


# Query the database system catalog to retrieve column metadata

# In[7]:


get_ipython().run_line_magic('sql', "SELECT count(name) FROM PRAGMA_TABLE_INFO('CHICAGO_PUBLIC_SCHOOLS_DATA');")


# Problem 1: Find the total number of crimes recorded in the CRIME table.

# In[8]:


# Rows in Crime Table
get_ipython().run_line_magic('sql', 'select COUNT(*) AS TOTAL_CRIMES from CHICAGO_CRIME_DATA')


# Problem 2: List community areas with per capita income less than 11000.

# In[9]:


get_ipython().run_line_magic('sql', 'SELECT COMMUNITY_AREA_NAME FROM CENSUS_DATA WHERE PER_CAPITA_INCOME < 11000;')


# Problem 3: List all case numbers for crimes involving minors?

# In[10]:


get_ipython().run_line_magic('sql', "SELECT DISTINCT CASE_NUMBER FROM CHICAGO_CRIME_DATA WHERE DESCRIPTION LIKE '%MINOR%'")


# Problem 4: List all kidnapping crimes involving a child?(children are not considered minors for the purposes of crime analysis)

# In[11]:


get_ipython().run_line_magic('sql', "SELECT DISTINCT CASE_NUMBER, PRIMARY_TYPE, DATE, DESCRIPTION FROM CHICAGO_CRIME_DATA WHERE PRIMARY_TYPE='KIDNAPPING'")


# 
# Problem 5: What kind of crimes were recorded at schools?

# In[12]:


get_ipython().run_line_magic('sql', "SELECT DISTINCT(PRIMARY_TYPE), LOCATION_DESCRIPTION FROM CHICAGO_CRIME_DATA WHERE LOCATION_DESCRIPTION LIKE '%SCHOOL%'")


# Problem 6: List the average safety score for all types of schools.

# In[13]:


get_ipython().run_line_magic('sql', 'SELECT "Elementary, Middle, or High School", AVG(SAFETY_SCORE) AVERAGE_SAFETY_SCORE FROM CHICAGO_PUBLIC_SCHOOLS_DATA GROUP BY "Elementary, Middle, or High School";')


# Problem 7: List 5 community areas with highest % of households below poverty line.

# In[14]:


get_ipython().run_line_magic('sql', 'SELECT COMMUNITY_AREA_NAME, PERCENT_HOUSEHOLDS_BELOW_POVERTY FROM CENSUS_DATA ORDER BY PERCENT_HOUSEHOLDS_BELOW_POVERTY DESC LIMIT 5 ;')


# Problem 8: Which community area(number) is most crime prone?

# In[15]:


get_ipython().run_cell_magic('sql', '', 'SELECT COMMUNITY_AREA_NUMBER ,COUNT(COMMUNITY_AREA_NUMBER) AS FREQUENCY\nFROM CHICAGO_CRIME_DATA \nGROUP BY COMMUNITY_AREA_NUMBER\nORDER BY COUNT(COMMUNITY_AREA_NUMBER) DESC\nLIMIT 1;')


# Problem 9: Use a sub-query to find the name of the community area with highest hardship index.

# In[16]:


get_ipython().run_line_magic('sql', 'SELECT COMMUNITY_AREA_NAME FROM  CENSUS_DATA WHERE HARDSHIP_INDEX = (SELECT MAX(HARDSHIP_INDEX) FROM CENSUS_DATA);')


# Problem 10: Use a sub-query to determine the Community Area Name with most number of crimes?

# In[17]:


get_ipython().run_cell_magic('sql', '', 'SELECT community_area_name FROM CENSUS_DATA \nWHERE COMMUNITY_AREA_NUMBER = (SELECT COMMUNITY_AREA_NUMBER FROM CHICAGO_CRIME_DATA \n    GROUP BY COMMUNITY_AREA_NUMBER\n    ORDER BY COUNT(COMMUNITY_AREA_NUMBER) DESC\n    LIMIT 1)\nLIMIT 1;')


# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

# 

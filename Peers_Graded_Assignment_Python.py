#!/usr/bin/env python
# coding: utf-8

# <h1><strong>Assignment Topic</strong>:</h1>
# 
# <h2>I was provided with an empty Jupyterlite notebook which I launched in the course, to complete this assignment. I needed to include a combination of markdown and code cells. I did not needed to use the Markdown cheat sheet to help me determine the appropriate syntax for my markdown.</h2>
# 
# <h3>Guidelines for the submission:</h3> <br>
# <h4>There are a total of 25 points possible for this final project.</h4>
# 
# <h5>I hope I will succeed, based on the following Exercises:</h5>
# 
# <ol>
# 
# <li>Exercise 2 - Create a markdown cell with the title of the notebook. (1 pt)</li>
# 
# <li>Exercise 3 - Create a markdown cell for an introduction. (1 pt)</li>
# 
# <li>Exercise 4 - Create a markdown cell to list data science languages. (3 pts)</li>
# 
# <li>Exercise 5 - Create a markdown cell to list data science libraries. (3 pts)</li>
# 
# <li>Exercise 6 - Create a markdown cell with a table of Data Science tools. (3 pts)</li>
# 
# <li>Exercise 7 - Create a markdown cell introducing arithmetic expression examples. (1 pt)</li>
# 
# <li>Exercise 8 - Create a code cell to multiply and add numbers.(2 pts)</li>
# 
# <li>Exercise 9 - Create a code cell to convert minutes to hours. (2 pts)</li>
# 
# <li>Exercise 10 -Insert a markdown cell to list Objectives. (3 pts)</li>
# 
# <li>Exercise 11 - Create a markdown cell to indicate the Author’s name. (2 pts)</li>
# 
# <li>Exercise 12 - Share your notebook through GitHub (3 pts)</li>
# 
# <li>Exercise 13 - Take a screenshot of the first page of the notebook. (1 pt)</li>
#     
# </ol>

# <h1>Exercise 1: Create a Jupyter Notebook</h1>

# <h2><strong>Name:</strong> Peers-Graded_Assignment_Python</h2>

# <strong>Exercise 2 - Create a markdown cell with the title of the notebook.</strong>

# <h1>Data Science Tools and Ecosystem</h1>

# <strong>Exercise 3 - Create a markdown cell for an introduction. (1 pt)</strong>

# <p>In this notebook, Data Science Tools and Ecosystem are summarized.<p>

# <strong>Objectives:</strong>
# <ul>
#     <li>List popular languages that Data Scientists use.</li>
#     <li>List commonly used libraries used by Data Scientists.</li>
#     <li>Comment on Data Science tools.</li>
# </ul>

# <strong>Exercise 4 - Create a markdown cell to list data science languages. (3 pts)</strong>

# Some of the popular languages that Data Scientists use are:
# <ol>
#     <li>Python.</li>
#     <li>R.</li>
#     <li>SQL.</li>
#     <li>Java.</li>
#     <li>Julia.</li>
#     <li>Scala.</li>
#     <li>C/C++.</li>
#     <li>JavaScript.</li>
# </ol>

# <strong>Exercise 5 - Create a markdown cell to list data science libraries. (3 pts)</strong>

# Some of the commonly used libraries used by Data Scientists include:
# <ol>
#     <li>TensorFlow.</li>
#     <li>NumPy.</li>
#     <li>SciPy.</li>
#     <li>Pandas.</li>
#     <li>Matplotlib.</li>
#     <li>Keras.</li>
#     <li>SciKit-Learn.</li>
#     <li>PyTorch.</li>
#     <li>Scrapy.</li>
#     <li>BeautifulSoup.</li>
#     <li>LightGBM.</li>
#     <li>ELI5.</li>
#     <li>Theano.</li>
#     <li>NuPIC.</li>
#     <li>Ramp.</li>
#     <li>Pipenv.</li>
#     <li>Bob.</li>
#     <li>PyBrain.</li>
#     <li>Caffe2.</li>
#     <li>Chainer.</li>
# </ol>

# <strong>Exercise 6 - Create a markdown cell with a table of Data Science tools. (3 pts)</strong>

# Data Science Tools:
# 
# <table style="width:100%">
#   <tr>
#     <th>Data Science Tools</th>
#     <th></th>
#     <th></th>
#   </tr>
#   <tr>
#     <td>SAS. It is one of those data science tools which are specifically designed for statistical operation</td>
#     <td></td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>Apache Spark</td>
#     <td></td>
#     <td></td>
#   </tr>
#   <tr>
#     <td>BigML</td>
#     <td></td>
#     <td></td>
#   </tr>
# </table>

# <strong>Exercise 7 - Create a markdown cell introducing arithmetic expression examples. (1 pt)</strong>

# <h3>Below are a few examples of evaluating arithmetic expressions in Python</h3>

# In[2]:


# Arithmetic operations
code = compile("5 + 4", "<string>", "eval")
eval(code)
# Result: 9


# In[3]:


code1 = compile("(5 + 7) * 2", "<string>", "eval")
eval(code1)
# Result: 24


# In[4]:


import math
# Volume of a sphere
code2 = compile("4 / 3 * math.pi * math.pow(25, 3)", "<string>", "eval")
eval(code2)
# Result: 65449.84694978735


# <strong>Exercise 8 - Create a code cell to multiply and add numbers.(2 pts)</strong>

# This a simple arithmetic expression to mutiply then add integers

# In[5]:


(3*4)+5
# Result: 17


# <strong>Exercise 9 - Create a code cell to convert minutes to hours. (2 pts)</strong>

# This will convert 200 minutes to hours by diving by 60

# In[6]:


days = 0
hours = 0
mins = 0

time = 200
#days = time / 1440
leftover_minutes = time % 1440
hours = leftover_minutes / 60
#mins = time - (days*1440) - (hours*60)
print(str(days) + " days, " + str(hours) + " hours, " + str(mins) +  " mins. ")

# Result: 3.3333333333333335 hours


# <strong>Exercise 10 -Insert a markdown cell to list Objectives.</strong>

# <p>Below the introduction cell created in Exercise 3, insert a new markdown cell to list the objectives that this notebook covered (i.e. some of the key takeaways from the course). In this new cell start with an introductory line titled: Objectives: in bold font. Then using an unordered list (bullets) indicate 3 to 5 items covered in this notebook, such as List popular languages for Data Science.</p>

# <strong>Exercise 11 - Create a markdown cell to indicate the Author’s name.</strong>

# <h2>Author:</h2> Kanchan Sonkar

# In[ ]:





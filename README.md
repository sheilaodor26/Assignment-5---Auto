# Exploring Categorical Feature Encoding Techniques in Car Analysis: A Python Project

The "Exploring Categorical Feature Encoding Techniques in Car Analysis" project aims to equip learners with fundamental skills in preprocessing categorical data for analysis using Python. Through a comprehensive exploration of replacing values, encoding labels, and one-hot encoding techniques, participants will gain insights into effectively transforming categorical features into numeric representations. By applying these techniques specifically to a dataset on cars, participants will understand the importance of preprocessing categorical data for machine learning tasks. Through hands-on examples and practical exercises, this project provides a solid foundation for individuals looking to enhance their data preprocessing skills and apply them to real-world datasets.

# Getting Started

## Prerequisites
- Python 3.x
- pandas
- numpy

## Installing 
clone the GitHub repository to your local machine

'```bash
git clone https://github.com/your_username/your_repository.git'
#Import Python Libraries
import numpy as np
#import scipy as sp
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt


### install the required python package

pip install pandas numpy

# Running the Tests
## Breakdown of Tests
### Part 1 - Replacing Values:

Load the dataset.
Use the 'replace' function to replace categorical values with numerical values.

df = pd.read_csv('auto.csv')
#### To see what the data set looks like, we'll use the head() method.
df.head()
#### Lets list the data types for each column
df.dtypes

#### assign each categpry the numarical value you wish. 
#### In this exmale I assign 1 to gas and 0 to diesel. We use the dictionary type in pandas to do such operation.
map_cat = {'fuel-type': {'gas': 1, 'diesel':0}}

#### BE SURE TO CONVERT THE COLUMN TYPE TO CATEGORY, FOR FASTER OPERATION AND BETTER PERFORMANCE.
df['fuel-type'] = df['fuel-type'].astype('category')
df['fuel-type'].dtype

#### Apply the replace function to get the desired result
df.replace(map_cat, inplace=False)

df[['num-of-doors']].value_counts()

df['num-of-doors']=df['num-of-doors'].astype('category')
map_cat1 = {'num-of-doors': {'four': 4, 'two':2}}
df.replace(map_cat1,inplace=True)
df.head()


### Part 2 : Label Encoding
Load the dataset.
Use the 'cat.codes' method to perform label encoding on a categorical column.

#### load the data file again or clear all output cell>>all output>>clear
df1 = pd.read_csv('auto.csv')

#### a very straight forward method is to use cat codes method 
df1['num-of-doors'] = df1['num-of-doors'].astype('category')
df1['num-of-doors'] = df1['num-of-doors'].cat.codes
df1.head()
### Part 3 : One-Hot encoding 

Load the dataset.
Use the 'get_dummies' function to perform one-hot encoding on a categorical column.
Concatenate the original dataframe with the one-hot encoded variables.

#### load the data file again or clear all output cell>>all output>>clear
df2 = pd.read_csv('auto.csv')

#### get indicator variables and assign it to data frame "dummy_variable_1" 
dummy_variable_1 = pd.get_dummies(df2["fuel-type"])
dummy_variable_1.head()

#### merge data frame "df" and "dummy_variable_1" 
df3 = pd.concat([df2, dummy_variable_1], axis=1)

#### drop original column "fuel-type" from "df"
df3.drop("fuel-type", axis = 1, inplace=True)
df3.head()
# Deployment
This project is a demonstration of encoding techniques and can be further extended and integrated into larger data preprocessing pipelines or machine learning workflows.

# Author
Sheila Odor

# License
This project is licensed under the MIT License.

# Acknowledgement
Data provided by UCI Machine Learning Repository


Make sure to replace `"your_username"` and `"your_repository"` with your actual GitHub username and repository name, respectively. Additionally, you may need to adjust the package names and dependencies based on your actual implementation.

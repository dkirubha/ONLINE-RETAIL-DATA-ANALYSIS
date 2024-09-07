#!/usr/bin/env python
# coding: utf-8

# # Portfolio Project: Online Retail Exploratory Data Analysis with Python

# ## Overview
# 
# In this project, you will step into the shoes of an entry-level data analyst at an online retail company, helping interpret real-world data to help make a key business decision.

# ## Case Study
# In this project, you will be working with transactional data from an online retail store. The dataset contains information about customer purchases, including product details, quantities, prices, and timestamps. Your task is to explore and analyze this dataset to gain insights into the store's sales trends, customer behavior, and popular products. 
# 
# By conducting exploratory data analysis, you will identify patterns, outliers, and correlations in the data, allowing you to make data-driven decisions and recommendations to optimize the store's operations and improve customer satisfaction. Through visualizations and statistical analysis, you will uncover key trends, such as the busiest sales months, best-selling products, and the store's most valuable customers. Ultimately, this project aims to provide actionable insights that can drive strategic business decisions and enhance the store's overall performance in the competitive online retail market.
# 
# ## Prerequisites
# 
# Before starting this project, you should have some basic knowledge of Python programming and Pandas. In addition, you may want to use the following packages in your Python environment:
# 
# - pandas
# - numpy
# - seaborn
# - matplotlib
# 
# These packages should already be installed in Coursera's Jupyter Notebook environment, however if you'd like to install additional packages that are not included in this environment or are working off platform you can install additional packages using `!pip install packagename` within a notebook cell such as:
# 
# - `!pip install pandas`
# - `!pip install matplotlib`

# ## Project Objectives
# 1. Describe data to answer key questions to uncover insights
# 2. Gain valuable insights that will help improve online retail performance
# 3. Provide analytic insights and data-driven recommendations

# ## Dataset
# 
# The dataset you will be working with is the "Online Retail" dataset. It contains transactional data of an online retail store from 2010 to 2011. The dataset is available as a .xlsx file named `Online Retail.xlsx`. This data file is already included in the Coursera Jupyter Notebook environment, however if you are working off-platform it can also be downloaded [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx).
# 
# The dataset contains the following columns:
# 
# - InvoiceNo: Invoice number of the transaction
# - StockCode: Unique code of the product
# - Description: Description of the product
# - Quantity: Quantity of the product in the transaction
# - InvoiceDate: Date and time of the transaction
# - UnitPrice: Unit price of the product
# - CustomerID: Unique identifier of the customer
# - Country: Country where the transaction occurred

# ## Tasks
# 
# You may explore this dataset in any way you would like - however if you'd like some help getting started, here are a few ideas:
# 
# 1. Load the dataset into a Pandas DataFrame and display the first few rows to get an overview of the data.
# 2. Perform data cleaning by handling missing values, if any, and removing any redundant or unnecessary columns.
# 3. Explore the basic statistics of the dataset, including measures of central tendency and dispersion.
# 4. Perform data visualization to gain insights into the dataset. Generate appropriate plots, such as histograms, scatter plots, or bar plots, to visualize different aspects of the data.
# 5. Analyze the sales trends over time. Identify the busiest months and days of the week in terms of sales.
# 6. Explore the top-selling products and countries based on the quantity sold.
# 7. Identify any outliers or anomalies in the dataset and discuss their potential impact on the analysis.
# 8. Draw conclusions and summarize your findings from the exploratory data analysis.

# ## Task 1: Load the Data

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[2]:


df = pd.read_excel("Online Retail.xlsx")
# To display the top 5 rows
df.head(5)


# In[3]:


df.dtypes


# In[4]:


num_of_rows = len(df)

print(f"The number of rows is {num_of_rows}")


# In[5]:


df.describe()


# # DATA CLEANING

# In[6]:


# Check for missing values in each column
df.isnull().sum()


# In[7]:


# First, define df_cleaned from the original dataframe df
df_cleaned = df.copy()

# Drop rows where CustomerID is missing
df_cleaned = df_cleaned.dropna(subset=['CustomerID'])

# Verify that CustomerID missing values are handled
print(df_cleaned.isnull().sum())


# # SUMMARY STATS

# In[8]:


# Summary statistics for numerical columns
summary_stats = df_cleaned.describe()
print(summary_stats)


# In[9]:


# Calculate mean
mean_quantity = df_cleaned['Quantity'].mean()
mean_unitprice = df_cleaned['UnitPrice'].mean()

# Calculate median
median_quantity = df_cleaned['Quantity'].median()
median_unitprice = df_cleaned['UnitPrice'].median()

# Calculate mode for categorical data
mode_stockcode = df_cleaned['StockCode'].mode()[0]
mode_country = df_cleaned['Country'].mode()[0]

print(f"Mean Quantity: {mean_quantity}")
print(f"Median Quantity: {median_quantity}")
print(f"Mode StockCode: {mode_stockcode}")
print(f"Mode Country: {mode_country}")


# In[10]:


# Standard Deviation
std_quantity = df_cleaned['Quantity'].std()
std_unitprice = df_cleaned['UnitPrice'].std()

# Variance
var_quantity = df_cleaned['Quantity'].var()
var_unitprice = df_cleaned['UnitPrice'].var()

# Range
range_quantity = df_cleaned['Quantity'].max() - df_cleaned['Quantity'].min()
range_unitprice = df_cleaned['UnitPrice'].max() - df_cleaned['UnitPrice'].min()

# Interquartile Range (IQR)
iqr_quantity = df_cleaned['Quantity'].quantile(0.75) - df_cleaned['Quantity'].quantile(0.25)
iqr_unitprice = df_cleaned['UnitPrice'].quantile(0.75) - df_cleaned['UnitPrice'].quantile(0.25)

print(f"Standard Deviation of Quantity: {std_quantity}")
print(f"Variance of Unit Price: {var_unitprice}")
print(f"Range of Quantity: {range_quantity}")
print(f"Interquartile Range of Unit Price: {iqr_unitprice}")


# # DATA VISUALIZATION

# In[11]:


# Find the number of unique values in each column
unique_values = df_cleaned.nunique()

# Display the result
print(unique_values)


# # Top 10 Products Sold

# In[12]:


# Group by 'Description' and sum the 'Quantity'
top_products = df_cleaned.groupby('Description')['Quantity'].sum().nlargest(10)

# Plot the top 10 products
plt.figure(figsize=(12, 8))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
plt.title('Top 10 Products by Quantity Sold')
plt.xlabel('Quantity Sold')
plt.ylabel('Product Description')
plt.show()


# # Top 10 Countries by Quantity Sold

# In[13]:


# Group by 'Country' and sum the 'Quantity'
top_countries = df_cleaned.groupby('Country')['Quantity'].sum().nlargest(10)

# Plot the top 10 countries
plt.figure(figsize=(12, 8))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='plasma')
plt.title('Top 10 Countries by Quantity Sold')
plt.xlabel('Quantity Sold')
plt.ylabel('Country')
plt.show()


# # Quantity Sold Over Time

# In[14]:


# Convert InvoiceDate to datetime if needed
df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])

# Group by date and calculate total quantity sold
daily_sales = df_cleaned.groupby(df_cleaned['InvoiceDate'].dt.date)['Quantity'].sum()

# Plot sales over time
plt.figure(figsize=(12, 6))
daily_sales.plot()
plt.title('Quantity Sold Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.grid(True)
plt.show()


# # Top 10 Customers by Quantity Sold

# In[15]:


# Group by CustomerID and sum the Quantity
top_customers = df_cleaned.groupby('CustomerID')['Quantity'].sum().nlargest(10)

# Create a bar plot for the top 10 customers
plt.figure(figsize=(10, 6))
sns.barplot(x=top_customers.index, y=top_customers.values)
plt.title('Top 10 Customers by Quantity Purchased')
plt.xlabel('Customer ID')
plt.ylabel('Quantity Purchased')
plt.xticks(rotation=45)
plt.show()


# In[16]:


# Correlation matrix
corr_matrix = df_cleaned[['Quantity', 'UnitPrice']].corr()

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Quantity and Unit Price')
plt.show()


# # Quantity Sold per Month

# In[17]:


# Ensure InvoiceDate is in datetime format
df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])

# Extract the month from InvoiceDate
df_cleaned['Month'] = df_cleaned['InvoiceDate'].dt.month

# Group by month and sum the quantity sold
monthly_sales = df_cleaned.groupby('Month')['Quantity'].sum()

# Plot the monthly sales
plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_sales.index, y=monthly_sales.values, palette='Blues_d')
plt.title('Total Quantity Sold per Month')
plt.xlabel('Month')
plt.ylabel('Quantity Sold')
plt.show()


# # Quantity Sold per week

# In[18]:


# Extract the day of the week name from InvoiceDate
df_cleaned['DayOfWeek'] = df_cleaned['InvoiceDate'].dt.day_name()

# Group by day of the week and sum the quantity sold
weekday_sales = df_cleaned.groupby('DayOfWeek')['Quantity'].sum()

# Sort days of the week for correct ordering
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Plot the sales by day of the week
plt.figure(figsize=(10, 6))
sns.barplot(x=weekday_sales.index, y=weekday_sales.values, order=day_order, palette='coolwarm')
plt.title('Total Quantity Sold per Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Quantity Sold')
plt.show()


# In[19]:


# Calculate z-scores for Quantity
from scipy import stats

z_scores_quantity = stats.zscore(df_cleaned['Quantity'].dropna())
outliers_quantity = df_cleaned.loc[abs(z_scores_quantity) > 3, 'Quantity']

# Box plot for Quantity
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_cleaned['Quantity'])
plt.title('Box Plot of Quantity Sold')
plt.xlabel('Quantity Sold')
plt.show()

print(f"Number of outliers in Quantity Sold: {len(outliers_quantity)}")



# Calculate z-scores for UnitPrice
z_scores_price = stats.zscore(df_cleaned['UnitPrice'].dropna())
outliers_price = df_cleaned.loc[abs(z_scores_price) > 3, 'UnitPrice']

# Box plot for UnitPrice
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_cleaned['UnitPrice'])
plt.title('Box Plot of Unit Prices')
plt.xlabel('Unit Price')
plt.show()

print(f"Number of outliers in Unit Price: {len(outliers_price)}")


# In[20]:


# Highlighting outliers in a scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df_cleaned['InvoiceDate'], y=df_cleaned['UnitPrice'], hue=(abs(z_scores_price) > 3), palette={True: 'red', False: 'blue'})
plt.title('Unit Price with Outliers Highlighted')
plt.xlabel('Invoice Date')
plt.ylabel('Unit Price')
plt.show()


# In[21]:


# Highlighting outliers in a scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df_cleaned['InvoiceDate'], y=df_cleaned['Quantity'], hue=(abs(z_scores_quantity) > 3), palette={True: 'red', False: 'blue'})
plt.title('Quantity Sold with Outliers Highlighted')
plt.xlabel('Invoice Date')
plt.ylabel('Quantity Sold')
plt.show()


# In[22]:


from scipy import stats

# Calculate Z-scores for Quantity
z_scores_quantity = stats.zscore(df_cleaned['Quantity'].dropna())

# Create a boolean mask for quantity outliers
outlier_mask_quantity = abs(z_scores_quantity) > 3

# Filter out the outliers from the dataset
df_no_outliers_quantity = df_cleaned[~outlier_mask_quantity]

# Verify the number of remaining rows
print(f"Number of rows after excluding quantity outliers: {df_no_outliers_quantity.shape[0]}")


# In[23]:


# Calculate Z-scores for UnitPrice
z_scores_price = stats.zscore(df_no_outliers_quantity['UnitPrice'].dropna())

# Create a boolean mask for unit price outliers
outlier_mask_price = abs(z_scores_price) > 3

# Filter out the outliers from the dataset
df_no_outliers_price = df_no_outliers_quantity[~outlier_mask_price]

# Verify the number of remaining rows
print(f"Number of rows after excluding unit price outliers: {df_no_outliers_price.shape[0]}")


# # Without OUTLIERS

# In[24]:


# Recalculate basic statistics for Quantity and UnitPrice
print("Basic statistics after removing outliers:")
print(df_no_outliers_price[['Quantity', 'UnitPrice']].describe())


# In[28]:



# Histogram of Unit Prices
plt.figure(figsize=(10, 6))
plt.hist(df_no_outliers_price['UnitPrice'], bins=50, edgecolor='k')
plt.title('Distribution of Unit Prices (After Removing Outliers)')
plt.xlabel('Unit Price')
plt.ylabel('Frequency')
plt.xlim(0, 25)  # Set x-axis limits
plt.show()


# In[31]:


# Bar plot for Top 10 Products
plt.figure(figsize=(10, 6))
top_10_products.plot(kind='bar', color='skyblue')
plt.title('Top 10 Products Sold by Quantity')
plt.xlabel('StockCode')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.show()


# In[32]:


# Bar plot for Top 10 Customers
plt.figure(figsize=(10, 6))
top_10_customers.plot(kind='bar', color='lightgreen')
plt.title('Top 10 Customers by Quantity Sold')
plt.xlabel('CustomerID')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.show()


# In[33]:


import matplotlib.pyplot as plt

# Ensure that 'InvoiceDate' is in datetime format
df_no_outliers_price['InvoiceDate'] = pd.to_datetime(df_no_outliers_price['InvoiceDate'])

# Group by date and sum the quantities
quantity_sold_over_time = df_no_outliers_price.groupby(df_no_outliers_price['InvoiceDate'].dt.date)['Quantity'].sum()

# Plot the result
plt.figure(figsize=(12, 6))
plt.plot(quantity_sold_over_time.index, quantity_sold_over_time.values, color='blue')
plt.title('Quantity Sold Over Time (After Removing Outliers)')
plt.xlabel('Date')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[ ]:





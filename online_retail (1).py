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

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[3]:


df = pd.read_excel("Online Retail.xlsx")
# To display the top 5 rows
df.head(5)


# In[4]:


df.dtypes
# Find the number of unique values in each column
unique_values = df.nunique()

# Display the result
print(unique_values)


# In[50]:



# Check for negative values in numeric columns
# Assuming 'Quantity' and 'UnitPrice' are numeric columns to check
negative_values_quantity = df[df['Quantity'] < 0]
negative_values_unitprice = df[df['UnitPrice'] < 0]

# Data cleaning

# Replace negative values with 0 for 'Quantity' and 'UnitPrice'
df['Quantity'] = df['Quantity'].clip(lower=0)
df['UnitPrice'] = df['UnitPrice'].clip(lower=0)

# Drop rows where essential columns have NaN values
# Drop rows where 'Description' or 'CustomerID' are NaN
df.dropna(subset=['Description', 'CustomerID'], inplace=True)

# Drop rows where 'InvoiceNo' or 'StockCode' are NaN (if necessary)
df.dropna(subset=['InvoiceNo', 'StockCode'], inplace=True)


print("\nSummary of null values after cleaning:")
print(df.isnull().sum())


# In[6]:


num_of_rows = len(df)

print(f"The number of rows is {num_of_rows}")


# In[7]:


df.describe()


# # DATA CLEANING

# In[8]:


# Check for missing values in each column
df.isnull().sum()


# # SUMMARY STATS

# In[9]:


# Summary statistics for numerical columns
summary_stats = df.describe()
print(summary_stats)


# In[10]:


# Calculate mean
mean_quantity = df['Quantity'].mean()
mean_unitprice = df['UnitPrice'].mean()

# Calculate median
median_quantity = df['Quantity'].median()
median_unitprice = df['UnitPrice'].median()

# Calculate mode for categorical data
mode_stockcode = df['StockCode'].mode()[0]
mode_country = df['Country'].mode()[0]

print(f"Mean Quantity: {mean_quantity}")
print(f"Median Quantity: {median_quantity}")
print(f"Mode StockCode: {mode_stockcode}")
print(f"Mode Country: {mode_country}")


# In[11]:


# Standard Deviation
std_quantity = df['Quantity'].std()
std_unitprice = df['UnitPrice'].std()

# Variance
var_quantity = df['Quantity'].var()
var_unitprice = df['UnitPrice'].var()

# Range
range_quantity = df['Quantity'].max() - df['Quantity'].min()
range_unitprice = df['UnitPrice'].max() - df['UnitPrice'].min()

# Interquartile Range (IQR)
iqr_quantity = df['Quantity'].quantile(0.75) - df['Quantity'].quantile(0.25)
iqr_unitprice = df['UnitPrice'].quantile(0.75) - df['UnitPrice'].quantile(0.25)

print(f"Standard Deviation of Quantity: {std_quantity}")
print(f"Variance of Unit Price: {var_unitprice}")
print(f"Range of Quantity: {range_quantity}")
print(f"Interquartile Range of Unit Price: {iqr_unitprice}")


# # DATA VISUALIZATION

# In[12]:


# Find the number of unique values in each column
unique_values = df.nunique()

# Display the result
print(unique_values)


# # Top 10 Products Sold

# In[13]:


# Group by 'Description' and sum the 'Quantity'
top_products = df.groupby('Description')['Quantity'].sum().nlargest(10)

# Plot the top 10 products
plt.figure(figsize=(12, 8))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
plt.title('Top 10 Products by Quantity Sold')
plt.xlabel('Quantity Sold')
plt.ylabel('Product Description')
plt.show()


# # Top 10 Countries by Quantity Sold

# In[48]:


# Group by 'Country' and sum the 'Quantity'
top_countries = df.groupby('Country')['Quantity'].sum().nlargest(10)

# Plot the top 10 countries
plt.figure(figsize=(12, 8))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='plasma')
plt.title('Top 10 Countries by Quantity Sold')
plt.xlabel('Quantity Sold')
plt.ylabel('Country')
plt.show()


# # Quantity Sold Over Time

# In[47]:


# Convert InvoiceDate to datetime if needed
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Group by date and calculate total quantity sold
daily_sales = df.groupby(df['InvoiceDate'].dt.date)['Quantity'].sum()

# Plot sales over time
plt.figure(figsize=(12, 6))
daily_sales.plot()
plt.title('Quantity Sold Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.grid(True)
plt.show()


# In[46]:


# Convert InvoiceDate to datetime if it isn't already
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Extract month from the InvoiceDate
df['Month'] = df['InvoiceDate'].dt.to_period('M')

# Group by month and sum Quantity
monthly_sales = df.groupby('Month')['Quantity'].sum()

# Plot the monthly sales trends
plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='line')
plt.title('Monthly Sales Trends (Quantity Sold)')
plt.xlabel('Month')
plt.ylabel('Total Quantity Sold')
plt.grid(True)
plt.show()


# # Top 10 Customers by Quantity Sold

# In[45]:


# Calculate total revenue for each customer
customer_revenue = df.groupby('CustomerID')['TotalRevenue'].sum()

# Get the top 10 customers by total revenue
top_customers = customer_revenue.nlargest(10)

# Plot top 10 customers by revenue with coolwarm colormap
plt.figure(figsize=(12, 6))
colors = cm.coolwarm(np.linspace(0, 1, len(top_customers)))
top_customers.plot(kind='bar', color=colors)  # Using a colormap
plt.title('Top 10 Most Valuable Customers (by Total Revenue)')
plt.xlabel('Customer ID')
plt.ylabel('Total Revenue')
plt.xticks(rotation=0)
plt.show()


# # Quantity Sold per Month

# In[44]:


# Ensure InvoiceDate is in datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Extract the month from InvoiceDate
df['Month'] = df['InvoiceDate'].dt.month

# Group by month and sum the quantity sold
monthly_sales = df.groupby('Month')['Quantity'].sum()

# Plot the monthly sales
plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_sales.index, y=monthly_sales.values, palette='Blues_d')
plt.title('Total Quantity Sold per Month')
plt.xlabel('Month')
plt.ylabel('Quantity Sold')

plt.show()


# # Quantity Sold per week

# In[43]:


# Extract the day of the week name from InvoiceDate
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()

# Group by day of the week and sum the quantity sold
weekday_sales = df.groupby('DayOfWeek')['Quantity'].sum()

# Sort days of the week for correct ordering
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Plot the sales by day of the week
plt.figure(figsize=(10, 6))
sns.barplot(x=weekday_sales.index, y=weekday_sales.values, order=day_order, palette='coolwarm')
plt.title('Total Quantity Sold per Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Quantity Sold')
plt.show()


# In[22]:


from scipy import stats

# Calculate Z-scores for Quantity
z_scores_quantity = stats.zscore(df['Quantity'].dropna())

# Create a boolean mask for quantity outliers
outlier_mask_quantity = abs(z_scores_quantity) > 3

# Filter out the outliers from the dataset
df_no_outliers_quantity = df[~outlier_mask_quantity]

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


# In[25]:



# Histogram of Unit Prices
plt.figure(figsize=(10, 6))
plt.hist(df_no_outliers_price['UnitPrice'], bins=50, edgecolor='k')
plt.title('Distribution of Unit Prices (After Removing Outliers)')
plt.xlabel('Unit Price')
plt.ylabel('Frequency')
plt.xlim(0, 25)  # Set x-axis limits
plt.show()


# In[53]:


# Group by 'StockCode' and sum the 'Quantity' to get total quantity sold for each product
product_quantity = df.groupby('StockCode')['Quantity'].sum()

# Get the top 10 products by total quantity sold
top_10_products = product_quantity.nlargest(10)


# Bar plot for Top 10 Products
plt.figure(figsize=(10, 6))
top_10_products.plot(kind='bar', color='skyblue')
plt.title('Top 10 Products Sold by Quantity')
plt.xlabel('StockCode')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[56]:


# The top 10 customers
customer_quantity = df_no_outliers_price.groupby('CustomerID')['Quantity'].sum()

# Sort customers by the total quantity sold and get the top 10
top_10_customers = customer_quantity.nlargest(10)

# Plot the result
plt.figure(figsize=(10, 6))
top_10_customers.plot(kind='bar', color='lightgreen')
plt.title('Top 10 Customers by Quantity Sold')
plt.xlabel('Customer ID')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[57]:


# Filter out outliers in 'UnitPrice'
# Let's assume the outliers are values beyond 99th percentile
q_low = df['UnitPrice'].quantile(0.01)  # 1st percentile
q_high = df['UnitPrice'].quantile(0.99)  # 99th percentile

# Create a DataFrame without the extreme outliers in UnitPrice
df_no_outliers_price = df[(df['UnitPrice'] >= q_low) & (df['UnitPrice'] <= q_high)]

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
plt.tight_layout()
plt.show()


# In[58]:



df['TotalRevenue'] = df['Quantity'] * df['UnitPrice']

# Calculate total revenue for each country
country_revenue = df.groupby('Country')['TotalRevenue'].sum()

# Get the top 10 countries by total revenue
top_countries = country_revenue.nlargest(10)

# Plot top 10 countries by total revenue
plt.figure(figsize=(12, 6))
ax = top_countries.plot(kind='bar', color='skyblue')

# Add total revenue labels on the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():,.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.title('Top 10 Countries by Total Revenue')
plt.xlabel('Country')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[59]:




# Filter the dataset for transactions in the UK
uk_data = df[df['Country'] == 'United Kingdom']

# Ensure InvoiceDate is in datetime format
uk_data['InvoiceDate'] = pd.to_datetime(uk_data['InvoiceDate'])

# Extract year and month from the InvoiceDate
uk_data['YearMonth'] = uk_data['InvoiceDate'].dt.to_period('M')

# Calculate total revenue for each month
monthly_sales = uk_data.groupby('YearMonth')['TotalRevenue'].sum()

# Convert Series to DataFrame for easier plotting
monthly_sales = monthly_sales.reset_index()

# Plot monthly sales for the UK
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['YearMonth'].astype(str), monthly_sales['TotalRevenue'], marker='o', linestyle='-')
plt.title('Monthly Sales for the UK')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[32]:




# Filter the dataset for transactions in the UK
uk_data = df[df['Country'] == 'United Kingdom']

# Calculate total revenue for each customer
customer_revenue_uk = uk_data.groupby('CustomerID')['TotalRevenue'].sum()

# Calculate total revenue for each product
product_revenue_uk = uk_data.groupby('Description')['TotalRevenue'].sum()

# Get the top 10 customers by total revenue
top_customers_uk = customer_revenue_uk.nlargest(10)

# Get the top 10 products by total revenue
top_products_uk = product_revenue_uk.nlargest(10)

# Plot top 10 customers by total revenue
plt.figure(figsize=(12, 6))
top_customers_uk.plot(kind='bar', color='skyblue')
plt.title('Top 10 Customers by Total Revenue (UK)')
plt.xlabel('Customer ID')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45, ha='right')
plt.show()

# Plot top 10 products by total revenue
plt.figure(figsize=(12, 8))
top_products_uk.plot(kind='barh', color='lightgreen')
plt.title('Top 10 Products by Total Revenue (UK)')
plt.xlabel('Total Revenue')
plt.ylabel('Product Description')
plt.gca().invert_yaxis()  # Invert y-axis to show the highest revenue at the top
plt.show()


# In[60]:


# Assuming df_cleaned is already defined and contains 'TotalRevenue' and 'Country' columns

# Calculate total revenue for each customer across the entire dataset
customer_revenue_all = df.groupby('CustomerID')['TotalRevenue'].sum()

# Get the top 10 customers by total revenue across the entire dataset
top_customers_all = customer_revenue_all.nlargest(10)

# Filter the dataset for transactions in the UK
uk_data = df[df['Country'] == 'United Kingdom']

# Calculate total revenue for each customer in the UK
customer_revenue_uk = uk_data.groupby('CustomerID')['TotalRevenue'].sum()

# Get the top 10 customers by total revenue in the UK
top_customers_uk = customer_revenue_uk.nlargest(10)

# Find the intersection of top 10 customers in the UK and top 10 customers across the entire dataset
top_customers_uk_ids = top_customers_uk.index
top_customers_all_ids = top_customers_all.index

# Find common customer IDs
common_customers = top_customers_uk_ids.intersection(top_customers_all_ids)
num_common_customers = len(common_customers)

print(f"Number of top 10 customers in the UK who are also in the top 10 across the entire dataset: {num_common_customers}")

# Print the common customers
print("Common Customer IDs:", list(common_customers))


# In[35]:


import pandas as pd

# Assuming df_cleaned is already defined and contains 'TotalRevenue' and 'Country' columns

# Calculate total revenue for each product across the entire dataset
product_revenue_all = df.groupby('Description')['TotalRevenue'].sum()

# Get the top 10 products by total revenue across the entire dataset
top_products_all = product_revenue_all.nlargest(10)

# Filter the dataset for transactions in the UK
uk_data = df[df['Country'] == 'United Kingdom']

# Calculate total revenue for each product in the UK
product_revenue_uk = uk_data.groupby('Description')['TotalRevenue'].sum()

# Get the top 10 products by total revenue in the UK
top_products_uk = product_revenue_uk.nlargest(10)

# Find the intersection of top 10 products in the UK and top 10 products across the entire dataset
top_products_uk_names = top_products_uk.index
top_products_all_names = top_products_all.index

# Find common product names
common_products = top_products_uk_names.intersection(top_products_all_names)
num_common_products = len(common_products)

print(f"Number of top 10 products in the UK that are also in the top 10 across the entire dataset: {num_common_products}")

# Print the common products
print("Common Product Names:", list(common_products))


# In[64]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df_cleaned is already defined and contains 'TotalRevenue', 'Description', and 'InvoiceDate' columns

# Calculate total revenue for each product across the entire dataset
product_revenue_all = df.groupby('Description')['TotalRevenue'].sum()

# Get the top 10 products by total revenue across the entire dataset
top_products_all = product_revenue_all.nlargest(10).index

# Filter dataset to include only top 10 products
top_products_data = df[df['Description'].isin(top_products_all)]

# Convert 'InvoiceDate' to datetime
top_products_data['InvoiceDate'] = pd.to_datetime(top_products_data['InvoiceDate'])

# Extract month and year from the 'InvoiceDate'
top_products_data['YearMonth'] = top_products_data['InvoiceDate'].dt.to_period('M')

# Group by product and month-year, then sum the total revenue
monthly_sales = top_products_data.groupby(['Description', 'YearMonth'])['TotalRevenue'].sum().unstack().fillna(0)

# Plot sales data for each product over time
plt.figure(figsize=(14, 8))
for product in top_products_all:
    plt.plot(monthly_sales.columns.astype(str), monthly_sales.loc[product], label=product)

plt.title('Monthly Sales of Top 10 Products Over the Year')
plt.xlabel('Month-Year')
plt.ylabel('Total Revenue')
plt.legend(title='Product')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert 'YearMonth' to a string format for easier display
monthly_sales.index = monthly_sales.index.astype(str)

# Reset index to make 'Description' a column
monthly_sales.reset_index(inplace=True)

print(monthly_sales)


# In[63]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame and it contains 'TotalRevenue' column

# Calculate total revenue for each product
product_revenue = df.groupby('Description')['TotalRevenue'].sum()

# Filter out products with zero or negative total revenue
positive_product_revenue = product_revenue[product_revenue > 0]

# Get the bottom 10 products by total revenue from positive values
bottom_10_products = positive_product_revenue.nsmallest(10)

# Filter the dataset for these bottom 10 products
bottom_10_data = df[df['Description'].isin(bottom_10_products.index)]

# Add YearMonth column
bottom_10_data['YearMonth'] = bottom_10_data['InvoiceDate'].dt.to_period('M')

# Group by product and YearMonth to calculate total revenue
monthly_revenue_bottom_10 = bottom_10_data.groupby(['Description', 'YearMonth'])['TotalRevenue'].sum().unstack(fill_value=0)

# Reset index for better readability
monthly_revenue_bottom_10 = monthly_revenue_bottom_10.reset_index()

# Ensure YearMonth is in string format for plotting
monthly_revenue_bottom_10.columns = monthly_revenue_bottom_10.columns.astype(str)

# Plot monthly sales for bottom 10 products
plt.figure(figsize=(14, 10))
for product in bottom_10_products.index:
    product_data = monthly_revenue_bottom_10[monthly_revenue_bottom_10['Description'] == product]
    if not product_data.empty:
        plt.plot(product_data.columns[1:], product_data.iloc[0, 1:], label=product)

plt.title('Monthly Sales of Bottom 10 Low-Selling Products (Positive Revenue)')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Print detailed sales data for bottom 10 products
print(monthly_revenue_bottom_10)


# In[ ]:





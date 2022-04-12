#%% [markdown]
## Stout Assessment: Case Study #2
#### Yu-Wei Lai 
# 2022.04.11
#%%
%load_ext rpy2.ipython

# %%
import pandas as pd
df_customer = pd.read_csv('casestudy_q2.csv')

# %% [markdown]
## Part 1. Dataset Description
#Take a look into basic information

# %%
df_customer = df_customer.iloc[:,1:] # remove the index column
# df_customer.columns # ['customer_email', 'net_revenue', 'year']
df_customer.head()

#%%
# How many records are in the dataset
len(df_customer)
# %%
# Missing records in each columns:
df_customer.isnull().any() # no missing value in each column

# %%
# The dataset contains three years of data:
df_customer.year.unique() # [2015, 2016, 2017]

# %%
# There are 604618 unique customers in the dataset
len(df_customer.customer_email.unique()) 

# %%
# Summary for 'net_revenue'
df_customer.net_revenue.describe()

# %% [markdown]
#For the first eye of the dataset, there is no missing value issues or missed imputed in attribues that are interger or float.

#%% [markdown]
#---
## Part 2. Revenue Inoformation

# %%
# Total revenue for the current year
def yearly_revenue(df, year):
    return sum(df[df.year == year].net_revenue)
# #%%
# sum(df_customer[df_customer.year == 2015].net_revenue)
#%%
# New Customer Revenue e.g. new customers not present in previous year only
def yearly_new_revenue(df, year):
    df_yearly = df[df.year == year]
    if (year - 1) in df.year.unique():
        old_customer_set = set(df[df.year == (year-1)].customer_email.unique())
        new_customer_set = set(df_yearly.customer_email) - old_customer_set
        df_yearly_new = df_yearly[df_yearly['customer_email'].isin(new_customer_set)]
        return sum(df_yearly_new.net_revenue)
    else:
        return sum(df_yearly.net_revenue) # All customers are new for the starting year

# Existing Customer Revenue Current Year
def yearly_existing_current_year(df, year):
    if (year - 1) not in df.year.unique():
        return 0

    old_customer_set = set(df[df.year == (year-1)].customer_email.unique())
    df_old = df[df['customer_email'].isin(old_customer_set)]
    return yearly_revenue(df_old, year)

# Existing Customer Revenue Prior Year
def yearly_existing_previous_year(df, year):
    if (year - 1) not in df.year.unique():
        return 0
    df_pre_year = df[df.year == (year-1)]
    existing_customer_set = set(df[df.year == year].customer_email.unique())
    df_pre_year = df_pre_year[df_pre_year['customer_email'].isin(existing_customer_set)]
    # return yearly_revenue(df_old, (year-1))
    return sum(df_pre_year.net_revenue)

# Existing Customer Growth. 
# To calculate this, use the Revenue of existing customers for current year –(minus) Revenue of existing customers from the previous year
def yearly_existing_growth(df, year):
    if (year - 1) not in df.year.unique():
        return 0 # There is no existing customers in the first year -> return 0

    return yearly_existing_current_year(df, year) - yearly_existing_previous_year(df, year)
#%%
# Revenue lost from attrition
def revenue_attrition(df, year):
    """
    The Revenue lost by loosing the customers -> revenue from who does not return this year coonsided as the lost
    """
    if (year - 1) in df.year.unique():
        return yearly_revenue(df, (year-1)) - yearly_existing_previous_year(df, year)
    else:
        return 0 # All customers are new -> no losed customers

# Total Customers Current Year
def total_customers_current(df, year):
    return len(df[df.year == year])

# Total Customers Previous Year
def total_customers_previous(df, year):
    if (year - 1) not in df.year.unique():
        return 0
    return len(df[df.year == (year-1)])

# New Customers
def new_customers(df, year):
    """
    Calculate the number of customers who are in the current year but not in the previous year
    """
    if (year - 1) not in df.year.unique():
        return total_customers_current(df, year)

    df_yearly = df[df.year == year]
    old_customer_set = set(df[df.year == (year-1)].customer_email.unique())
    new_customer_set = set(df_yearly.customer_email) - old_customer_set
    return len(new_customer_set)

# Lost Customers
def lost_customers(df, year):
    """
    Calculate the number of customers who are in the previous year but not in the current year
    """
    if (year - 1) not in df.year.unique():
        return 0
    
    df_yearly = df[df.year == year]
    old_customer_set = set(df[df.year == (year-1)].customer_email.unique())
    lost_customer_set = old_customer_set - set(df_yearly.customer_email)
    return len(lost_customer_set)

# %% [markdown]
#### The Result for three years from 2015 to 2017
years = df_customer.year.unique()
col = ['Year', 'Total Revenue', 'New Customer Revenue', 
        'Existing Customer Growth', 'Revenue Lost From Attrition', 
        'Existing Customer Revenue (Current)', 'Existing Customer Revenue (Previous)',
        'Total Customers (Current)', 'Total Customers (Previous)',
        'New Customers', 'Lost Customers']
result = pd.DataFrame(columns=col)

for year in years:
    df = {'Year': year, 
        'Total Revenue': str(yearly_revenue(df_customer, year)), 
        'New Customer Revenue': yearly_new_revenue(df_customer, year), 
        'Existing Customer Growth': yearly_existing_growth(df_customer, year), 
        'Revenue Lost From Attrition': revenue_attrition(df_customer, year), 
        'Existing Customer Revenue (Current)': yearly_existing_current_year(df_customer, year), 
        'Existing Customer Revenue (Previous)': yearly_existing_previous_year(df_customer, year),
        'Total Customers (Current)': total_customers_current(df_customer, year), 
        'Total Customers (Previous)': total_customers_previous(df_customer, year),
        'New Customers': new_customers(df_customer, year), 
        'Lost Customers': lost_customers(df_customer, year)}
    result = result.append(df, ignore_index = True)

result

#%%
result.to_csv('case_2_result.csv')

#%% [markdown]
#---
## Part 3. Visualizations
# Due to some issues in magic cell with Jupyter, please referred to the anothere file 'visualization.html'.
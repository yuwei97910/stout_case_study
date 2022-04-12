#%% [markdown]
## Stout Assessment: Case Study #1
#### Yu-Wei Lai 
# 2022.04.11
#%%
%load_ext rpy2.ipython

# %%
from matplotlib.style import library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
#%%
df_loan = pd.read_csv('loans_full_schema.csv')

# %% [markdown]
## Part 1. Preprocessing and Dataset Description
#Take a look into basic information

# %%
df_loan.head()

# %%
len(df_loan) # 10000 records in the dataset

# %%
df_loan.columns

# %%
#%%
for column in df_loan:
    if df_loan[column].isnull().any():
       print('{}: {}'.format(column, df_loan[column].isnull().sum()))
# %% [markdown]
### Deal with missing values
#- For those columns have over the half for missing values: remove the attributes -> remove annual_income_joint, verification_income_joint, debt_to_income_joint, months_since_last_delinq, months_since_90d_late
#- 'debt_to_income': remove the records which are missing (only 24 records)
#- 'emp_title' and 'emp_length': It is common to miss both two attributes, and this might be the conditions such as unemployeed. -> impute empty string for emp_title and impute 0 for emp_length
#- 'months_since_last_credit_inquiry': impute as 0; these records might be those who never be inquiried.
#- 'num_accounts_120d_past_due': impute as 0.

#%%
df_loan.drop(columns=['annual_income_joint', 'verification_income_joint', 'debt_to_income_joint', 'months_since_last_delinq', 'months_since_90d_late'], inplace=True)
df_loan.emp_title = df_loan.emp_title.fillna('')
df_loan = df_loan.fillna(0)

# %% [markdown]
### Outliers
# There are several attributes that contains outliers
#- annual_income
# %%
# outliers and remove outliers for annual_income
plt.scatter(df_loan.annual_income, df_loan.interest_rate)

upper_limit = df_loan['annual_income'].mean() + 3 * df_loan['annual_income'].std()
lower_limit = df_loan['annual_income'].mean() - 3 * df_loan['annual_income'].std()

df_loan = df_loan[(df_loan['annual_income'] < upper_limit) & (df_loan['annual_income'] > lower_limit)]
len(df_loan)
# %% [markdown]
### Multicollinear
# Several attributes that contains same or similar information, and this would cause multicollinearity issues when building models to the onterest rate.
# We should consider to remove some properties or do the dimension reduction.
# %% [markdown]
### Other issues
#### 'emp_title'
# The employee titles contains various contents, it should be cleaned into specific categories before the analysis.

#%%
df_loan.to_csv('loans_full_schema_cleaned.csv', index=False)
# %% [markdown]
#----
## Part 2. Visualization
### Due to some issues in magic cell with Jupyter, please referred to the anothere file 'visualization.html'.

# %% [markdown]
#----
## Part 3. Modeling - make prediction to interest_rate

### Training and testing split
# %%
df_loan_x = df_loan.drop(columns=['interest_rate'])
x_train, x_test, y_train, y_test = train_test_split(df_loan_x, df_loan.interest_rate, test_size=0.3, random_state=0)

# %% [R]
%%R -i x_train -i x_test -i y_train -i y_test
fit0 = lm(y_train~.-emp_title-state, data = x_train)
summary(fit0)
#%% [markdown]
# As mentioned grade has dominant effect on predicting the interest rate. However, it would not be reasonable to put in the model. 
# The reason is that the grade attribute is forrmed by other attributes for developing the interest rate.
#%%
rm_col = ['emp_title', 'state', 'grade', 'sub_grade']
x_train = x_train.drop(columns=rm_col)
x_test = x_test.drop(columns=rm_col)

# %% 
%%R -i x_train -i x_test -i y_train -i y_test
fit1 = lm(y_train ~ ., data = x_train)
summary(fit1)
#%% [markdown]
# The Rooted Mean Squared Error:
#%%
%%R
y_pred = predict(fit1, x_test, ncomp=best.ncomp)
rmse = mean((y_pred - y_test)^2)^(1/2)
print(rmse)
#%%
%%R
plot(fit1, which=1) # residual plot
plot(fit1, which=2) # q-q plot

#%%
# The result with grade removed can still have 70% ability for prediction. 
# No serious issues by looking into the distribution of the residuals. 
# %% [markdown]
# ----
### Try a tree method: Gradient Boosting Trees
# The tree model did not perform better than the linear model.
# %%
# one-hot encode the categorical features
category_col = ['homeownership', 'verified_income', 'loan_purpose', 'application_type', 
            'issue_month', 'loan_status', 'initial_listing_status', 'disbursement_method']
df_loan_x = df_loan_x.drop(columns=rm_col)

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(df_loan_x)
df_loan_x = encoder.fit_transform(df_loan_x)
x_train, x_test, y_train, y_test = train_test_split(df_loan_x, df_loan.interest_rate, test_size=0.3, random_state=0)

# Fit the model
model_gbm = GradientBoostingRegressor(random_state=0)
model_gbm.fit(x_train, y_train)

y_pred = model_gbm.predict(x_test)
rmse = np.mean((y_pred - y_test)**2)**(1/2)

#%%
print('R-squared: ', r2_score(y_test, y_pred))
print('RMSE: ', rmse)

#%% [markdown]
# The model can only predict about 30% total sum of square, which is not better than regression. 
# The reason might because the sparse result of one-hot encoding for categorical variables.
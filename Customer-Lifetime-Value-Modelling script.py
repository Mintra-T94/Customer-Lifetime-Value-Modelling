#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[14]:


# Load the dataset
file_path = "D:\Decision tree - project CLV\Customer_Data_v2.xlsx"
df = pd.read_excel(file_path)

df.info(), df.describe(include='all'), df.isnull().sum(), df.nunique(), df.select_dtypes(include=['number']).corr()


# In[15]:


#EDA - Correlation Analysis
numeric_cols = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()


# In[16]:


# EDA - Distribution Plots
for col in ['Customer Age', 'Order Quantity', 'Unit Cost', 'Unit Price', 'Cost', 'Revenue']:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


# In[17]:


# EDA - Categorical Analysis
categorical_cols = ['Customer Gender', 'State', 'Product Category', 'Sub Category']
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
    plt.show()


# In[18]:


# Drop 'frame size' since it has too many missing values
df.drop(columns=['frame size'], inplace=True)


# In[19]:


# Create CLV Variable
clv_df = df.groupby("Customer ID").agg(
    total_revenue=("Revenue", "sum"),
    total_orders=("Order Quantity", "sum"),
    avg_order_value=("Revenue", "mean"),
    purchase_frequency=("Customer ID", "count"),
    first_purchase=("Date", "min"),
    last_purchase=("Date", "max")
).reset_index()

clv_df["customer_lifespan"] = (clv_df["last_purchase"] - clv_df["first_purchase"]).dt.days
clv_df["CLV"] = clv_df["avg_order_value"] * clv_df["purchase_frequency"] * (clv_df["customer_lifespan"] / 365)


# In[20]:


# Feature Engineering
categorical_cols = ['Customer Gender', 'State', 'Product Category', 'Sub Category', 'Product']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# In[21]:


# Split Data into Training and Testing Sets
X = clv_df.drop(columns=['Customer ID', 'CLV', 'first_purchase', 'last_purchase'])
y = clv_df['CLV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


# Normalize Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[23]:


# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("\nLinear Regression Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))


# In[24]:


# Lasso Regression (L1 Regularization)
lasso = Lasso(alpha=1.0)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
print("\nLasso Regression Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_lasso))
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("R2 Score:", r2_score(y_test, y_pred_lasso))


# In[25]:


# Decision Tree Model
dt = DecisionTreeRegressor(random_state=50)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

print("\nDecision tree Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_dt))
print("MSE:", mean_squared_error(y_test, y_pred_dt))
print("R2 Score:", r2_score(y_test, y_pred_dt))


# In[26]:


# Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("\nDecision tree Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))


# In[27]:


# Function to plot Actual vs. Predicted values
def plot_actual_vs_predicted(y_test, y_pred, model_name):
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual CLV')
    plt.ylabel('Predicted CLV')
    plt.title(f'Actual vs. Predicted CLV - {model_name}')
    plt.show()

# Generate scatter plots for different models
plot_actual_vs_predicted(y_test, y_pred_lr, "Linear Regression")
plot_actual_vs_predicted(y_test, y_pred_lasso, "Lasso Regression")
plot_actual_vs_predicted(y_test, y_pred_dt, "Decision Tree")
plot_actual_vs_predicted(y_test, y_pred_rf, "Random Forest")

print("\nData Preprocessing, Model Training, and Visualization Completed Successfully.")


# In[ ]:





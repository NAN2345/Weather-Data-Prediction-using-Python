#!/usr/bin/env python
# coding: utf-8

# # Working on real project with Python and Machine Learning
# 
#    (A part of Big Data Analysis)

# # The Weather Dataset
# 
#    Here, the Weather Dataset is a time-series dataset with per-hour information about the weather conditions at a particular        location.It records Temperature,Dew Point Temperature,Relative Humidity,Wind Speed,Visibility,Pressure,and Conditions.
#    
#   This data is available as a CSV file.We are going to analyze this dataset using the Pandas DataFrame

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv("Downloads/Weather dataset for python.csv")


# In[3]:


data


# # How to Analyze Dataframe?

# # .head()
# It shows the first N rows in he data (by default, N=5)

# In[4]:


data.head()


# # .shape
# It shows the total no. of rows and no. of columns of the dataframe

# In[5]:


data.shape


# # .index
# This attribute provides the index of the dataframe

# In[6]:


data.index


# # .Coulumns
# It shows the name of each column

# In[7]:


data.columns


# # .dtypes
# It shows the data-type of each column

# In[8]:


data.dtypes


# # .unique()
# In a column, it shows all the unique values. It can be applied in a single column only, not on the whole dataframe

# In[9]:


data['Weather'].unique()


# # .nunique()
# It shows the total no. of unique values in each column. It can be applied on a single column as well as on whole dataframe

# In[10]:


data.nunique()


# # .count
# It shows the total no. of non-null in each column. It can be applied on a single column as well as on whole dataframe

# In[11]:


data.count()


# # .value_counts
# In a column, it shows all the unique values with their count.It can be applied on single column only

# In[12]:


data['Weather'].value_counts()


# # .info()
# Provides basic information about the dataframe

# In[13]:


data.info()


# # Q. 1)Find all the unique 'Wind Speed' values in the data.

# In[14]:


data.head(2)


# In[15]:


data.nunique()


# In[16]:


data['Wind Speed_km/h'].nunique()


# In[17]:


data['Wind Speed_km/h'].unique()#Answer


# # Q. 2) Find the number of times when the 'Weather is exactly Clear'. 

# In[18]:


#value_count()
data['Weather'].value_counts()


# In[19]:


#filtering
#data.head(2)
data[data.Weather == 'Clear']


# In[20]:


#groupby()
#data.head(2)
data.groupby('Weather').get_group('Clear')#get_group fuction is used to pick a particular element from a column

#From the above three methods we can conclude that Weather is exactly clear for 1326 times


# # Q. 3) Find the number of times when the 'Wind Speed was exactly 4 km/h'. 

# In[21]:


data.head(2)


# In[22]:


data[data['Wind Speed_km/h'] == 4]
#From this result we can conclude that Wind Speed was exactly 4km/h for 474 times


# # Q. 4) Find out all the Null Values in the data.

# In[23]:


data.isnull().sum()


# In[24]:


data.notnull().sum()


# # Q. 5) Rename the column name 'Weather' of the dataframe to 'Weather Condition'.

# In[25]:


data.rename(columns={'Weather':'Weather Condition'},inplace=True)
#we use inplace=True to make the change of column name parmanent


# In[26]:


data.head(2)


# # Q. 6) What is the mean 'Visibility' ?

# In[27]:


data.Visibility_km.mean()


# # Q. 7) What is the Standard Deviation of 'Pressure' in this data?

# In[28]:


data.Press_kPa.std()


# # Q. 8) What is the Variance of 'Relative Humidity' in this data ?

# In[29]:


data['Rel Hum_%'].var()


# # Q. 9) Find all instances when 'Snow' was recorded. 

# In[30]:


#value_counts()
data['Weather Condition'].value_counts()


# In[ ]:





# In[31]:


#filtering
data[data['Weather Condition'] == 'Snow']


# In[32]:


#str.contains
data[data['Weather Condition'].str.contains('Snow')]


# # Q. 10) Find all instances when 'Wind Speed is above 24' and 'Visibility is 25'. 

# In[33]:


data[(data['Wind Speed_km/h']>24) & (data['Visibility_km'] == 25)]


# # Q. 11) What is the Mean value of each column against each 'Weather Condition ? 

# In[34]:


data.groupby('Weather Condition').mean()


# # Q. 12) What is the Minimum & Maximum value of each column against each 'Weather Condition ? 

# In[35]:


data.groupby('Weather Condition').min()


# In[36]:


data.groupby('Weather Condition').max()


# # Q. 13) Show all the Records where Weather Condition is Fog.

# In[37]:


data[data['Weather Condition'] == 'Fog']


# # Q. 14) Find all instances when 'Weather is Clear' or 'Visibility is above 40'. 

# In[38]:


data[(data['Weather Condition'] == 'Clear') | (data['Visibility_km'] > 40)]


# # Q. 15) Find all instances when : A. 'Weather is Clear' and 'Relative Humidity is greater than 50' or B. 'Visibility is above 40'

# In[39]:


data[(data['Weather Condition'] == 'Clear') & (data['Rel Hum_%'] > 50) | (data['Visibility_km'] >40)]


# In[40]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[41]:


# Set the style
sns.set_style("whitegrid")

# Plot histogram of temperature
plt.figure(figsize=(10, 6))
sns.histplot(data['Temp_C'], bins=20, color='skyblue', kde=True)
plt.title('Distribution of Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# In[ ]:





# In[42]:


# Convert 'Date/Time' column to datetime format
data['Date/Time'] = pd.to_datetime(data['Date/Time'])

# Set the style
sns.set_style("whitegrid")

# Plot temperature trend over time using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date/Time', y='Temp_C', data=data, color='purple')
plt.title('Temperature Trend Over Time')
plt.xlabel('Date/Time')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# In[43]:


# Set the style
sns.set_style("whitegrid")

# Plot wind speed vs visibility using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x='Wind Speed_km/h', y='Visibility_km', data=data, color='green')
plt.title('Wind Speed vs Visibility')
plt.xlabel('Wind Speed (km/h)')
plt.ylabel('Visibility (km)')
plt.tight_layout()
plt.show()


# In[44]:


sns.set_style("whitegrid")

# Plot bar chart for distribution of weather conditions
plt.figure(figsize=(10, 6))
sns.countplot(x='Weather Condition', data=data, palette='pastel')
plt.title('Distribution of Weather Conditions')
plt.xlabel('Weather Condition')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[107]:


# Compute the correlation matrix
corr = data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

# Add title
plt.title('Correlation Heatmap')

# Show the plot
plt.show()


# In[45]:


data


# In[94]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# In[74]:


# supress warnings
import warnings
warnings.filterwarnings("ignore")


# In[108]:


#Here we have performed linear regression to predict temperature based on other weather variables
#Features (Independent Variables):
X = data[['Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']]
#Target Variable (Dependent Variable):
y = data['Temp_C']


#importing the scaler
from sklearn.preprocessing import MinMaxScaler
mn= MinMaxScaler()
X=mn.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# # Model training using Linear Regression

# In[109]:


# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


#checking the coefficient and intercept values
m=model.coef_
print('M',m)
c=model.intercept_
print('C',c)

# Predict on the testing set
y_pred = model.predict(X_test)



# In[76]:


print("Training Accuracy :", model.score(X_train, y_train))
print("Testing Accuracy :", model.score(X_test, y_test))


# In[77]:


# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)


# In[89]:


#compare the actual values and the predicted values
y_test=list(y_test)

datapredict= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
sns.regplot(x='Actual', y='Predicted', data=datapredict)
plt.title('Actual vs Predicted')
plt.show()


# # Model training using Random Forest

# In[83]:


regressor=RandomForestRegressor(n_estimators=100)


# In[90]:


#training the model
regressor.fit(X_train,y_train)

#prediction on the Test Data
y_pred_1= regressor.predict(X_test)


# In[91]:


#Calculating the Accuracy
print("Training Accuracy :", regressor.score(X_train, y_train))
print("Testing Accuracy :", regressor.score(X_test, y_test))



# In[99]:


mae = mean_absolute_error(y_test, y_pred_1)
mse = mean_squared_error(y_test, y_pred_1)
rmse = mean_squared_error(y_test, y_pred_1, squared=False)

# Print the metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)


# In[100]:


y_test=list(y_test)
datapredict= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_1})
sns.regplot(x='Actual', y='Predicted', data=datapredict)
plt.title('Actual vs Predicted')
plt.show()


# # Model training using Decision Tree

# In[101]:


from sklearn.tree import DecisionTreeRegressor
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X,y)


# In[102]:


y_pred_2 = regr_1.predict(X_test)


# In[103]:


#Calculating the Accuracy
print("Training Accuracy :", regr_1.score(X_train, y_train))
print("Testing Accuracy :", regr_1.score(X_test, y_test))


# In[104]:


mae = mean_absolute_error(y_test, y_pred_2)
mse = mean_squared_error(y_test, y_pred_2)
rmse = mean_squared_error(y_test, y_pred_2, squared=False)

# Print the metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)


# In[106]:


y_test=list(y_test)
datapredict= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_2})
sns.regplot(x='Actual', y='Predicted', data=datapredict)
plt.title('Actual vs Predicted')
plt.show()


# In[ ]:





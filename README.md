# Project-2
Heart attack prediction
<br>

import numpy as np
<br>
import pandas as pd
<br>
import matplotlib.pyplot as plt
<br>
import seaborn as sns
<br>

df = pd.read_csv("1.healthcare-dataset-stroke-data.csv")
<br>

df.head(5)
<br>

df1 = df.drop(["id"],axis=1)
<br>

df1.head(10)
<br>

df1.info()
<br>

df1.shape #Total rows initially 5110
<br>

df1["gender"].unique() #['Male', 'Female', 'Other']
<br>
df1["work_type"].unique() # ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
<br>

df1.describe()
<br>

df1.isnull().sum() 
<br>

df1.corr()["bmi"] 
<br>
#BMI IS SUCH A THING THAT CAN'T BE USED AS AN AVERAGE VALUE, 
<br>
#BECAUSE IT IS UNIQUE DESPITE AGE ,ONLY RELATED TO WEIGHT. HENCE IT'S BETTER TO DROP THE NULL VALUES IN BMI.
<br>

df2 = df1.dropna()#contains the rows after dropping the null values
<br>

df2.shape
<br>

df2.info()
<br>

print(df2["gender"].value_counts())
<br>
print("--"*30)
<br>
print(df2["hypertension"].value_counts())
<br>
print("--"*30)
<br>
print(df2["heart_disease"].value_counts())
<br>
print("--"*30)
<br>
print(df2["ever_married"].value_counts())
<br>
print("--"*30)
<br>
print(df2["work_type"].value_counts())  
<br>
print("--"*30)
<br>
print(df2["Residence_type"].value_counts()) 
<br>
print("--"*30)
<br>
print(df2["avg_glucose_level"].value_counts()) 
<br>
print("--"*30)
<br>
print(df2["bmi"].value_counts())  
<br>
print("--"*30)
<br>
print(df2["smoking_status"].value_counts())
<br>
print("--"*30)
<br>
print(df2["stroke"].value_counts())  
<br>
df2.shape
<br>

df2.head(10)
<br>
df2.shape
<br>

df2.head()
<br>

#Now I will make changes in df2!! We will use OHE for categorical datas
<br>
dummies = pd.get_dummies(df2["gender"])
<br>
dummies.head()
<br>

df3 = pd.concat([df2,dummies],axis=1)
<br>

df3.drop(["gender"],1,inplace=True)
<br>

df3.head()
<br>

dummies1 = pd.get_dummies(df3["work_type"])
<br>
dummies1.head()
<br>
df3 = pd.concat([df3,dummies1],axis=1)
<br>

df3.drop(["work_type"],1,inplace=True)
<br>

df3.head()
<br>

# Married people and heart_stroke relation is pretty meaningless.Hence dropping the column
<br>
df3.drop(["ever_married"],axis=1,inplace=True)
<br>
df3.drop(["Residence_type"],axis=1,inplace=True) # Not required for heart_stroke_prediction
<br>

df3.head()
<br>

df3["smoking_status"].unique()
<br>

dummies2 = pd.get_dummies(df3["smoking_status"])
<br>
dummies2.head()
<br>

df3 = pd.concat([df3,dummies2],axis=1)
<br>

df3.drop(["smoking_status"],1)
<br>

df4 = df3.copy()
<br>
df4.head(5)
<br>

plt.subplot(221)
<br>
plt.rcParams["figure.figsize"]= (20,20)
<br>
plt.title("Stroke Correlation")
<br>
df4.corr()["stroke"].sort_values().plot(kind="bar")
<br>

plt.subplot(222)
<br>
plt.title("Average Glucose Correlation")
<br>
df4.corr()["avg_glucose_level"].sort_values().plot(kind="bar")
<br>

#1.Based on above Stroke has good correlation with factors such as Heart_disease,avg_glucose level,hypertension,age*
<br>
#2.glucose level shows a spike for people with: stroke,heart_disease,BMI,hypertension,age
<br>







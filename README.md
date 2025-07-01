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

#Let us plot the heat map for correlation 
<br>
plt.rcParams["figure.figsize"]=(10,10)
<br>
sns.heatmap(df4.corr(),cmap="autumn",annot=True,fmt=".1f")
<br>

df4.describe()
<br>

sns.countplot(x="children",data=df4,hue="stroke")
<br>
# no children has stroke probabilty according to data
<br>
# sns.countplot(x="Male",data=df4,hue="stroke")
<br>

df4.head()
<br>

print(df4["smoking_status"].unique())
<br>

plt.rcParams["figure.figsize"]=(15,15)
<br>
plt.subplot(221)
<br>
sns.countplot(x="smoking_status",data=df4,hue="stroke")
<br>

plt.subplot(222)
<br>
sns.countplot(x="smoking_status",data=df4,hue="heart_disease")
<br>

plt.rcParams["figure.figsize"]=(15,15)
<br>

plt.subplot(221)
<br>
plt.title("Female_stroke_Analysis")
<br>
sns.countplot(x="Female",data=df4,hue="stroke")
<br>


plt.subplot(222)
<br>
plt.title("Male_stroke_Analysis")
<br>
sns.countplot(x="Male",data=df4,hue="stroke")
<br>


plt.subplot(223)
<br>
plt.title("Female_Heart_Disease_Analysis")
<br>
sns.countplot(x="Female",data=df4,hue="heart_disease")
<br>

plt.subplot(224)
<br>
plt.title("Male_Heart_Disease_Analysis")
<br>
sns.countplot(x="Male",data=df4,hue="heart_disease")
<br>

# BASED ON THE ABOVE PLOTS:
<br>
# CONCLUSIONS CAN BE DRAWN SUCH AS:
<br>
# 1. THOSE WHO NEVER SMOKED COMPARED TO OTHERS HAVE A GREATER PROBABILITY OF STROKE AND HEART ATTACK.
<br>
# 2. FEMALES HAVE GREATER PROBABILITY FOR STROKES COMPARED TO MALE
<br>
# 3. AND MEN HAVE HIGHER HEART DISEASE CHANCES IN COMPARISON TO WOMEN.
<br>

# df4.drop(["smoking_status"],axis=1,inplace=True)
<br>
X = df4.drop(["stroke"],axis = 1)
<br>
y = df4["stroke"]
<br>

from sklearn.model_selection import train_test_split
<br>

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
<br>

X_train.shape
<br>

from sklearn.linear_model import LogisticRegression
<br>

log = LogisticRegression(max_iter=600)
<br>

log.fit(X_train,y_train)
<br>

y_pred = log.predict(X_test)
<br>

log.score(X_test,y_test)
<br>

from sklearn.metrics import accuracy_score,confusion_matrix
<br>

accuracy = accuracy_score(y_test,y_pred)
<br>

conf_matrix = confusion_matrix(y_test,y_pred)
<br>

print(f"ACCURACY_SCORE is {accuracy}.")
<br>
print(f"CONFUSION_MATRIX is: \n {conf_matrix}")
<br>








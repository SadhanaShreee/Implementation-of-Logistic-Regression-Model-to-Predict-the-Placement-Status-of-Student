# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SADHANA SHREE B
RegisterNumber: 212223230177
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data=pd.read_csv("Placement_Data.csv")
data.head()
data.tail()
data.info()
#delete a column
data=data.drop("sl_no",axis=1)
data=data.drop("salary",axis=1)

data["gender"] = data["gender"].astype('category')
data["hsc_b"] = data["hsc_b"].astype('category')
data["ssc_b"] = data["ssc_b"].astype('category')
data["hsc_s"]= data["hsc_s"].astype('category')
data["degree_t"] = data["degree_t"].astype('category')
data["workex"] = data["workex"].astype('category')
data["specialisation"] = data["specialisation"].astype('category')
data["status"] = data["status"].astype('category')

data["gender"]=data["gender"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes

#splitting the data as x and y
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x.shape
y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
clf=LogisticRegression()
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()

```


## Output:

![Screenshot 2024-10-16 105048](https://github.com/user-attachments/assets/a7313e73-1c2d-49fd-9928-f69b633c059f)

![Screenshot 2024-10-16 105058](https://github.com/user-attachments/assets/95a424e5-4e74-45d1-8547-7ccb48d7cbb5)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

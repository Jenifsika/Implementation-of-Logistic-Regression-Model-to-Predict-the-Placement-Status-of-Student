# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Annie Jenifsika A
RegisterNumber:  212224230019

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv("/content/Placement_Data (1).csv")
print("Placement Data:")
print(data)
if 'salary' in data.columns:
    print("\nSalary Data:")
    print(data['salary'])
else:
    print("\n'Salary' column not found in DataFrame")
data1 = data.drop(["salary"], axis=1, errors='ignore')
print("\nMissing Values Check:")
print(data1.isnull().sum())
print("\nDuplicate Rows Check:")
print(data1.duplicated().sum())

print("\nCleaned Data:")
print(data1)
le = LabelEncoder()

categorical_columns = ['workex', 'status', 'hsc_s', 'gender']  
for column in categorical_columns:
    if column in data1.columns:
        data1[column] = le.fit_transform(data1[column])
    else:
        print(f"'{column}' column not found in DataFrame")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
x = pd.get_dummies(x, drop_first=True)  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_report1 = classification_report(y_test, y_pred)

from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()



print("\nAccuracy:", accuracy)

print("Classification Report:\n", classification_report1)
print("\nY Prediction Array:")
print(y_pred)

*/
```

## Output:
Placement data:

<img width="848" height="592" alt="image" src="https://github.com/user-attachments/assets/b5e42f4a-2192-4751-93cd-f84a875a9efd" />

Salary Data :

<img width="476" height="298" alt="image" src="https://github.com/user-attachments/assets/b828497a-263d-4038-a862-652d634b4d4e" />

Missing value check:

<img width="334" height="411" alt="image" src="https://github.com/user-attachments/assets/5fa4d0ed-3b57-4662-9f67-3f8a33236d0a" />

Cleaned data:

<img width="838" height="651" alt="image" src="https://github.com/user-attachments/assets/744f0f75-9b5f-410f-b94e-3fab1a9173db" />

Y prediction array:

<img width="760" height="117" alt="image" src="https://github.com/user-attachments/assets/3ec3765a-603a-45f6-82a0-de61191c2ac2" />

Accuracy value:

<img width="303" height="33" alt="image" src="https://github.com/user-attachments/assets/82307d8e-63b0-42d2-b858-702950136b69" />

Confusion Matix:

<img width="858" height="641" alt="image" src="https://github.com/user-attachments/assets/760fa3c5-2ea6-4320-89e6-07d52ec99a0e" />


Classification Report :

<img width="630" height="204" alt="image" src="https://github.com/user-attachments/assets/3529b023-dee9-4b16-9d92-d2d99f164760" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

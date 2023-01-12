# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

data = pd.read_csv('mushrooms.csv')

data = data.drop(["veil-type"],axis=1)

for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

data.head()

X = data.drop('class', axis=1)
y = data['class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=5000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

print(classification_report(y_test,lr_pred))
print('\n')
cm = confusion_matrix(y_test, lr_pred)

x_axis_labels = ["Edible", "Poisonous"]
y_axis_labels = ["Edible", "Poisonous"]

f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED")
plt.ylabel("TRUE")
plt.title('Confusion Matrix for Logistic Regression')
plt.show()


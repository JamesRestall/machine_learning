import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("customer_churn_data.csv")

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

X = df[["TotalCharges", "tenure", "MonthlyCharges"]]
y = df["Churn"]

# Dataset statistics
print("Head: ", df.head())
print("Shape: ", df.shape)
print("Distribution of target variables: ", df.Churn.value_counts())

X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=9)

model = LogisticRegression()
model.fit(X_train, y_train)

y_prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, y_prediction)
print("Accuracy of model: ", accuracy)

conf_matrix = confusion_matrix(y_test, y_prediction)
ConfusionMatrixDisplay(conf_matrix, display_labels=model.classes_).plot()
plt.show()

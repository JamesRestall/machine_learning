import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score

df = pd.read_csv("mobil_mesin_harga.csv")



X = df[["KekuatanMesin"]]
y = df["Harga"]

# Dataset statistics
print("Head: ", df.head())
print("Shape: ", df.shape)


X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=9)

model = LinearRegression()
model.fit(X_train, y_train)

y_prediction = model.predict(X_test)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_prediction)
r2 = r2_score(y_test, y_prediction)

print("Mean Squared Error (MSE): ", mse)
print("R-squared (R2): ", r2)

# Plotting actual vs predicted
plt.scatter(y_test, y_prediction)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("mobil_mesin_harga.csv")

X = df["KekuatanMesin"].values.reshape(-1, 1)
y = df["Harga"].values

# print("X: ", X)
# print("Y: ", y)

scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
X_scaled = X

# print("X scaled: ", X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=9)

model = LinearRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_scaled)

plt.scatter(X_train, y_train)
plt.show()


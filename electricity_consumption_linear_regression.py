import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



df = pd.read_csv("electricity_consumption.csv")


print("\nFirst 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

print("\nShape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nInfo")
df.info()

print("\nStatistics")
print(df.describe())



print("\nMissing Values")
print(df.isnull().sum())
df.dropna(inplace=True)

print("\nDuplicate Rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)


X = df.drop("ElectricityConsumption", axis=1)
y = df["ElectricityConsumption"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nR2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))



print("\nIntercept:", model.intercept_)
print("\nCoefficients")
for col, coef in zip(X.columns, model.coef_):
    print(col, ":", coef)



comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nComparison")
print(comparison.head(10))



plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(range(len(y_test)), y_test.values, label="Actual")
plt.plot(range(len(y_pred)), y_pred, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Consumption")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df["ElectricityConsumption"], bins=10)
plt.title("Electricity Consumption Distribution")
plt.xlabel("Consumption")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


print("\nEnter Details")

temp = float(input("Temperature: "))
humidity = float(input("Humidity: "))
occupancy = int(input("Occupancy: "))
appliances = int(input("Number of Appliances Running: "))

sample = pd.DataFrame({
    "Temperature":[temp],
    "Humidity":[humidity],
    "Occupancy":[occupancy],
    "Appliances":[appliances]
})

sample = scaler.transform(sample)

prediction = model.predict(sample)

print("\nPredicted Electricity Consumption:", prediction[0])


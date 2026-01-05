# Linear Regression – House Price Prediction (Jupyter Notebook)

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("C:/Users/win 10/Downloads/train.csv")

# Select required features
df = df[["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath", "SalePrice"]]

# Combine bathrooms
df["TotalBath"] = df["FullBath"] + df["HalfBath"]

# Define features and target
X = df[["GrLivArea", "BedroomAbvGr", "TotalBath"]]
y = df["SalePrice"]

# Handle missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Predict price for a new house
new_house = pd.DataFrame({
    "GrLivArea": [2000],
    "BedroomAbvGr": [3],
    "TotalBath": [2]
})

print("Predicted House Price:", model.predict(new_house)[0])
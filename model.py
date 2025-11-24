import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset Link: https://www.kaggle.com/datasets/mirichoi0218/insurance
insurance_data = pd.read_csv("data/insurance.csv")

# Exploring Data
print(insurance_data.head())
print(insurance_data.info())
print(insurance_data.describe())

# Checking Null values
print(insurance_data.isna().sum())


# Function to visualize the outliers
def visualize_outliers(data):
    numerical_data = data.select_dtypes("number")
    fig, ax = plt.subplots(len(numerical_data.columns), 1, figsize=(7, 18), dpi=95)
    for i, col in enumerate(numerical_data.columns):
        ax[i].boxplot(data[col], vert=False)
        ax[i].set_ylabel(col)
    plt.tight_layout()
    plt.show()


# Function to visualize the distribution
def visualize_distribution(data):
    numerical_data = data.select_dtypes("number")
    fig, ax = plt.subplots(len(numerical_data.columns), 1, figsize=(7, 18), dpi=95)
    for i, col in enumerate(numerical_data.columns):
        ax[i].hist(data[col])
        ax[i].set_ylabel(col)
    plt.tight_layout()
    plt.show()


# Before Handling Outliers
visualize_outliers(insurance_data)
visualize_distribution(insurance_data)

# Handling outliers
insurance_data["charges"] = np.log1p(insurance_data["charges"])

numerical_data = insurance_data.select_dtypes("float64")
for col in numerical_data.columns:
    q1 = insurance_data[col].quantile(0.25)
    q3 = insurance_data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    insurance_data[col] = np.clip(insurance_data[col], lower_bound, upper_bound)

# After Handling Outliers
visualize_outliers(insurance_data)
visualize_distribution(insurance_data)

# Checking the linearity
numerical_data = insurance_data.select_dtypes("number").drop("charges", axis=1)
fig, ax = plt.subplots(len(numerical_data.columns), 1, figsize=(7, 18), dpi=95)
for i, col in enumerate(numerical_data.columns):
    ax[i].scatter(data=insurance_data, x=col, y="charges")
    ax[i].set_ylabel(col)
plt.tight_layout()
plt.show()

# Correlation Matrix
sns.heatmap(insurance_data.corr(numeric_only=True), annot=True)
plt.show()

# Ecnoding Categorical Features
onehot_cols = ["sex", "region"]
onehot_encoding = OneHotEncoder(sparse_output=False)
encoded_data = onehot_encoding.fit_transform(insurance_data[onehot_cols])
encoded_col = onehot_encoding.get_feature_names_out(onehot_cols)
insurance_data[encoded_col] = encoded_data

insurance_data["encoded_smoker"] = (insurance_data["smoker"] == "yes").astype(int)

# Feature Engineering
insurance_data["bmi_age"] = insurance_data["bmi"] * insurance_data["age"]
insurance_data["smoker_bmi"] = insurance_data["encoded_smoker"] * insurance_data["bmi"]
insurance_data["smoker_age"] = insurance_data["encoded_smoker"] * insurance_data["age"]
insurance_data["bmi_age_smoker"] = (
    insurance_data["bmi"] * insurance_data["age"] * insurance_data["encoded_smoker"]
)

insurance_data["age_group"] = pd.cut(
    insurance_data["age"], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4]
).astype(int)
insurance_data["bmi_category"] = pd.cut(
    insurance_data["bmi"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]
).astype(int)

# Splitting the data to train and test
drop_columns = insurance_data.select_dtypes("object").columns.tolist()
drop_columns.extend(["charges"])

X = insurance_data.drop(drop_columns, axis=1)
y = insurance_data["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40
)

# Linear Regression Model
model_LR = LinearRegression()
model_LR.fit(X_train, y_train)

kf = KFold(n_splits=5, shuffle=True, random_state=40)
cross_validation = cross_val_score(model_LR, X_train, y_train, cv=kf)

y_pred_log = model_LR.predict(X_test)
y_pred_original = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

print(f"Training Score: {model_LR.score(X_train, y_train)}")
print(f"Testing Score: {r2_score(y_test_original, y_pred_original)}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test_original, y_pred_original)}")
print(f"Cross Validation Score: {np.mean(cross_validation)}")

# Saving the model
import joblib

joblib.dump(model_LR, "models/insurance_model.pkl")
joblib.dump(onehot_encoding, "models/onehot_encoder.pkl")



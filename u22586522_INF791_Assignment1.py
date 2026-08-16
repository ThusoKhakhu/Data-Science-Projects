import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#Specifying filepath
file_path = r"C:\Users\Marcia\Desktop\INF 791 (Applied Data Science) Assignments\Assignment 1\Data-Science-Projects\diabetes_prediction_dataset.csv"

########### Part 1: Data Preparation
#################################### 1.1 Data cleaning
    #Loading data

data_frame = pd.read_csv(file_path)
print("Dataset loaded successfully.")
print(f"Dataset shape: {data_frame.shape}")
     #Identifying missing values and placeholders

#Inspecting the dataset for potential same value variations across the dataset
categorical_columns = data_frame.select_dtypes(include=["object", "string"]).columns

for column in categorical_columns:
    print(data_frame[column].value_counts(dropna=False))



missing_v = data_frame.isna().sum()
count = missing_v[missing_v > 0]
if count.empty:
    print(f'No missing value found')
else:
    print(f'Missing value per column: {count}')

   #Data normalisation
normalised_d = (data_frame["smoking_history"]
                .astype("string")
                .str.strip()
                .str.lower()
    
)

    
count = (normalised_d == "no info").sum()
print(f"\nPlaceholder value: {count}:")


     #Handling missing and placeholder values
data_frame["smoking_history"] = data_frame["smoking_history"].replace("No Info", "Unknown")



print(data_frame["smoking_history"].value_counts(dropna=False))



#Identifying categorical columns

print(data_frame.dtypes) #Displaying data types

# Categorise features
numerical_features = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

categorical_features = [
    "gender",
    "hypertension",
    "heart_disease",
    "smoking_history"
]

target_variable = "diabetes"

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

print("\nTarget variable:")
print(target_variable)
# pd.set_option('display.max_columns', None)
# print(data_frame.head())



# print(data_frame["smoking_history"].value_counts())
#################################### 1.2 Feature Engineering: New Features
  ## BMI Category
data_frame["bmi_category"] = pd.cut(
    data_frame["bmi"],
    bins = [0, 18.5, 25, 30, float("inf")],
    labels = ["Underweight", "Normal", "Overweight", "Obese"]
)



  ## Age group
data_frame["age_group"] = pd.cut(
    data_frame["age"],
    bins = [0, 18, 40, 60, float("inf")],
    labels = ["Young", "Adult", "Middle-aged", "Elderly"]
)


  ##High glucose risk
data_frame["high_glucose_risk"] = (
    (data_frame["HbA1c_level"] >= 6.5) |
    (data_frame["blood_glucose_level"] > 162)
).astype(int)


  ## Cardiovascular risk

data_frame["cardiovascular_risk"] = (
    (data_frame["hypertension"] == 1) |
    (data_frame["heart_disease"] == 1)
).astype(int)

print(data_frame[
    [
        "age",
        "age_group",
        "bmi",
        "bmi_category",
        "HbA1c_level",
        "blood_glucose_level",
        "high_glucose_risk",
        "hypertension",
        "heart_disease",
        "cardiovascular_risk"
    ]
].head())
# print(data_frame.columns.tolist())

#################################### 1.3 Basic Statistics

  ##
numerical_features = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "hypertension",
    "heart_disease",
    "diabetes"
]

pd.set_option("display.max_columns", None)
print("\nSummary Statistics:")
print(data_frame[numerical_features].describe())


 #Correlation matrix
corr_m = data_frame.corr(numeric_only=True)
print("\nCorrelation MAtrix:")
print(corr_m)

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_m,
    annot = True,
    fmt = ".2f",
    cmap = "coolwarm",
    center = 0
)

plt.title("Correlation matric of numerical features")
plt.xticks(rotation = 35, ha = "right")
plt.yticks(rotation = 0)
plt.tight_layout()
plt.show()

correlation_with_diabetes = (
    corr_m["diabetes"]
    .drop("diabetes")
    .sort_values(ascending=False)
)

print("\nCorrelation with diabetes:")
print(correlation_with_diabetes)


# Remove duplicate correlation pairs and the diagonal
upper_triangle = corr_ma.where(
    np.triu(np.ones(corr_m.shape), k=1).astype(bool)
)

correlation_pairs = upper_triangle.stack().sort_values(
    ascending=False
)

print("\nStrongest positive correlations between features:")
print(correlation_pairs.head(10))

print("\nStrongest negative correlations between features:")
print(correlation_pairs.tail(10))

#################################### 1.4 Data Visualisation
#################################### 1.5 Data Processing: Additional Data Transformations
#################################### 1.6 Data Encoding
########### Part 2: Supervised Learning

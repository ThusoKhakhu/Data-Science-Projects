import pandas as pd
import re

#Specifying filepath
file_path = r"C:\Users\u22586522\Downloads\diabetes_prediction_dataset.csv\diabetes_prediction_dataset.csv"

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
#################################### 1.3 Basic Statistics
#################################### 1.4 Data Visualisation
#################################### 1.5 Data Processing: Additional Data Transformations
#################################### 1.6 Data Encoding
########### Part 2: Supervised Learning

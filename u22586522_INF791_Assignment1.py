import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder


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
    cmap = "RdYlGn",
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
upper_triangle = corr_m.where(
    np.triu(np.ones(corr_m.shape), k=1).astype(bool)
)

correlation_pairs = upper_triangle.stack().sort_values(
    ascending=False
)

print("\nStrongest positive correlations between features:")
print(correlation_pairs.head(10))

negative_corr = correlation_pairs[correlation_pairs < 0]

print("\nStrongest negative correlations between features:")
print(negative_corr.tail(10))




#################################### 1.4 Data Visualisation
  ##Age distribution
plt.figure(figsize=(10, 6))

sns.histplot(data = data_frame, 
             x = "age", 
             hue = "diabetes", 
             alpha = 0.45, 
             binwidth = 1, 
             kde = True,
             palette=["#A8D5BA", "#F2A6A6"],
             legend = True,
             element = "step")

plt.title("Distribution of Patient Age")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.legend(title = "Diabetes Status", labels = ["Diabetes", "No Diabetes"])
plt.tight_layout()
plt.show()


 ##Glucose vs daibetes

plt.figure(figsize = (8, 5))
sns.violinplot(
    data = data_frame,
    x = "diabetes",
    y = "blood_glucose_level",
    palette=["#4C956C", "#C8553D"],
    inner = "box",
    cut = 0
)

plt.title("Blood Glucose Level by Diabetes Status")
plt.xlabel("Diabetes")
plt.ylabel("Blood Glucose Level")
plt.xticks([0, 1], ["No Diabetes", "Diabetes"])
plt.tight_layout()
plt.show()

  ##HbA1c vs diabetes
plt.figure(figsize = (8, 5))
sns.boxplot(
    data = data_frame,
    x = "diabetes",
    y = "HbA1c_level",
    palette=["#4C956C", "#C8553D"],
    legend = False,
)

plt.title("HbA1c Level by Diabetes Status")
plt.xlabel("Diabetes")
plt.ylabel("HbA1c Level")
plt.tight_layout()
plt.show()

  ##High glucose risk vs diabetes
plt.figure(figsize = (8, 5))
sns.countplot(
    data = data_frame,
    x = "high_glucose_risk",
    hue = "diabetes",
    palette=["#4C956C", "#C8553D"],
    legend = False,
)
plt.title("High Glucose Risk by Diabetes Status")
plt.xlabel("High Glucose Risk")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

 ## BMI category vs diabetes
print(data_frame["bmi_category"].value_counts())
bmi_diabetes = pd.crosstab(
    data_frame["bmi_category"],
    data_frame["diabetes"],
    normalize = "index"
)
bmi_diabetes = bmi_diabetes*100

print(bmi_diabetes)

plt.figure(figsize =(8, 5))
sns.barplot(
    data = bmi_diabetes.reset_index(),
    x = "bmi_category",
    y = 1
)
plt.title("Diabetes Prevalance Across BMI Categories")
plt.xlabel("BMI Category")
plt.ylabel("Diabetes Prevalence (%)")
plt.xticks(rotation = 20)
plt.tight_layout()
plt.show()

  ##Risk factor map
risk_table = pd.crosstab(
    data_frame["cardiovascular_risk"],
    data_frame["high_glucose_risk"],
    values = data_frame["diabetes"],
    aggfunc = "mean"
)

risk_table = risk_table*100
print(risk_table)
plt.figure(figsize = (9, 7))
sns.heatmap(
    risk_table,
    annot = True,
    fmt = ".1f",
    cmap = "YlOrRd",
    linewidths = 0.5,
    cbar_kws = {"label": "Diabetes Prevalence (%)"}
)
plt.title("Diabetes Prevalence by Cardiovascular and High Glucose Risk")
plt.xlabel("High Glucose Risk")
plt.ylabel("Cardiovascular Risk")
plt.tight_layout()
plt.show()

#################################### 1.5 Data Processing: Additional Data Transformations
 ##Identifying irrelevant, duplicate and redundant features
print("\nFinal dataset columns:")
print(data_frame.columns.tolist())

print("\nNumber of duplicate rows:")
print(data_frame.duplicated().sum())

data_frame  = data_frame.drop_duplicates()
print("\nAre duplicates removed?")
print(data_frame.duplicated().sum()) #Verifying duplicates are removed

print("\nData types:")
print(data_frame.dtypes) #Checking data types

##Removing age_group from the data for ML
model_df = data_frame.copy()

model_df = model_df.drop(
    columns = ["age_group"]
)

##Checking categorical values content
cf = [
    "gender",
    "smoking_history",
    "bmi_category"
]

for column in cf:
    print(f"\n{column}:")
    print(data_frame[column].value_counts())

  ##Checking numerical skewness
print("\nSkewness of numerical values:")
print(data_frame[numerical_features].skew())

  ##Transforming highly skewed values
# date_frame["bmi_log"] = np.log1(data_frame["bmi"])

plt.figure(figsize = (8, 5))
sns.histplot(
        data = data_frame,
        x = "bmi",
        bins = 40,
        kde = True,
        alpha = 0.6
)
plt.title("Distribution of BMI")
plt.xlabel("BMI")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

## Blood glucose distribution

plt.figure(figsize=(10, 6))

sns.histplot(
    data=data_frame,
    x="blood_glucose_level",
    bins=30,
    kde=False,
    alpha=0.6
)

plt.title("Distribution of Blood Glucose Level")
plt.xlabel("Blood Glucose Level")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

  ##Testing log transformation
blood_glucose_log = np.log1p(
    data_frame["blood_glucose_level"]
)
print("Original skewness:",
     data_frame["blood_glucose_level"].skew()
     )
print("Log transformed skewness:",
     blood_glucose_log.skew()
     )


bmi_log = np.log1p(data_frame["bmi"])

print("\nBMI skewness:")
print("Original skewness:", data_frame["bmi"].skew())
print("Log transformed skewness:", bmi_log.skew())

##Creating a model data frame
model_df = data_frame[[
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "hypertension",
    "heart_disease",
    "gender",
    "smoking_history",
    "diabetes"
]].copy()
print("\nInitial modelling dataset:")
print(model_df.head())

print("\nModelling dataset columns:")
print(model_df.columns.tolist())

 ##Applying log transformation on the model data
model_df["bmi"] = np.log1p(model_df["bmi"])
model_df["blood_glucose_level"] = np.log1p(
    model_df["blood_glucose_level"]
)

print("\nSkewness after transformation:")

print("BMI:",
      model_df["bmi"].skew())

print("blood_glucose_level", 
      model_df["blood_glucose_level"].skew())

print("\nGender categories:")
print(model_df["gender"].unique())

print("\nSmoking history categories:")
print(model_df["smoking_history"].unique())

print("\nGender distribution:")
print(model_df["gender"].value_counts())

print("\nSmoking history distribution:")
print(model_df["smoking_history"].value_counts())

model_df["gender"] = (
    model_df["gender"]
    .astype(str)
    .str.strip()
    .str.lower()
)

model_df["smoking_history"] = (
    model_df["smoking_history"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print("\nCleaned gender categories:")
print(model_df["gender"].unique())

print("\nCleaned smoking history categories:")
print(model_df["smoking_history"].unique())

print("\nData types after preprocessing:")
print(model_df.dtypes)

print("\nMissing values:")
print(model_df.isnull().sum())

print("\nNumber of duplicate rows:")
print(model_df.duplicated().sum())

####################################

#################################### 1.6 Data Encoding
## Separate features and target

X = model_df.drop("diabetes", axis=1)
y = model_df["diabetes"]

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)


categorical_features = [
    "gender",
    "smoking_history"
]

print("\nEncoded categorical features:")
print(categorical_features)

  ##Applying OneHotEncoding
encoder = OneHotEncoder(
    sparse_output = False,
    handle_unknown = "ignore"
)

encoded_data = encoder.fit_transform(
    X[categorical_features]
)

encoded_columns = encoder.get_feature_names_out(
    categorical_features
)

  ##Creating data frame containing encoded variables
encoded_data_frame = pd.DataFrame(
    encoded_data,
    columns = encoded_columns,
    index = X.index
)

print("\nEncoded categorical features")
print(encoded_data_frame.head())

##Comining the encoded numerical data with already existing numerical data
X_numerical = X.drop(
    columns = categorical_features
)

X_encoded = pd.concat(
    [X_numerical, encoded_data_frame],
    axis = 1
)

model_ready_df = pd.concat(
    [X_encoded, y],
    axis = 1
)

print("\nFinal dataset")
print(model_ready_df)

print("\nFinal dataset shape")
print(model_ready_df.shape)

print("\nFinal dataset data types")
print(model_ready_df.dtypes)

print("\nNon-numerical columns")
print(model_ready_df.select_dtypes(
    exclude = "number"
).columns.tolist())

print("\nMissing values")
print(model_ready_df.isnull().sum())

print("\nTotal number of missinf values")
print(model_ready_df.isnull().sum().sum())

# Save your actual dataframe to a CSV file
model_ready_df.to_csv("ThusoKhakhu_preprocessed_diabetes_data.csv", index=False)
print("Saved preprocessed dataset to CSV successfully!")


########### Part 2: Supervised Learning

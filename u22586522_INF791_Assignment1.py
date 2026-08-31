import kagglehub
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import time




#Specifying filepath
# file_path = r"C:\Users\Marcia\Desktop\INF 791 (Applied Data Science) Assignments\Assignment 1\Data-Science-Projects\diabetes_prediction_dataset.csv"
folder_path = kagglehub.dataset_download("iammustafatz/diabetes-prediction-dataset")
file_path = os.path.join(folder_path, "diabetes_prediction_dataset.csv")
########### Part 1: Data Preparation
#################################### 1.1 Data cleaning
    #Loading data

data_frame = pd.read_csv(file_path)
print("Dataset loaded successfully.")
print(f"Dataset shape: {data_frame.shape}")


#Identifying missing values and placeholders
#Inspecting the dataset for possible missing values
string_columns = data_frame.select_dtypes(
    include=["object", "string"]
).columns

print("\nString values count:")
for column in string_columns:
    print(data_frame[column].value_counts(dropna=False))

numerical_columns = data_frame.select_dtypes(
    include = ["int64", "float64"]
).columns

for column in numerical_columns:
    print(f"\n{column}:")
    print(data_frame[column].unique())

print(data_frame[numerical_columns].describe())
    

# Display the unique values in each string column
for column in string_columns:
    print(f"\nUnique values in '{column}':")
    print(data_frame[column].unique())

#Checking for missing values
missing_v = data_frame.isna().sum()
count = missing_v[missing_v > 0]
if count.empty:
    print(f'No missing value found')
else:
    print(f'Missing value per column: {count}')

#Counting placeholder values
p_count = (
    data_frame["smoking_history"]
    .astype("string")
    .str.strip()
    .eq("No Info")
    .sum()
)
print(f"\nPlaceholder value count: {p_count}")

     #Handling missing and placeholder values
data_frame["smoking_history"] = (
    data_frame["smoking_history"]
    .astype("string")
    .str.strip()
    .str.lower()
    .replace({"no info": "unknown"})
)

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
]

binary_features = [
    "hypertension",
    "heart_disease",
    "diabetes"
]

pd.set_option("display.max_columns", None)
print("\nSummary Statistics:")
print("\nNumerical features stat summary")
print(data_frame[numerical_features].describe())
print("\nBinary variable frequences:")

for column in binary_features:
    print(f"\n{column}:")
    print(data_frame[column].value_counts())


 #Correlation matrix
pd.set_option("display.max_columns", None)
corr_m = data_frame.corr(numeric_only=True)
print("\nCorrelation MAtrix:")
print(corr_m)

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_m,
    annot = True,
    fmt = ".2f",
    cmap = "rocket",
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
plt.figure(figsize=(8, 5))

sns.histplot(data = data_frame, 
             x = "age", 
             hue = "diabetes", 
             alpha = 0.45, 
             binwidth = 1, 
             kde = True,
             palette = ["limegreen", "crimson"],
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
    palette = ["limegreen", "crimson"],
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
fig, ax = plt.subplots(figsize = (10, 5))
ax.set_facecolor("lightgrey")
sns.boxplot(
    data = data_frame,
    x = "diabetes",
    y = "HbA1c_level",
    palette = ["limegreen", "crimson"],
    legend = False,
    ax = ax
)

#stats = data_frame.groupby("diabetes")["HbA1c_level"].describe()

#Looping through stats groups adding labels
for i, category in enumerate(data_frame["diabetes"].unique()):

    stats = data_frame[data_frame["diabetes"] == category]["HbA1c_level"]
    
    #extracting 5 number summary
    low = stats.min().item()
    q1 = stats.quantile(0.25).item()
    med = stats.median().item()
    q3 = stats.quantile(0.75).item()
    high = stats.max().item()

    #positioning the label text in the box plot
    x_position = i + 0.30

    #Adding the text labels into the plot
    ax.text(x_position, low, f"Min: {low:.2f}", va = "center", ha = "left", fontsize = 9)
    ax.text(x_position, q1, f"Q1: {q1:.2f}", va = "center", ha = "left", fontsize = 9)
    ax.text(x_position, med, f"Med: {med:.2f}", va = "center", ha = "left", fontsize = 9)
    ax.text(x_position, q3, f"Q3: {q3:.2f}", va = "center", ha = "left", fontsize = 9)
    ax.text(x_position, high, f"Max: {high:.2f}", va = "center", ha = "left", fontsize = 9)

plt.title("HbA1c Level by Diabetes Status")
plt.xlabel("Diabetes")
plt.ylabel("HbA1c Level")
ax.set_xlim(-0.5, 1.6)
plt.tight_layout()
plt.show()

  ##High glucose risk vs diabetes
plt.figure(figsize = (8, 5))
sns.countplot(
    data = data_frame,
    x = "high_glucose_risk",
    hue = "diabetes",
    palette = ["limegreen", "crimson"],
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
    y = 1,
   
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
plt.figure(figsize = (8, 5))
sns.heatmap(
    risk_table,
    annot = True,
    fmt = ".1f",
    cmap = "coolwarm",
    linewidths = 0.5,
    cbar_kws = {"label": "Diabetes Prevalence (%)"}
)
plt.title("Diabetes Prevalence by Cardiovascular and High Glucose Risk")
plt.xlabel("High Glucose Risk")
plt.ylabel("Cardiovascular Risk")
plt.tight_layout()
plt.show()

#################################### 
 #1.5 Data Processing: Additional Data Transformations
####################################

 #Identifying irrelevant, duplicate and redundant features
print("\nFinal dataset columns:")
print(data_frame.columns.tolist())

print("\nNumber of duplicate rows:")
print(data_frame.duplicated().sum())

#Remove duplicate rows
data_frame  = data_frame.drop_duplicates()

print("\nAre duplicates removed?")
print(data_frame.duplicated().sum()) #Verifying duplicates are removed

 #Checking data types
print("\nData types:")
print(data_frame.dtypes)

 #Creating a model data frame
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

 #Checking categorical values content
cf = [
    "gender",
    "smoking_history",
    "bmi_category"
]

for column in cf:
    print(f"\n{column}:")
    print(data_frame[column].value_counts())


 #Checking numerical skewness
print("\nSkewness of numerical values:")
print(data_frame[numerical_features].skew())

 #Transforming highly skewed values
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

 # Blood glucose distribution

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

#Testing log transformation for skewwd variables
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

 #Applying log transformation on the model data
model_df["bmi"] = np.log1p(model_df["bmi"])
model_df["blood_glucose_level"] = np.log1p(
    model_df["blood_glucose_level"]
)

 #Verifying skewness after transformation
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

 #Checking cleaned categories
print("\nCleaned gender categories:")
print(model_df["gender"].unique())

print("\nCleaned smoking history categories:")
print(model_df["smoking_history"].unique())

 #Checking data types after preprocessing
print("\nData types after preprocessing:")
print(model_df.dtypes)

 #Checking missing values
print("\nMissing values:")
print(model_df.isnull().sum())

 #Checking duplicates
print("\nNumber of duplicate rows:")
print(model_df.duplicated().sum())

####################################
 #1.6 Data Encoding
####################################
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

print("\nTotal number of missing values")
print(model_ready_df.isnull().sum().sum())

# Saving actual dataframe to a CSV file: Submission instruction
# model_ready_df.to_csv("ThusoKhakhu_preprocessed_diabetes_data.csv", index=False)
# print("Saved preprocessed dataset to CSV successfully!")


########### Part 2: Supervised Learning
##2.1 Training a classification model

X = X_encoded
y = model_ready_df["diabetes"]

print ("\nFeatures")
print(X.columns.tolist())

print("\nTarget")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.20, #20% of dataset
    random_state = 42,
    stratify = y
)

print("\nTraining set shape:")
print(X_train.shape)

print("\nTesting set shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    random_state = 42,
    max_iter = 1000
)

model.fit(X_train_scaled, y_train) #Taining the model

y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Diabetes", "Diabetes"]
)

disp.plot()

plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()

# Logistic Regression pipeline
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        random_state=42,
        max_iter=1000
    ))
])
logistic_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(X_train, y_train)


# Logistic Regression training time
start_time = time.perf_counter()

logistic_model.fit(X_train, y_train)

logistic_training_time = time.perf_counter() - start_time


# Random Forest training time
start_time = time.perf_counter()

random_forest_model.fit(X_train, y_train)

random_forest_training_time = time.perf_counter() - start_time



# Logistic Regression predictions
logistic_pred = logistic_model.predict(X_test)
logistic_prob = logistic_model.predict_proba(X_test)[:, 1]


# Random Forest predictions
rf_pred = random_forest_model.predict(X_test)
rf_prob = random_forest_model.predict_proba(X_test)[:, 1]

# Logistic Regression metrics
logistic_accuracy = accuracy_score(y_test, logistic_pred)
logistic_precision = precision_score(y_test, logistic_pred)
logistic_recall = recall_score(y_test, logistic_pred)
logistic_f1 = f1_score(y_test, logistic_pred)
logistic_roc_auc = roc_auc_score(y_test, logistic_prob)


# Random Forest metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_roc_auc = roc_auc_score(y_test, rf_prob)

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        logistic_accuracy,
        rf_accuracy
    ],
    "Precision": [
        logistic_precision,
        rf_precision
    ],
    "Recall": [
        logistic_recall,
        rf_recall
    ],
    "F1 Score": [
        logistic_f1,
        rf_f1
    ],
    "ROC-AUC": [
        logistic_roc_auc,
        rf_roc_auc
    ],
    "Training Time (seconds)": [
        logistic_training_time,
        random_forest_training_time
    ]
})

print("\nModel Comparison:")
print(results.round(4))



# Calculate ROC curves
logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test,
    logistic_prob
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_prob
)


plt.figure(figsize=(9, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic Regression (AUC = {logistic_roc_auc:.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    logistic_pred,
    display_labels=["No Diabetes", "Diabetes"],
    ax=axes[0]
)

axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_pred,
    display_labels=["No Diabetes", "Diabetes"],
    ax=axes[1]
)

axes[1].set_title("Random Forest")

plt.tight_layout()
plt.show()
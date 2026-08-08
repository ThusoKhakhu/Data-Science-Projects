import pandas as pd

#Specifying filepath
file_path = r"C:\Users\u22586522\Downloads\diabetes_prediction_dataset.csv\diabetes_prediction_dataset.csv"

########### Part 1: Data Preparation
#Loading data
data_frame = pd.read_csv(file_path)

#
print(data_frame.head())
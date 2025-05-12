import pandas as pd
import os
import matplotlib.pyplot as plt

features_path = r'D:\Vittoria\Code\data'
'''
#plots all features for all participants
file_path = rf'{features_path}\all_features.csv'
for part in range (1, 17):
    data = pd.read_csv(file_path)
    if any(data.columns.str.contains("Time")):
        data = data.set_index("Time")
    data_part = data[data["Participant_ID"] == part]
    for col in data_part.columns:
        if col != "Participant_ID":
            plt.figure(figsize=(10, 5))
            plt.plot(data_part[col], label=col, linewidth=2)
            plt.xlabel("Time")
            plt.ylabel(col)
            plt.title(f"Time-Series Plot for {col}")
            plt.legend()
            plt.savefig(rf'{features_path}\plots\feature_plots\{part}_{col}')
        
#Dexcom plot: plots Gluocose Mean for all participants 
path = rf'{features_path}\dexcom\dexcom_data_5mins.csv'
data = pd.read_csv(path)
if any(data.columns.str.contains("Timestamp")):
    data = data.set_index("Timestamp")
for part in range (1, 17):
    data_part = data[data["Participant_ID"] == part]
    plt.figure(figsize=(10, 5))
    plt.plot(data_part["Glucose_Mean"], label="Glucose_Mean", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Glucose_Mean")
    plt.title(f"Time-Series Plot for Glucose_Mean")
    plt.legend()
    plt.savefig(rf'{features_path}\plots\dexcom_plots\{part}_Glucose_Mean')
'''

#plots HbA1c values of all participants in a bar plot with different colours for women and men
#horizontal line characterizing prediabetes range 
path = rf'{features_path}\demographics.csv'
data = pd.read_csv(path)
data = data.set_index("ID")
male_data = data[data["Gender"] == "MALE"]
female_data = data[data["Gender"] == "FEMALE"]
plt.bar(male_data.index, male_data["HbA1c"], color="blue", label="Male", alpha=0.7)
plt.bar(female_data.index, female_data["HbA1c"], color="red", label="Female", alpha=0.7)
plt.axhline(y=5.7, color='r', linestyle='-')
plt.axhline(y=6.4, color='r', linestyle='-')
plt.legend()
plt.xlabel("Feature Name")
plt.ylabel("Value")
plt.savefig(rf'{features_path}\plots\HbA1c')

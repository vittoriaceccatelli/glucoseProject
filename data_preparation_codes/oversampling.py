import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from imblearn.over_sampling import SMOTE

# Select GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define path to feature files
features_path = r'D:\Vittoria\Code\data'

# Ask user to select dataset version
dataset = input("Labeled or NoCorrelation? ")

# Load appropriate dataset based on user input
if dataset == "Labeled":
    file_path = rf'{features_path}\labeled_features.csv'
elif dataset == "NoCorrelation":
    file_path = rf'{features_path}\correlation\labeled_features_no_correlation_RandomForest_Classification.csv'

# Read the selected CSV file
data = pd.read_csv(file_path)

# Convert time column to datetime object for sorting and filtering
data['Time'] = pd.to_datetime(data['Time'])

# Count class distribution before oversampling for each participant
df_list = []
for part in range(1, 17):
    data_part = data[data["Participant_ID"] == part]
    part_data = {
        "Participant_ID": part,
        "PersNorm_before_balancing": (data_part["PersStatus"] == "PersNorm").sum(),
        "PersHigh_before_balancing": (data_part["PersStatus"] == "PersHigh").sum(),
        "PersLow_before_balancing": (data_part["PersStatus"] == "PersLow").sum(),
    }
    df_list.append(part_data)

# Save pre-oversampling class distribution
df = pd.DataFrame(df_list)
df.to_csv(rf"D:\Vittoria\Code\data\balanced_datasets\persClasses_before_oversampling_{dataset}.csv")

# Apply SMOTE to oversample minority classes
oversample = SMOTE(random_state=42)
X = data.drop(columns=['Time', 'PersStatus', 'Glucose_Mean'])  # Features
y = data['PersStatus']  # Labels
df_balanced, y_over = oversample.fit_resample(X, y)  # Perform oversampling
df_balanced['PersStatus'] = y_over  # Add resampled labels to dataframe

# Save oversampled dataset
df_balanced.to_csv(rf'data\balanced_datasets\oversampled_{dataset}.csv', index=False)

# Count class distribution after oversampling for each participant
df_list = []
for part in range(1, 17):
    data_part = df_balanced[df_balanced["Participant_ID"] == part]
    part_data = {
        "Participant_ID": part,
        "PersNorm_before_balancing": (data_part["PersStatus"] == "PersNorm").sum(),
        "PersHigh_before_balancing": (data_part["PersStatus"] == "PersHigh").sum(),
        "PersLow_before_balancing": (data_part["PersStatus"] == "PersLow").sum(),
    }
    df_list.append(part_data)

# Save post-oversampling class distribution
df = pd.DataFrame(df_list)
df.to_csv(rf"D:\Vittoria\Code\data\balanced_datasets\persClasses_after_oversampling_{dataset}.csv")
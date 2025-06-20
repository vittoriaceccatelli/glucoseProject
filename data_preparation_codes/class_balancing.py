import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import torch

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define base path for input features
features_path = r'D:\Vittoria\Code\data'

# Ask user to select dataset type and balancing strategy
dataset = input("Labeled or NoCorrelation? ")
continuous = input("ContinuousYes or ContinuousNo? ")

# Load appropriate dataset based on user input
if dataset == "Labeled":
    file_path = rf'{features_path}\labeled_features.csv'
elif dataset == "NoCorrelation":
    file_path = rf'{features_path}\correlation\labeled_features_no_correlation_RandomForest_Classification.csv'

# Read CSV file into DataFrame
data = pd.read_csv(file_path)

# Convert time column to datetime object
data['Time'] = pd.to_datetime(data['Time'])

# Count pre-balancing class distribution per participant

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

df = pd.DataFrame(df_list)
# df.to_csv(rf"D:\Vittoria\Code\data\balanced_datasets\persClasses_before_balancing_{dataset}.csv")  # Save if needed

# Perform balancing

# Split data into classes
df_persnorm = data[data['PersStatus'] == 'PersNorm']
df_pershigh = data[data['PersStatus'] == 'PersHigh']
df_perslow = data[data['PersStatus'] == 'PersLow']

# Get the minimum sample size across the three classes
target_n_samples = min([len(df_persnorm), len(df_pershigh), len(df_perslow)])

# Downsample high and low classes to match target size
df_pershigh_sampled = df_pershigh.sample(n=target_n_samples, random_state=42)
df_perslow_sampled = df_perslow.sample(n=target_n_samples, random_state=42)

# BALANCING STRATEGY: RANDOM
if continuous == "ContinuousNo": 
    df_persnorm_sampled = df_persnorm.sample(n=target_n_samples, random_state=42)

    # Combine and sort by Participant and Time
    df_balanced = pd.concat([df_persnorm_sampled, df_pershigh_sampled, df_perslow_sampled])
    df_balanced = df_balanced.sort_values(by=['Participant_ID', 'Time']).reset_index(drop=True)

    # Save to CSV
    df_balanced.to_csv(rf'D:\Vittoria\Code\data\balanced_datasets\balanced_dataset_random_{dataset}.csv', index=False)

# BALANCING STRATEGY: CONTINUOUS SEQUENCES
elif continuous == "ContinuousYes":
    drop = 0  # Counter for dropped points
    drop_limit = len(df_persnorm) - target_n_samples  # How many can be dropped to reach balance
    df_persnorm_sampled = pd.DataFrame()

    # Iterate over each participant
    for part in range(1, 17):
        data_part = data[data["Participant_ID"] == part]
        current_sequence = 0
        indices_to_drop = []
        kept_indices = []

        # Iterate through participant's time series
        for index, row in data_part.iterrows():
            if row["PersStatus"] == "PersNorm":
                current_sequence += 1 
                indices_to_drop.append(index)
            else:
                # Drop only if long enough and under global drop limit
                if current_sequence > 12:
                    if drop + current_sequence <= drop_limit:
                        drop += current_sequence
                        kept_indices.extend(indices_to_drop)

                current_sequence = 0
                indices_to_drop = []

        # Final check for trailing PersNorm streak
        if current_sequence > 12:
            if drop + current_sequence <= drop_limit:
                drop += current_sequence
                kept_indices.extend(indices_to_drop)

        # Drop selected sequences and merge remaining data
        data_part = data_part.drop(kept_indices)
        df_persnorm_sampled = pd.concat([df_persnorm_sampled, data_part])

    # Filter only PersNorm rows after time filtering
    df_persnorm_sampled = df_persnorm_sampled[df_persnorm_sampled["PersStatus"] == "PersNorm"]

    # Combine with high and low samples
    df_balanced = pd.concat([df_persnorm_sampled, df_pershigh_sampled, df_perslow_sampled])
    df_balanced = df_balanced.sort_values(by=['Participant_ID', 'Time']).reset_index(drop=True)

    # Save to CSV
    df_balanced.to_csv(rf'D:\Vittoria\Code\data\balanced_datasets\balanced_dataset_more_continuous_{dataset}.csv', index=False)

# Print final class distribution after balancing
print(df_balanced["PersStatus"].value_counts())

# Count post-balancing class distribution per participant

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

df = pd.DataFrame(df_list)

# Save class counts after balancing
if continuous == "ContinuousNo": 
    df.to_csv(rf"D:\Vittoria\Code\data\balanced_datasets\persClasses_after_balancing_random_{dataset}.csv")
elif continuous == "ContinuousYes":
    df.to_csv(rf"D:\Vittoria\Code\data\balanced_datasets\persClasses_after_balancing_more_continuous_{dataset}.csv")
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

#GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

features_path = r'D:\Vittoria\Code\data'

dataset = input("Labeled or NoCorrelation? ")

#import labeled features and demographics
if dataset=="Labeled":
    file_path = rf'{features_path}\labeled_features.csv'
elif dataset=="NoCorrelation":
    file_path = rf'{features_path}\labeled_features_no_correlation_RandomForest_Classification.csv'

data = pd.read_csv(file_path)

data['Time'] = pd.to_datetime(data['Time'])

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
df.to_csv(rf"D:\Vittoria\Code\data\persClasses_before_oversampling_{dataset}.csv")


oversample = SMOTE(random_state=42)
X = data.drop(columns=['Time', 'PersStatus', 'Glucose_Mean'])
y = data['PersStatus']
df_balanced, y_over = oversample.fit_resample(X, y)
df_balanced['PersStatus'] = y_over

df_balanced.to_csv(rf'data\oversampled_{dataset}.csv', index=False)





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
df.to_csv(rf"D:\Vittoria\Code\data\persClasses_after_oversampling_{dataset}.csv")

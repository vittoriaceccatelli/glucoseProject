import pandas as pd
import torch

data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features.csv")
#demo = pd.read_csv(r"D:\Vittoria\Code\data\labeled_demographics.csv")
#data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")
#data = pd.read_csv(r"D:\Vittoria\Code\data\balanced_dataset_more_continuous_NoCorrelation.csv")
#data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features.csv")
#print(data["Glucose_Mean"].mean())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
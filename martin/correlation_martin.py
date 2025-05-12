import pandas as pd 
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import gc 
gc.collect()

#GPU
torch.cuda.empty_cache()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

#import labeled features and labeled demographics 
data = pd.read_csv(r"D:\Vittoria\Code\data\martin\martins_features.csv")
demo = pd.read_csv(r"D:\Vittoria\Code\data\labeled_demographics.csv")
#importance = pd.read_csv(r"D:\Vittoria\Code\data\martin\sorted_importance_with_martin_glucose.csv")
importance = pd.read_csv(r"D:\Vittoria\Code\data\martin\sorted_importance_with_martin_pers.csv")

importance = importance.set_index("Feature").T

data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")
data = data.drop(columns=['DiabStatus', 'ID'])
data['Gender'] = data['Gender'].map({'MALE': 0, 'FEMALE': 1})

correlation_matrix = data.select_dtypes(include=[np.number]).corr(method="spearman")
#correlation_matrix.to_csv(r"D:\Vittoria\Code\data\martin\correlation_matrix_martin_pers.csv", index = False)
#correlation_matrix.to_csv(r"D:\Vittoria\Code\data\martin\correlation_matrix_martin_glucose.csv", index = False)


plt.figure(figsize=(12, 8))
sns.set(font_scale=0.5)
sns.heatmap(correlation_matrix, cmap="coolwarm", linewidths=0.5, xticklabels=True, yticklabels=True)
plt.title(f"Feature Correlation Matrix, Spearman, Glucose as ground truth")
#plt.savefig(r"D:\Vittoria\Code\data\martin\before_drop_martin_glucose")
#plt.savefig(r"D:\Vittoria\Code\data\martin\before_drop_martin_pers")
#plt.show()

threshold = 0.7

correlated_features = []
to_drop = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold and correlation_matrix.index[i] != "Glucose_Mean" and correlation_matrix.index[j] != "Glucose_Mean" and correlation_matrix.index[i] != "Time" and correlation_matrix.index[j] != "Time" and correlation_matrix.index[i] != "PersStatus" and correlation_matrix.index[j] != "PersStatus":
            correlated_features.append((correlation_matrix.index[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))
            col1 = correlation_matrix.columns[i] 
            col2 = correlation_matrix.columns[j] 
            if importance[col1].values[0]>importance[col2].values[0]:
                if col2 not in to_drop:
                    to_drop.add(col2)
            elif importance[col2].values[0]>importance[col1].values[0]:
                if col1 not in to_drop:
                    to_drop.add(col1)

data = data.drop(columns=to_drop)
print(data['Time'])
data.to_csv(r"D:\Vittoria\Code\data\martin\labeled_features_no_correlation_martin_pers.csv", index = False)
#data.to_csv(r"D:\Vittoria\Code\data\martin\labeled_features_no_correlation_martin_glucose.csv", index = False)

correlation_matrix = data.select_dtypes(include=[np.number]).corr(method="spearman")

plt.figure(figsize=(12, 8))
sns.set(font_scale=0.5)
sns.heatmap(correlation_matrix, cmap="coolwarm", linewidths=0.5, xticklabels=True, yticklabels=True)
plt.title(f"Feature Correlation Matrix, Spearman, Threshold {threshold}, Martin, Ground Truth: Glucose")
#plt.show()
#plt.savefig(r"D:\Vittoria\Code\data\martin\after_drop_martin_glucose")
#plt.savefig(r"D:\Vittoria\Code\data\martin\after_drop_martin_pers")
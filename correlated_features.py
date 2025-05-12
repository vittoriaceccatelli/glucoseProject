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

data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features.csv")
demo = pd.read_csv(r"D:\Vittoria\Code\data\labeled_demographics.csv")
data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")
data = data.drop(columns=['DiabStatus', 'ID'])
data['Gender'] = data['Gender'].map({'MALE': 0, 'FEMALE': 1})
correlation_matrix = data.select_dtypes(include=[np.number]).corr(method="spearman")
correlation_matrix = correlation_matrix.drop(columns=['Glucose_Mean'], index=['Glucose_Mean'])


model = input("XGBoost or RandomForest? ")
label = input("Prediction or Classification? ")

correlation_matrix.to_csv(rf"D:\Vittoria\Code\data\correlation_matrix_{model}_{label}.csv", index = False)


#import labeled features and labeled demographics 
if model=="XGBoost": 
    if label=="Prediction":
        importance = pd.read_csv(r"D:\Vittoria\Code\data\sorted_importance_XGBoost_Prediction_Labeled_FoodYes.csv")
    elif label=="Classification":
        importance = pd.read_csv(r"D:\Vittoria\Code\data\sorted_importance_XGBoost_Classification_Labeled_FoodYes.csv")
elif model=="RandomForest": 
    if label=="Prediction":
        importance = pd.read_csv(r"D:\Vittoria\Code\data\sorted_importance_RandomForest_Prediction_Labeled_FoodYes.csv")
    elif label=="Classification":
        importance = pd.read_csv(r"D:\Vittoria\Code\data\sorted_importance_RandomForest_Classification_Labeled_FoodYes.csv")

importance = importance.set_index("Feature").T

plt.figure(figsize=(12, 8))
sns.set(font_scale=0.5)
sns.heatmap(correlation_matrix, cmap="coolwarm", linewidths=0.5, xticklabels=True, yticklabels=True)
plt.title(f"Feature Correlation Matrix, Spearman, {model}, {label}")
plt.tight_layout()
plt.savefig(rf"D:\Vittoria\Code\data\plots\correlation_plots\before_drop_{model}_{label}")
#plt.show()

threshold = 0.8 

correlated_features = []
to_drop = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
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
data.to_csv(rf"D:\Vittoria\Code\data\labeled_features_no_correlation_{model}_{label}.csv", index = False)

correlation_matrix = data.select_dtypes(include=[np.number]).corr(method="spearman")

plt.figure(figsize=(12, 8))
sns.set(font_scale=0.5)
sns.heatmap(correlation_matrix, cmap="coolwarm", linewidths=0.5, xticklabels=True, yticklabels=True)
plt.title(f"Feature Correlation Matrix, Spearman, Threshold {threshold}, {model}, {label}")
plt.tight_layout()
#plt.show()
plt.savefig(rf"D:\Vittoria\Code\data\plots\correlation_plots\after_drop_{model}_{label}")
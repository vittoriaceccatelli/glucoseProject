import pandas as pd 
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Clean up unused memory
gc.collect()
torch.cuda.empty_cache()

# Load feature and demographic datasets
data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features.csv")
demo = pd.read_csv(r"D:\Vittoria\Code\data\other\labeled_demographics.csv")

# Merge features with demographics on participant ID
data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")

# Drop unnecessary columns and encode gender
data = data.drop(columns=['DiabStatus', 'ID'])
data['Gender'] = data['Gender'].map({'MALE': 0, 'FEMALE': 1})

# Compute Spearman correlation matrix for all numerical features (excluding target)
correlation_matrix = data.select_dtypes(include=[np.number]).corr(method="spearman")
correlation_matrix = correlation_matrix.drop(columns=['Glucose_Mean'], index=['Glucose_Mean'])

# Ask user to specify model and task type
model = input("XGBoost or RandomForest? ")
label = input("Prediction or Classification? ")

# Save correlation matrix to CSV
correlation_matrix.to_csv(rf"D:\Vittoria\Code\data\correlation\correlation_matrix_{model}_{label}.csv", index=False)

# Load precomputed feature importance based on model and task
if model == "XGBoost":
    if label == "Prediction":
        importance = pd.read_csv(r"D:\Vittoria\Code\data\feature_importance\sorted_importance_XGBoost_Prediction_Labeled_FoodYes.csv")
    elif label == "Classification":
        importance = pd.read_csv(r"D:\Vittoria\Code\data\feature_importance\sorted_importance_XGBoost_Classification_Labeled_FoodYes.csv")
elif model == "RandomForest":
    if label == "Prediction":
        importance = pd.read_csv(r"D:\Vittoria\Code\data\feature_importance\sorted_importance_RandomForest_Prediction_Labeled_FoodYes.csv")
    elif label == "Classification":
        importance = pd.read_csv(r"D:\Vittoria\Code\data\feature_importance\sorted_importance_RandomForest_Classification_Labeled_FoodYes.csv")

# Transpose importance DataFrame for easier access
importance = importance.set_index("Feature").T

# Plot and save initial correlation heatmap
plt.figure(figsize=(12, 8))
sns.set(font_scale=0.5)
sns.heatmap(correlation_matrix, cmap="coolwarm", linewidths=0.5, xticklabels=True, yticklabels=True)
plt.title(f"Feature Correlation Plot", fontsize=14)
plt.tight_layout()
plt.savefig(rf"D:\Vittoria\Code\data\plots\correlation_plots\before_drop_{model}_{label}")
# plt.show()

# Define correlation threshold above which features are considered too similar
threshold = 0.8

# Initialize list for tracking correlated features to drop
correlated_features = []
to_drop = set()

# Iterate through upper triangle of correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            correlated_features.append((correlation_matrix.index[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            # Drop the less important feature between col1 and col2
            if importance[col1].values[0] > importance[col2].values[0]:
                if col2 not in to_drop:
                    to_drop.add(col2)
            elif importance[col2].values[0] > importance[col1].values[0]:
                if col1 not in to_drop:
                    to_drop.add(col1)

# Drop the selected correlated features from the dataset
data = data.drop(columns=to_drop)

# Optionally save the updated dataset
# data.to_csv(rf"D:\Vittoria\Code\data\correlation\labeled_features_no_correlation_{model}_{label}.csv", index=False)

# Recompute correlation matrix after feature removal
correlation_matrix = data.select_dtypes(include=[np.number]).corr(method="spearman")

# Rename label for plot title if regression
if label == "Prediction":
    l = "Regression"
else:
    l = label

# Plot and save final correlation heatmap after feature removal
plt.figure(figsize=(12, 8))
sns.set(font_scale=0.8)
sns.heatmap(correlation_matrix, cmap="coolwarm", linewidths=0.5, xticklabels=True, yticklabels=True)
plt.title(f"Feature Correlation Plot using feature importances from the {model} model for the {l} task", fontsize=14)
plt.tight_layout()
# plt.show()
plt.savefig(rf"D:\Vittoria\Code\data\plots\correlation_plots\after_drop_{model}_{label}")
import pandas as pd 
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import gc 

# Clear memory
gc.collect()
torch.cuda.empty_cache()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# Load Martin's features and demographic data
data = pd.read_csv(r"D:\Vittoria\Code\data\martin\martins_features.csv")
demo = pd.read_csv(r"D:\Vittoria\Code\data\other\labeled_demographics.csv")

# Load feature importance (choose between glucose or PersStatus ground truth)
# importance = pd.read_csv(r"D:\Vittoria\Code\data\martin\sorted_importance_with_martin_glucose.csv")
importance = pd.read_csv(r"D:\Vittoria\Code\data\martin\sorted_importance_with_martin_pers.csv")

# Transpose importance for easier access by column name
importance = importance.set_index("Feature").T

# Merge with demographics and clean up columns
data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")
data = data.drop(columns=['DiabStatus', 'ID'])

# Encode gender column to numeric
data['Gender'] = data['Gender'].map({'MALE': 0, 'FEMALE': 1})

# Compute Spearman correlation matrix for all numerical features
correlation_matrix = data.select_dtypes(include=[np.number]).corr(method="spearman")

# Optionally save correlation matrix to file
# correlation_matrix.to_csv(r"D:\Vittoria\Code\data\martin\correlation_matrix_martin_pers.csv", index = False)
# correlation_matrix.to_csv(r"D:\Vittoria\Code\data\martin\correlation_matrix_martin_glucose.csv", index = False)

# Visualize correlation matrix before feature removal
plt.figure(figsize=(12, 8))
sns.set(font_scale=0.5)
sns.heatmap(correlation_matrix, cmap="coolwarm", linewidths=0.5, xticklabels=True, yticklabels=True)
plt.title(f"Feature Correlation Matrix, Spearman, Glucose as ground truth")
# plt.savefig(r"D:\Vittoria\Code\data\martin\before_drop_martin_glucose")
# plt.savefig(r"D:\Vittoria\Code\data\martin\before_drop_martin_pers")
# plt.show()

# Define correlation threshold for identifying multicollinearity
threshold = 0.7

# Initialize tracking for correlated feature pairs and features to drop
correlated_features = []
to_drop = set()

# Iterate through upper triangle of the correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        # Exclude comparisons with target columns and check if correlation is above threshold
        if (
            abs(correlation_matrix.iloc[i, j]) > threshold and
            correlation_matrix.index[i] not in ["Glucose_Mean", "Time", "PersStatus"] and
            correlation_matrix.index[j] not in ["Glucose_Mean", "Time", "PersStatus"]
        ):
            correlated_features.append((correlation_matrix.index[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]

            # Drop the less important of the two correlated features
            if importance[col1].values[0] > importance[col2].values[0]:
                if col2 not in to_drop:
                    to_drop.add(col2)
            elif importance[col2].values[0] > importance[col1].values[0]:
                if col1 not in to_drop:
                    to_drop.add(col1)

# Drop identified correlated features from dataset
data = data.drop(columns=to_drop)

# Print time column to verify it's still in the dataset
print(data['Time'])

# Save the new dataset without highly correlated features
data.to_csv(r"D:\Vittoria\Code\data\martin\labeled_features_no_correlation_martin_pers.csv", index=False)
# data.to_csv(r"D:\Vittoria\Code\data\martin\labeled_features_no_correlation_martin_glucose.csv", index=False)

# Recompute correlation matrix after feature removal
correlation_matrix = data.select_dtypes(include=[np.number]).corr(method="spearman")

# Plot the new correlation matrix
plt.figure(figsize=(12, 8))
sns.set(font_scale=0.5)
sns.heatmap(correlation_matrix, cmap="coolwarm", linewidths=0.5, xticklabels=True, yticklabels=True)
plt.title(f"Feature Correlation Matrix, Spearman, Threshold {threshold}, Martin, Ground Truth: Glucose")
# plt.show()
# plt.savefig(r"D:\Vittoria\Code\data\martin\after_drop_martin_glucose")
# plt.savefig(r"D:\Vittoria\Code\data\martin\after_drop_martin_pers")
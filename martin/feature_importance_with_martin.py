import torch
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import gc
import torch 
from xgboost import XGBRegressor, XGBClassifier

#GPU
#gc.collect()
#torch.cuda.empty_cache()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#import balanced dataset 
data = pd.read_csv(r"D:\Vittoria\Code\data\martin\martins_features.csv")
demo = pd.read_csv(r"D:\Vittoria\Code\data\other\labeled_demographics.csv")
data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")
data['Gender'] = data['Gender'].map({'MALE': 0, 'FEMALE': 1})


#separate features and labels (labels: pers classes) and groups (participant IDs)
feature_columns = [col for col in data.columns if col not in ['Participant_ID', 'Time', 'PersStatus', 'Glucose_Mean', 'Unnamed: 0', 'DiabStatus', 'ID', 'index']]

X = data[feature_columns].values
y = data['PersStatus'].map({'PersNorm': 0, 'PersHigh': 1, 'PersLow': 2}).values
#y = data['Glucose_Mean']
groups = data['Participant_ID'].values

xgb_params = {
    "max_depth": 6,
    "n_estimators": 100,
    "learning_rate": 0.1,
    "random_state": 42
}

#use leave one group out (participant) and initiate a list to save feature_importances
logo = LeaveOneGroupOut()
feature_importances = []
for train_idx, test_idx in logo.split(X, y, groups):
    #separate into training and testing sets
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    #use RandomForestRegressor with 5 estimators and fit model on training data 
    #model = RandomForestClassifier(n_estimators=5, random_state=42)
    #model = RandomForestRegressor(n_estimators=5, random_state=42)
    model = XGBClassifier(**xgb_params)
    #model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    #append feature importance to list
    feature_importances.append(model.feature_importances_)

#calculate mean of feature importances 
feature_importances = np.array(feature_importances)
mean_importance = np.mean(feature_importances, axis=0)

#sort mean importances descending
sorted_idx = np.argsort(mean_importance)[::-1] #sorted indices
sorted_features = np.array(feature_columns)[sorted_idx] #sorted features 
sorted_importance = mean_importance[sorted_idx] #sorted importances
sorted_dict = dict(zip(sorted_features, sorted_importance))
sorted_df = pd.DataFrame(sorted_dict.items(), columns=["Feature", "Importance"])
#sorted_df.to_csv(r"D:\Vittoria\Code\data\martin\sorted_importance_with_martin_glucose.csv", index=False)
#sorted_df.to_csv(r"D:\Vittoria\Code\data\martin\sorted_importance_with_martin_pers.csv", index=False)

#plot a bar plot with all features and their importances in descending order
plt.figure(figsize=(10, 6))
plt.bar(sorted_features, sorted_importance, color='skyblue')
plt.xticks(rotation=90, ha='right', fontsize=5)
plt.ylabel("Feature Importance")
plt.title("Feature Importance Bar Plot, Ground Truth: PersStatus")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
#plt.savefig(r"D:\Vittoria\Code\data\martin\feature_importance_martin_glucose.png")
#plt.savefig(r"D:\Vittoria\Code\data\martin\feature_importance_martin_pers.png")


#categorise all features in following categories
feature_categories = {
    "food": ["Calories_2h","Protein_2h","Sugar_2h","Carbs_2h","Eat_cnt_2h", "Eat_mean_2h","Eat","Calories_8h","Protein_8h","Sugar_8h","Carbs_8h","Eat_cnt_8h","Eat_mean_8h","Calories_24h", "Protein_24h","Sugar_24h","Carbs_24h","Eat_cnt_24h","Eat_mean_24h"], 
    "circadian rhythm": ["Wake_Time", "Hours_from_midnight","Minutes_from_midnight"],
    "stress": ['HRV_CVNN', 'HRV_MeanNN', 'HRV_SDSD', 'HRV_Max', 'HRV_Min', 'HRV_Mean', 'HRV_Median', 'BVP_PPI_Mean', 'BVP_PPI_STD', 'BVP_Amplitude_Mean', ' ibi', ' bvp', "HRV_Max","HRV_Min","HRV_Mean","HRV_Median","SDNN","RMSSD","NNX","PNNX"],
    "activity": ['ACC_Median', 'ACC_5th', 'ACC_95th', 'acc_vector_mag', "Activity Bouts","Activity Bouts_24hr","Activity Bouts_1hr", "ACC_Mean","ACC_Std","ACC_Min","ACC_Max","ACC_q1G","ACC_q3G","ACC_skew","ACC_Mean2hrs","ACC_Max2hrs"],
    "temperature": ['Temp_Median', 'Temp_5th', 'Temp_95th', 'Temp_Variability', ' temp', "TEMP_Mean","TEMP_Std","TEMP_Min","TEMP_Max","TEMP_q1G","TEMP_q3G","TEMP_skew"],
    "heart rate": ['HR_Variability', ' hr', "HR_Mean","HR_Std","HR_Min","HR_Max","HR_q1G","HR_q3G","HR_skew"],
    "electro-dermal activity": ['EDA_Median', 'EDA_5th', 'EDA_95th', ' eda', "Peak_eda_5min_sum","Peak_eda_2hr_mean","EDA_Mean","EDA_Std","EDA_Min","EDA_Max","EDA_q1G","EDA_q3G","EDA_skew"],
    "personalization": ["ID"],
    "sex": ["Gender"],
    "HbA1c": ["HbA1c"]
}

#categorise all features in following sources
source = {
    "food log":["Calories_2h","Protein_2h","Sugar_2h","Carbs_2h","Eat_cnt_2h", "Eat_mean_2h","Eat","Calories_8h","Protein_8h","Sugar_8h","Carbs_8h","Eat_cnt_8h","Eat_mean_8h","Calories_24h", "Protein_24h","Sugar_24h","Carbs_24h","Eat_cnt_24h","Eat_mean_24h"], 
    "wearable": ['EDA_Median', 'EDA_5th', 'EDA_95th', ' eda', "Peak_eda_5min_sum","Peak_eda_2hr_mean","EDA_Mean","EDA_Std","EDA_Min","EDA_Max","EDA_q1G","EDA_q3G","EDA_skew", 'HR_Variability', ' hr', "HR_Mean","HR_Std","HR_Min","HR_Max","HR_q1G","HR_q3G","HR_skew", 'Temp_Median', 'Temp_5th', 'Temp_95th', 'Temp_Variability', ' temp', "TEMP_Mean","TEMP_Std","TEMP_Min","TEMP_Max","TEMP_q1G","TEMP_q3G","TEMP_skew", 'ACC_Median', 'ACC_5th', 'ACC_95th', 'acc_vector_mag', "Activity Bouts","Activity Bouts_24hr","Activity Bouts_1hr", "ACC_Mean","ACC_Std","ACC_Min","ACC_Max","ACC_q1G","ACC_q3G","ACC_skew","ACC_Mean2hrs","ACC_Max2hrs", 'HRV_CVNN', 'HRV_MeanNN', 'HRV_SDSD', 'HRV_Max', 'HRV_Min', 'HRV_Mean', 'HRV_Median', 'BVP_PPI_Mean', 'BVP_PPI_STD', 'BVP_Amplitude_Mean', ' ibi', ' bvp', "HRV_Max","HRV_Min","HRV_Mean","HRV_Median","SDNN","RMSSD","NNX","PNNX", "Wake_Time", "Hours_from_midnight","Minutes_from_midnight"],
    "model": ["ID"],
    "user-defined": ["HbA1c","Gender"]
}

category_importance = {category: 0 for category in feature_categories} #initialise importance of all categories to 0

total_importance = np.sum(mean_importance) #sum all mean_importances to get ideally 100%
feature_name_to_index = {name: idx for idx, name in enumerate(feature_columns)} #of all features initialise a dictionary

#iterate over all features and sum importances of all features in a category 
for category, features in feature_categories.items():
    category_importance[category] = np.sum(
        [mean_importance[feature_name_to_index[f]] for f in features if f in feature_name_to_index]
    )

#calculate summed importance of all categories
category_importance_percent = {k: (v / total_importance) * 100 for k, v in category_importance.items()}

#create a dataframe with the category and the importance 
importance_df = pd.DataFrame(list(category_importance_percent.items()), columns=["Category", "Importance"])


source_importance = {category: 0 for category in source} #initialise importance of all sources to 0

#iterate over all features and sum importances of all features in a source 
for category, features in source.items():
    source_importance[category] = np.sum(
        [mean_importance[feature_name_to_index[f]] for f in features if f in feature_name_to_index]
    )

#calculate summed importance of all sources
source_importance_percent = {k: (v / total_importance) * 100 for k, v in source_importance.items()}

#create a dataframe with the source and the importance 
importance_df_source = pd.DataFrame(list(source_importance_percent.items()), columns=["Source", "Importance"])

fig, ax = plt.subplots(figsize=(24, 12))
size = 0.3


outer_colors = ["indianred", "peru", "olive", "lightseagreen", "royalblue", "mediumpurple", "palevioletred", "teal", "forestgreen", "antiquewhite"]
inner_colors = ["orangered", "gold", "limegreen", "deepskyblue"]

# Create outer labels with both category and percentage combined
outer_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(importance_df['Category'], 
                                                                         importance_df['Importance'] / importance_df['Importance'].sum() * 100)]
inner_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(importance_df_source['Source'], 
                                                                         importance_df_source['Importance'] / importance_df_source['Importance'].sum() * 100)
                                                                         if percentage!=0.00]

# Plot the outer pie chart
mypie, _ = ax.pie(importance_df['Importance'], 
       startangle=90, 
       radius=1, 
       colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'), 
       labeldistance=0.85, 
       textprops={'fontsize': 8})
plt.setp(mypie, width=0.3)

# Plot the inner pie chart
d = pd.Series([i for i in importance_df_source['Importance'].tolist() if i!=0.00])
mypie2, _ = ax.pie(d,
       labels=inner_labels,
       startangle=90, 
       radius=1-size, 
       colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'), 
       labeldistance=0.55, 
       textprops={'fontsize': 12})
plt.setp(mypie2, width=0.4)
plt.margins(0,0)

# Add a legend for the inner pie chart
# Creating custom legend handles using mpatches
other = [mpatches.Patch(color=color, label=label) for color, label in zip(outer_colors, outer_labels)]
ax.legend(handles=other, loc='upper left', bbox_to_anchor=(0.75, 1), fontsize=12)

ax.set(aspect="equal")  # Ensures the pie chart remains circular
ax.set_title('Feature Importance Pie Plot, Ground Truth: PersStatus', fontsize=24)

# Adjust layout and display plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#plt.savefig(r"D:\Vittoria\Code\data\plots\feature_importance_pie_pers.png")
#plt.savefig(r"D:\Vittoria\Code\data\plots\feature_importance_pie_glucose.png")
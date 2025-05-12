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
data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features_no_correlation_pers_importance.csv")
#data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features_no_correlation_glucose_importance.csv")
data['Gender'] = data['Gender'].map({'MALE': 0, 'FEMALE': 1})


#separate features and labels (labels: pers classes) and groups (participant IDs)
feature_columns = [col for col in data.columns if col not in ['Time', 'Glucose_Mean', 'PersStatus', 'DiabStatus']]

#drop food log features
print(feature_columns)
feature_columns = feature_columns[:25] + feature_columns[30:]
print(feature_columns)

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
    model = RandomForestClassifier(n_estimators=10, random_state=42)
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

#plot a bar plot with all features and their importances in descending order
plt.figure(figsize=(10, 6)) 
plt.bar(sorted_features, sorted_importance, color='skyblue')
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.ylabel("Feature Importance")
plt.title("Feature Importance Bar Plot, Ground Truth: Interstitial Glucose Values")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
#plt.savefig(r"D:\Vittoria\Code\data\plots\feature_importance_glucose_after_correlation.png")
#plt.savefig(r"D:\Vittoria\Code\data\plots\feature_importance_pers_after_correlation.png")
#categorise all features in following categories














#PERS
feature_categories = {
    "food": ['Eat_mean_2h', 'Carbs_24h', 'Eat_mean_24h', 'Protein_24h', 'Eat_mean_8h'], 
    "circadian rhythm": ['Minutes_from_midnight', 'Wake_Time'],
    "stress": ['NNX', 'HRV_Mean', 'RMSSD'],
    "activity": ['Activity Bouts', 'ACC_skew', 'Activity Bouts_1hr', 'ACC_Std', 'ACC_Mean2hrs', 'Activity Bouts_24hr', 'ACC_Max2hrs', 'ACC_q1G', 'ACC_Mean'],
    "temperature": ['TEMP_skew', 'TEMP_Max', 'TEMP_Std'],
    "heart rate": ['HR_skew', 'HR_Std'],
    "electro-dermal activity": ['Peak_eda_5min_sum', 'EDA_skew', 'Peak_eda_2hr_sum', 'EDA_q1G', 'EDA_Std'],
    "personalization": ['Participant_ID'],
    "sex": ['Gender'],
    "HbA1c": ['HbA1c']
}

#categorise all features in following sources
source = {
    "food log": ['Eat_mean_2h', 'Carbs_24h', 'Eat_mean_24h', 'Protein_24h', 'Eat_mean_8h'], 
    "wearable": ['Peak_eda_5min_sum', 'EDA_skew', 'Peak_eda_2hr_sum', 'EDA_q1G', 'EDA_Std','HR_skew', 'HR_Std', 'TEMP_skew', 'TEMP_Max', 'TEMP_Std', 'Activity Bouts', 'ACC_skew', 'Activity Bouts_1hr', 'ACC_Std', 'ACC_Mean2hrs', 'Activity Bouts_24hr', 'ACC_Max2hrs', 'ACC_q1G', 'ACC_Mean', 'Minutes_from_midnight', 'Wake_Time', 'NNX', 'HRV_Mean', 'RMSSD'],
    "model": ["Participant_ID"],
    "user-defined": ["HbA1c","Gender"]
}







#GLUCOSE
feature_categories = {
    "food": ["Protein_2h","Calories_8h","Protein_24h","Eat_mean_24h"], 
    "circadian rhythm": ["Wake_Time", "Minutes_from_midnight"],
    "stress": ["HRV_Mean","SDNN","NNX","PNNX"],
    "activity": ["Activity Bouts","Activity Bouts_24hr","Activity Bouts_1hr", "ACC_Std","ACC_q1G","ACC_q3G","ACC_skew","ACC_Mean2hrs","ACC_Max2hrs"],
    "temperature": ["TEMP_Std","TEMP_Max","TEMP_skew"],
    "heart rate": ["HR_skew"],
    "electro-dermal activity": ["Peak_eda_5min_sum","Peak_eda_2hr_mean","EDA_Std","EDA_Max","EDA_q1G","EDA_skew"],
    "personalization": ["Participant_ID"],
    "sex": ["Gender"],
    "HbA1c": ["HbA1c"]
}
#categorise all features in following sources
source = {
    "food log": ["Protein_2h","Calories_8h","Protein_24h","Eat_mean_24h"], 
    "wearable": ["Peak_eda_5min_sum","Peak_eda_2hr_mean","EDA_Std","EDA_Max","EDA_q1G","EDA_skew", "HR_skew", "Wake_Time", "Minutes_from_midnight", "HRV_Mean","SDNN","NNX","PNNX", "Activity Bouts","Activity Bouts_24hr","Activity Bouts_1hr", "ACC_Std","ACC_q1G","ACC_q3G","ACC_skew","ACC_Mean2hrs","ACC_Max2hrs", "TEMP_Std","TEMP_Max","TEMP_skew"],
    "model": ["Participant_ID"],
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
mypie2, texts = ax.pie(importance_df_source['Importance'],
       startangle=90, 
       radius=1-size, 
       colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'), 
       labeldistance=0.55, 
       textprops={'fontsize': 10})
plt.setp(mypie2, width=0.4)
plt.margins(0,0)

# Add a legend for the inner pie chart
# Creating custom legend handles using mpatches
outer = [mpatches.Patch(color=color, label=label) for color, label in zip(outer_colors, outer_labels)]
inner = [mpatches.Patch(color=color, label=label) for color, label in zip(inner_colors, inner_labels)]
other = outer+inner
ax.legend(handles=other, loc='upper left', bbox_to_anchor=(0.75, 1), fontsize=10)

ax.set(aspect="equal")  # Ensures the pie chart remains circular
ax.set_title('Feature Importance Pie Plot, Ground Truth: Interstitial Glucose Values', fontsize=24)

# Adjust layout and display plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#plt.savefig(r"D:\Vittoria\Code\data\plots\feature_importance_pie_pers_after_correlation.png")
#plt.savefig(r"D:\Vittoria\Code\data\plots\feature_importance_pie_glucose_after_correlation.png")
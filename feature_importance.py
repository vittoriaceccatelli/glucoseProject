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
import matplotlib.colors as mcolors

#GPU
#gc.collect()
#torch.cuda.empty_cache()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def loop_through_all(chosen_model, label, dataset, foodfeatures):
    #import dataset 
    if dataset=="Labeled":
        data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features.csv")
        #demo = pd.read_csv(r"D:\Vittoria\Code\data\labeled_demographics.csv")
        #data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")
        data['Gender'] = data['Gender'].map({'MALE': 0, 'FEMALE': 1})
        feature_columns = [col for col in data.columns if col not in ['Time', 'PersStatus', 'Glucose_Mean', 'DiabStatus', 'ID']]
        if foodfeatures=="FoodNo":
            feature_columns = feature_columns[:48] + feature_columns[67:]
    elif dataset=="NoCorrelation":
        if label=="Prediction":
            if chosen_model=="XGBoost":
                data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features_no_correlation_XGBoost_Prediction.csv")
                feature_columns = [col for col in data.columns if col not in ['Time', 'PersStatus', 'Glucose_Mean', 'DiabStatus', 'ID']]
                if foodfeatures=="FoodNo":
                    feature_columns = feature_columns[:26] + feature_columns[30:]
            elif chosen_model=="RandomForest":
                data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features_no_correlation_RandomForest_Prediction.csv")
                feature_columns = [col for col in data.columns if col not in ['Time', 'PersStatus', 'Glucose_Mean', 'DiabStatus', 'ID']]
                if foodfeatures=="FoodNo":
                    feature_columns = feature_columns[:26] + feature_columns[31:]
        elif label=="Classification":
            if chosen_model=="XGBoost":
                data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features_no_correlation_XGBoost_Classification.csv")
                feature_columns = [col for col in data.columns if col not in ['Time', 'PersStatus', 'Glucose_Mean', 'DiabStatus', 'ID']]
                if foodfeatures=="FoodNo":
                    feature_columns = feature_columns[:25] + feature_columns[31:]
            elif chosen_model=="RandomForest":
                data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features_no_correlation_RandomForest_Classification.csv")
                feature_columns = [col for col in data.columns if col not in ['Time', 'PersStatus', 'Glucose_Mean', 'DiabStatus', 'ID']]
                if foodfeatures=="FoodNo":
                    feature_columns = feature_columns[:25] + feature_columns[30:]

    X = data[feature_columns].values

    if label=="Prediction":
        y = data['Glucose_Mean']
    elif label=="Classification":
        y = data['PersStatus'].map({'PersNorm': 0, 'PersHigh': 1, 'PersLow': 2}).values

    groups = data['Participant_ID'].values

    #use leave one group out (participant) and initiate a list to save feature_importances
    logo = LeaveOneGroupOut()
    feature_importances = []

    xgb_params = {
        "max_depth": 6,
        "n_estimators": 100,
        "learning_rate": 0.1,
        "random_state": 42
    }

    for train_idx, test_idx in logo.split(X, y, groups):
        #separate into training and testing sets
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        #use RandomForestRegressor with 100 estimators and fit model on training data 
        if chosen_model=="XGBoost":
            if label=="Prediction":
                model = XGBRegressor(**xgb_params)
            elif label=="Classification":
                model = XGBClassifier(**xgb_params)
        elif chosen_model=="RandomForest":
            if label=="Prediction":
                model = RandomForestRegressor(n_estimators=10, random_state=42)
            elif label=="Classification":
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
    #if dataset == "Labeled":
        #sorted_df.to_csv(rf"D:\Vittoria\Code\data\sorted_importance_{chosen_model}_{label}_{dataset}_{foodfeatures}.csv", index=False)

    #plot a bar plot with all features and their importances in descending order
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_features, sorted_importance, color='skyblue')
    plt.xticks(rotation=90, ha='right', fontsize=5)
    plt.ylabel("Feature Importance")
    plt.title(f"Feature Importance Bar Plot, {chosen_model}, {label}, {dataset}, {foodfeatures}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    #plt.savefig(fr"D:\Vittoria\Code\data\plots\feature_importance_bar\{chosen_model}_{label}_{dataset}_{foodfeatures}.png")

    if dataset=="Labeled":
        #categorise all features in following categories
        feature_categories = {
            "food": ["Calories_2h","Protein_2h","Sugar_2h","Carbs_2h","Eat_cnt_2h", "Eat_mean_2h","Eat","Calories_8h","Protein_8h","Sugar_8h","Carbs_8h","Eat_cnt_8h","Eat_mean_8h","Calories_24h", "Protein_24h","Sugar_24h","Carbs_24h","Eat_cnt_24h","Eat_mean_24h"], 
            "circadian rhythm": ["Wake_Time", "Hours_from_midnight","Minutes_from_midnight"],
            "stress": ["HRV_Max","HRV_Min","HRV_Mean","HRV_Median","SDNN","RMSSD","NNX","PNNX"],
            "activity": ["Activity Bouts","Activity Bouts_24hr","Activity Bouts_1hr", "ACC_Mean","ACC_Std","ACC_Min","ACC_Max","ACC_q1G","ACC_q3G","ACC_skew","ACC_Mean2hrs","ACC_Max2hrs"],
            "temperature": ["TEMP_Mean","TEMP_Std","TEMP_Min","TEMP_Max","TEMP_q1G","TEMP_q3G","TEMP_skew"],
            "heart rate": ["HR_Mean","HR_Std","HR_Min","HR_Max","HR_q1G","HR_q3G","HR_skew"],
            "electro-dermal activity": ["Peak_eda_5min_sum","Peak_eda_2hr_sum","Peak_eda_2hr_mean","EDA_Mean","EDA_Std","EDA_Min","EDA_Max","EDA_q1G","EDA_q3G","EDA_skew"],
            "personalization": ["Participant_ID"],
            "sex": ["Gender"],
            "HbA1c": ["HbA1c"]
        }

        #categorise all features in following sources
        source = {
            "food log":["Calories_2h","Protein_2h","Sugar_2h","Carbs_2h","Eat_cnt_2h", "Eat_mean_2h","Eat","Calories_8h","Protein_8h","Sugar_8h","Carbs_8h","Eat_cnt_8h","Eat_mean_8h","Calories_24h", "Protein_24h","Sugar_24h","Carbs_24h","Eat_cnt_24h","Eat_mean_24h"], 
            "wearable": ["Peak_eda_5min_sum","Peak_eda_2hr_sum","Peak_eda_2hr_mean","EDA_Mean","EDA_Std","EDA_Min","EDA_Max","EDA_q1G","EDA_q3G","EDA_skew","HR_Mean","HR_Std","HR_Min","HR_Max","HR_q1G","HR_q3G","HR_skew","TEMP_Mean","TEMP_Std","TEMP_Min","TEMP_Max","TEMP_q1G","TEMP_q3G","TEMP_skew","HRV_Max","HRV_Min","HRV_Mean","HRV_Median","SDNN","RMSSD","NNX","PNNX", "Wake_Time", "Hours_from_midnight","Minutes_from_midnight", "Activity Bouts","Activity Bouts_24hr","Activity Bouts_1hr", "ACC_Mean","ACC_Std","ACC_Min","ACC_Max","ACC_q1G","ACC_q3G","ACC_skew","ACC_Mean2hrs","ACC_Max2hrs"],
            "model": ["Participant_ID"],
            "user-defined": ["HbA1c","Gender"]
        }
        if foodfeatures=="FoodNo":
            feature_categories = {
                "circadian rhythm": ["Wake_Time", "Hours_from_midnight","Minutes_from_midnight"],
                "stress": ["HRV_Max","HRV_Min","HRV_Mean","HRV_Median","SDNN","RMSSD","NNX","PNNX"],
                "activity": ["Activity Bouts","Activity Bouts_24hr","Activity Bouts_1hr", "ACC_Mean","ACC_Std","ACC_Min","ACC_Max","ACC_q1G","ACC_q3G","ACC_skew","ACC_Mean2hrs","ACC_Max2hrs"],
                "temperature": ["TEMP_Mean","TEMP_Std","TEMP_Min","TEMP_Max","TEMP_q1G","TEMP_q3G","TEMP_skew"],
                "heart rate": ["HR_Mean","HR_Std","HR_Min","HR_Max","HR_q1G","HR_q3G","HR_skew"],
                "electro-dermal activity": ["Peak_eda_5min_sum","Peak_eda_2hr_sum","Peak_eda_2hr_mean","EDA_Mean","EDA_Std","EDA_Min","EDA_Max","EDA_q1G","EDA_q3G","EDA_skew"],
                "personalization": ["Participant_ID"],
                "sex": ["Gender"],
                "HbA1c": ["HbA1c"]
            }

            #categorise all features in following sources
            source = {
                "wearable": ["Peak_eda_5min_sum","Peak_eda_2hr_sum","Peak_eda_2hr_mean","EDA_Mean","EDA_Std","EDA_Min","EDA_Max","EDA_q1G","EDA_q3G","EDA_skew","HR_Mean","HR_Std","HR_Min","HR_Max","HR_q1G","HR_q3G","HR_skew","TEMP_Mean","TEMP_Std","TEMP_Min","TEMP_Max","TEMP_q1G","TEMP_q3G","TEMP_skew","HRV_Max","HRV_Min","HRV_Mean","HRV_Median","SDNN","RMSSD","NNX","PNNX", "Wake_Time", "Hours_from_midnight","Minutes_from_midnight", "Activity Bouts","Activity Bouts_24hr","Activity Bouts_1hr", "ACC_Mean","ACC_Std","ACC_Min","ACC_Max","ACC_q1G","ACC_q3G","ACC_skew","ACC_Mean2hrs","ACC_Max2hrs"],
                "model": ["Participant_ID"],
                "user-defined": ["HbA1c","Gender"]
            }
    elif dataset=="NoCorrelation":
        if label=="Prediction":
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
            if foodfeatures=="FoodNo":
                feature_categories = {
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
                    "wearable": ["Peak_eda_5min_sum","Peak_eda_2hr_mean","EDA_Std","EDA_Max","EDA_q1G","EDA_skew", "HR_skew", "Wake_Time", "Minutes_from_midnight", "HRV_Mean","SDNN","NNX","PNNX", "Activity Bouts","Activity Bouts_24hr","Activity Bouts_1hr", "ACC_Std","ACC_q1G","ACC_q3G","ACC_skew","ACC_Mean2hrs","ACC_Max2hrs", "TEMP_Std","TEMP_Max","TEMP_skew"],
                    "model": ["Participant_ID"],
                    "user-defined": ["HbA1c","Gender"]
                }
        elif label=="Classification":
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
            if foodfeatures=="FoodNo":
                feature_categories = {
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
                    "wearable": ['Peak_eda_5min_sum', 'EDA_skew', 'Peak_eda_2hr_sum', 'EDA_q1G', 'EDA_Std','HR_skew', 'HR_Std', 'TEMP_skew', 'TEMP_Max', 'TEMP_Std', 'Activity Bouts', 'ACC_skew', 'Activity Bouts_1hr', 'ACC_Std', 'ACC_Mean2hrs', 'Activity Bouts_24hr', 'ACC_Max2hrs', 'ACC_q1G', 'ACC_Mean', 'Minutes_from_midnight', 'Wake_Time', 'NNX', 'HRV_Mean', 'RMSSD'],
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

    # Filter zero-importance sources
    filtered_source_df = importance_df_source[importance_df_source['Importance'] != 0.00]

    # Dynamically assign colors
    outer_colors = plt.cm.tab20.colors[:len(importance_df)]
    inner_colors = plt.cm.Pastel1.colors[:len(filtered_source_df)]

    # Format labels
    outer_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(
        importance_df['Category'], importance_df['Importance'])]


    inner_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(
        filtered_source_df['Source'], filtered_source_df['Importance'])]

    # Outer pie
    mypie, _ = ax.pie(importance_df['Importance'], 
                    startangle=90, radius=1, colors=outer_colors,
                    wedgeprops=dict(width=size, edgecolor='w'), 
                    labeldistance=0.85, textprops={'fontsize': 8})


    # Inner pie
    mypie2, _ = ax.pie(filtered_source_df['Importance'], 
                    labels=inner_labels,
                    startangle=90, radius=1-size, colors=inner_colors,
                    wedgeprops=dict(width=size, edgecolor='w'), 
                    labeldistance=0.55, textprops={'fontsize': 12})

    # Legend
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(outer_colors, outer_labels)]
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0.75, 1), fontsize=12)

    ax.set(aspect="equal")
    ax.set_title(f'Feature Importance Pie Plot, {chosen_model}, {label}, {dataset}, {foodfeatures}', fontsize=24)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    #plt.savefig(rf"D:\Vittoria\Code\data\plots\feature_importance_pie\{chosen_model}_{label}_{dataset}_{foodfeatures}.png")

#chosen_model = input("XGBoost or RandomForest? ")
#label = input("Prediction or Classification? ")
#dataset = input("Labeled or NoCorrelation? ")
#foodfeatures = input("FoodYes or FoodNo? ")

chosen_model = ["RandomForest", "XGBoost"]
label = ["Prediction", "Classification"]
dataset = ["Labeled", "NoCorrelation"]
foodfeatures = ["FoodYes", "FoodNo"]

for model in chosen_model:
    for l in label:
        for d in dataset:
            for f in foodfeatures:
                loop_through_all(model, l, d, f)
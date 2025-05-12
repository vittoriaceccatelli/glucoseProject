import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest 
from numpy import random
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#load data
result_dfs = []
dexcom_data = pd.read_csv(rf'D:\Vittoria\Code\data\dexcom\dexcom_data_5mins.csv')
#loop through participants 

# Assuming 'dexcom_data' already loaded from your file
dexcom_data['Timestamp'] = pd.to_datetime(dexcom_data['Timestamp'])  # Ensure the timestamp is a datetime object
dexcom_data.set_index('Timestamp', inplace=True)  # Set the timestamp as the index


dexcom_data = dexcom_data.groupby(dexcom_data.index).mean()  # Taking the mean of duplicate timestamps

# Now we can resample the data to 1-second intervals and forward-fill
dexcom_data_resampled = dexcom_data.resample('1S').ffill()  # Resample to 1-second intervals, forward filling

# Reset the index so that 'Timestamp' becomes a column and 'Participant_ID' is accessible
dexcom_data_resampled = dexcom_data_resampled.reset_index()

print(dexcom_data_resampled)
# Now, apply your glucose classification to the resampled data

# Rewriting the classify_glucose function so it works on the resampled data
def classify_glucose(df):
    df['PersStatus'] = 'PersNorm'

    # Calculate rolling mean and std for the last day, using a 5-minute window (288 points if 5-minute data)
    rolling_mean = df['Glucose_Mean'].rolling(window=288, min_periods=1).mean()
    rolling_std = df['Glucose_Mean'].rolling(window=288, min_periods=1).std()

    df['HighThreshold'] = rolling_mean + rolling_std
    df['LowThreshold'] = rolling_mean - rolling_std

    df.loc[df['Glucose_Mean'] > df['HighThreshold'], 'PersStatus'] = 'PersHigh'
    df.loc[df['Glucose_Mean'] < df['LowThreshold'], 'PersStatus'] = 'PersLow'
    
    return df

# Process each participant's data as before
result_dfs = []
for part in range(1, 17):
    # get participant-specific data
    data_part = dexcom_data_resampled[dexcom_data_resampled["Participant_ID"] == part]
    
    # Classify glucose values and append to the result list
    data_part_classified = classify_glucose(data_part)
    result_dfs.append(data_part_classified)

# Concatenate all participants' data and drop unnecessary columns
final_classified_df = pd.concat(result_dfs, ignore_index=True)
final_classified_df = final_classified_df.drop(columns=["LowThreshold", "HighThreshold"])

final_classified_df.to_csv(r'data\dexcom\dexcom_classified_zdenek.csv', index=False)



#figure with three boxplots for each participant characterizing persnorm, pershigh and perslow
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=final_classified_df, 
    x="Participant_ID", 
    y="Glucose_Mean", 
    hue="PersStatus",
    flierprops={"marker": "."},
    hue_order=["PersHigh", "PersNorm", "PersLow"], 
    order=sorted(final_classified_df["Participant_ID"].unique())
)
plt.xlabel("Participant ID")
plt.ylabel("Interstitial Glucose")
plt.title("Glucose Levels by Participant and Persistence Status")
plt.legend(title="Persistence Status")
plt.xticks(rotation=45)
file_save = rf'D:\Vittoria\Code\data\plots\pers_plots\pers_plots_per_participant'
#plt.savefig(file_save)


#subsets of persData divided into persnorm, pershigh and perslow
pers_norm = final_classified_df[final_classified_df['PersStatus']=="PersNorm"]
pers_high = final_classified_df[final_classified_df['PersStatus']=="PersHigh"]
pers_low = final_classified_df[final_classified_df['PersStatus']=="PersLow"]

#count number of persnorm, pershigh and perslow datapoints
count_norm = pers_norm.Glucose_Mean.value_counts()
count_high = pers_high.Glucose_Mean.value_counts()
count_low = pers_low.Glucose_Mean.value_counts()

#figure showing count of interstitial glucose values divided into different pers classes
plt.figure(figsize=(10, 5))
plt.bar(count_norm.index, count_norm, color = "gray")
plt.bar(count_high.index, count_high, color = "red")
plt.bar(count_low.index, count_low, color = "green")
file_save = rf'D:\Vittoria\Code\data\plots\pers_plots\pers_plots'
#plt.show()
#plt.savefig(file_save)

#Kolmogorov-Smirnov for persNorm vs normal distribution
#prints mean, std, min, max and skew
ks_norm = kstest((pers_norm['Glucose_Mean'].dropna() - pers_norm['Glucose_Mean'].mean()) / pers_norm['Glucose_Mean'].std(), 'norm')
print(pers_norm['Glucose_Mean'].mean())
print(pers_norm['Glucose_Mean'].std())
print(pers_norm['Glucose_Mean'].min())
print(pers_norm['Glucose_Mean'].max())
print(pers_norm['Glucose_Mean'].skew())
print(ks_norm)
print()

#Kolmogorov-Smirnov for persHigh vs normal distribution
#prints mean, std, min, max and skew
ks_high = kstest((pers_high['Glucose_Mean'].dropna() - pers_high['Glucose_Mean'].mean()) / pers_high['Glucose_Mean'].std(), 'norm')
print(pers_high['Glucose_Mean'].mean())
print(pers_high['Glucose_Mean'].std())
print(pers_high['Glucose_Mean'].min())
print(pers_high['Glucose_Mean'].max())
print(pers_high['Glucose_Mean'].skew())
print(ks_high)
print()

#Kolmogorov-Smirnov for persLow vs normal distribution
#prints mean, std, min, max and skew
ks_low = kstest((pers_low['Glucose_Mean'].dropna() - pers_low['Glucose_Mean'].mean()) / pers_low['Glucose_Mean'].std(), 'norm')
print(pers_low['Glucose_Mean'].mean())
print(pers_low['Glucose_Mean'].std())
print(pers_low['Glucose_Mean'].min())
print(pers_low['Glucose_Mean'].max())
print(pers_low['Glucose_Mean'].skew())
print(ks_low)
print()
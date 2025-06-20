import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest 
from numpy import random
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


#classify glucose into Pers status
def classify_glucose(df):
    #first assign all datapoints PersNorm status
    df['PersStatus'] = 'PersNorm'

    #calculate mean and std of last day, for first datapoints smaller window is used 
    rolling_mean = df['Glucose_Mean'].rolling(window=288, min_periods=1).mean()
    rolling_std = df['Glucose_Mean'].rolling(window=288, min_periods=1).std()

    #calculate mean + one std and mean - one std
    df['HighThreshold'] = rolling_mean + rolling_std
    df['LowThreshold'] = rolling_mean - rolling_std

    #assign pershigh if higher than highthreshold and perslow is lower than lowthreshold 
    df.loc[df['Glucose_Mean'] > df['HighThreshold'], 'PersStatus'] = 'PersHigh'
    df.loc[df['Glucose_Mean'] < df['LowThreshold'], 'PersStatus'] = 'PersLow'
    return df

#load data
result_dfs = []
dexcom_data = pd.read_csv(rf'D:\Vittoria\Code\data\dexcom\dexcom_data_5mins.csv')
#loop through participants 
for part in range (1, 17):
    #get participant specific data
    data_part = dexcom_data[dexcom_data["Participant_ID"] == part]
    #classify glucose values and append to list
    data_part_classified = classify_glucose(data_part)
    result_dfs.append(data_part_classified)

#concatenate all particiapnts, drop irrelevant columns and save to csv
final_classified_df = pd.concat(result_dfs, ignore_index=True)
final_classified_df = final_classified_df.drop(columns=["LowThreshold","HighThreshold"])
#final_classified_df.to_csv(r'data\dexcom\dexcom_classified.csv', index=False)

classified_and_nonzero = final_classified_df[final_classified_df["Glucose_Mean"].notna()] 
#classified_and_nonzero.to_csv(r'data\dexcom\dexcom_classified_and_nonzero.csv', index=False)



#figure with three boxplots for each participant characterizing persnorm, pershigh and perslow
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=classified_and_nonzero, 
    x="Participant_ID", 
    y="Glucose_Mean", 
    hue="PersStatus",
    flierprops={"marker": "."},
    hue_order=["PersHigh", "PersNorm", "PersLow"], 
    order=sorted(classified_and_nonzero["Participant_ID"].unique())
)
plt.xlabel("Participant ID", fontsize=12)
plt.ylabel("Interstitial Glucose Values [mg/dL]", fontsize=12)
plt.legend()
file_save = rf'D:\Vittoria\Code\data\plots\pers_plots\pers_plots_per_participant'
plt.savefig(file_save)


#subsets of persData divided into persnorm, pershigh and perslow
pers_norm = classified_and_nonzero[classified_and_nonzero['PersStatus']=="PersNorm"]
pers_high = classified_and_nonzero[classified_and_nonzero['PersStatus']=="PersHigh"]
pers_low = classified_and_nonzero[classified_and_nonzero['PersStatus']=="PersLow"]

#count number of persnorm, pershigh and perslow datapoints
count_norm = pers_norm.Glucose_Mean.value_counts()
count_high = pers_high.Glucose_Mean.value_counts()
count_low = pers_low.Glucose_Mean.value_counts()

#figure showing count of interstitial glucose values divided into different pers classes
plt.figure(figsize=(12, 6))
plt.bar(count_norm.index, count_norm, color="slategrey", alpha=0.5, label="PersNorm")
plt.bar(count_high.index, count_high, color="coral", alpha=0.5, label="PersHigh")
plt.bar(count_low.index, count_low, color="teal", alpha=0.5, label="PersLow")
plt.xlabel("Interstitial Glucose Values [mg/dL]", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend(fontsize=14)
file_save = rf'D:\Vittoria\Code\data\plots\pers_plots\pers_plots'
#plt.show()
plt.savefig(file_save)

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
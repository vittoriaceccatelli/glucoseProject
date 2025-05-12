import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def loop(time, time_numeric, data, ax, title):
    #initiate variables for plots for dexcom and wearable
    start_datetime = time.min() 
    segments = []
    colors = []
    start = None
    previous_state = None

    for i in range(len(time)):
        #decide if current state is True (values)
        current_state = data.iloc[i] > 0 

        #if looking at the start of a sequence with values
        if start is None: 
            start = time_numeric[i] #save start position
            previous_state = current_state #previous state equal to current state (True: values, False: Zeros/NaNs)

        #if change from values to zeros/NaNs or vice versa
        elif current_state != previous_state:
            segments.append((start, time_numeric[i] - start)) #append to list segments tuple of start, end of sequence of values orr NaNs/zeros
            colors.append("red" if previous_state else "black") #append to list colors red if values or black if Nans/zeros
            start = time_numeric[i]
            previous_state = current_state

    #append to segments and colors lists last segment
    segments.append((start, time_numeric[-1] - start))
    colors.append("red" if previous_state else "black")

    ax.broken_barh(segments, (0, 1), facecolors=colors)
    ax.set_ylabel(f"{title}")
    ax.set_yticks([])
    tick_positions = np.linspace(time_numeric.min(), time_numeric.max(), num=10)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels((start_datetime + pd.to_timedelta(tick_positions, unit="s")).strftime("%Y-%m-%d"), rotation=45)

    return segments, colors


def overlap_plots(part, all):
    #select data from a single participant 
    part_data = all[all["Participant_ID"] == part]

    time = pd.to_datetime(part_data['Time'])

    #only select columns i want to look at 
    temp = part_data['TEMP_Mean']
    eda = part_data['EDA_Mean']
    hr = part_data['HR_Mean']
    acc = part_data['ACC_Mean']
    hrv = part_data['HRV_Mean']
    dexcom = part_data['Glucose_Mean']

    #create time list 
    time_numeric = (time - time.min()).dt.total_seconds().values

    #calculate percentage of zeros and NaNs
    temp_zero_na_percentage = ((temp.isna() | (temp == 0)).sum() / len(temp)) * 100
    eda_zero_na_percentage = ((eda.isna() | (eda == 0)).sum() / len(eda)) * 100
    hr_zero_na_percentage = ((hr.isna() | (hr == 0)).sum() / len(hr)) * 100
    acc_zero_na_percentage = ((acc.isna() | (acc == 0)).sum() / len(acc)) * 100
    hrv_zero_na_percentage = ((hrv.isna() | (hrv == 0)).sum() / len(hrv)) * 100
    dexcom_zero_na_percentage = ((dexcom.isna() | (dexcom == 0)).sum() / len(dexcom)) * 100

    #plot two bars showing overlap between dexcom and wearable device data over time
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(12, 6), sharex=True)

    fig.suptitle(f"Participant {part}")

    loop(time, time_numeric, temp, ax1, "TEMP")
    loop(time, time_numeric, eda, ax2, "EDA")
    loop(time, time_numeric, hr, ax3, "HR")
    loop(time, time_numeric, acc, ax4, "ACC")
    loop(time, time_numeric, hrv, ax5, "HRV")
    loop(time, time_numeric, dexcom, ax6, "DEXCOM")

    ax6.set_xlabel("Time")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    size = part_data.shape[0]
    #plt.show()
    plt.savefig(rf"D:\Vittoria\Code\data\plots\overlap\participant_{part}_post_discarding")
    #plt.savefig(rf"D:\Vittoria\Code\data\plots\overlap\participant_{part}")
    return temp_zero_na_percentage, eda_zero_na_percentage, hr_zero_na_percentage, acc_zero_na_percentage, hrv_zero_na_percentage, dexcom_zero_na_percentage, size

#import data of dexcom and features
wearable = pd.read_csv(r"D:\Vittoria\Code\data\all_features.csv")
dexcom = pd.read_csv(r"D:\Vittoria\Code\data\dexcom\dexcom_data_5mins.csv")
dexcom.rename(columns={"Timestamp": "Time"}, inplace=True)
wearable.rename(columns={"Time": "Hours_from_midnight"}, inplace=True)
wearable.rename(columns={"Time.1": "Minutes_from_midnight"}, inplace=True)
wearable.rename(columns={"Unnamed: 0": "Time"}, inplace=True)
all = pd.merge(wearable, dexcom, on=["Time", "Participant_ID"], how="outer")
all = all.sort_values(by=["Participant_ID", "Time"])

temp_zero = []
eda_zero = []
hr_zero = []
acc_zero = []
hrv_zero = []
dexcom_zero = []
discarded_data = []

#loop over all participants
for part in range(1,17):
    temp_zero_na_percentage, eda_zero_na_percentage, hr_zero_na_percentage, acc_zero_na_percentage, hrv_zero_na_percentage, dexcom_zero_na_percentage, size_before = overlap_plots(part, all)
    temp_zero.append(temp_zero_na_percentage) #append NaN percentage to wearable list
    eda_zero.append(eda_zero_na_percentage)
    hr_zero.append(hr_zero_na_percentage)
    acc_zero.append(acc_zero_na_percentage)
    hrv_zero.append(hrv_zero_na_percentage)
    dexcom_zero.append(dexcom_zero_na_percentage) #append NaN percentage to dexcom list
    discarded_data.append(size_before)
    

#create a Pandas Series from the NaN percentage lists 
temp_zero = pd.Series([float(x) for x in temp_zero])
eda_zero = pd.Series([float(x) for x in eda_zero])
hr_zero = pd.Series([float(x) for x in hr_zero])
acc_zero = pd.Series([float(x) for x in acc_zero])
hrv_zero = pd.Series([float(x) for x in hrv_zero])
dexcom_zero = pd.Series([float(x) for x in dexcom_zero])
discarded_data = pd.Series(discarded_data)

discarded_before = pd.DataFrame({
    'temp_zero': temp_zero,
    'eda_zero': eda_zero,
    'hr_zero': hr_zero,
    'acc_zero': acc_zero,
    'hrv_zero': hrv_zero,
    'dexcom_zero': dexcom_zero,
    'discarded_data': discarded_data
})

#import data with discarded datapoints
all = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features.csv")

temp_zero = []
eda_zero = []
hr_zero = []
acc_zero = []
hrv_zero = []
dexcom_zero = []
discarded_data = []

#loop over all participants
for part in range(1,17):
    temp_zero_na_percentage, eda_zero_na_percentage, hr_zero_na_percentage, acc_zero_na_percentage, hrv_zero_na_percentage, dexcom_zero_na_percentage, size_after = overlap_plots(part, all)
    temp_zero.append(temp_zero_na_percentage) #append NaN percentage to wearable list
    eda_zero.append(eda_zero_na_percentage)
    hr_zero.append(hr_zero_na_percentage)
    acc_zero.append(acc_zero_na_percentage)
    hrv_zero.append(hrv_zero_na_percentage)
    dexcom_zero.append(dexcom_zero_na_percentage) #append NaN percentage to dexcom list

    discarded_data.append(size_after)

#create a Pandas Series from the NaN percentage lists 
#temp_zero = pd.Series([float(x) for x in temp_zero])
#eda_zero = pd.Series([float(x) for x in eda_zero])
#hr_zero = pd.Series([float(x) for x in hr_zero])
#acc_zero = pd.Series([float(x) for x in acc_zero])
#hrv_zero = pd.Series([float(x) for x in hrv_zero])
#dexcom_zero = pd.Series([float(x) for x in dexcom_zero])
discarded_data = pd.Series(discarded_data)

discarded_afer = pd.DataFrame({
    #'temp_zero': temp_zero,
    #'eda_zero': eda_zero,
    #'hr_zero': hr_zero,
    #'acc_zero': acc_zero,
    #'hrv_zero': hrv_zero,
    #'dexcom_zero': dexcom_zero,
    'discarded_data': discarded_data
})

discarded = pd.merge(discarded_before, discarded_afer, left_on=discarded_before.index, right_on=discarded_afer.index)
discarded['discarded_percent'] = (1-discarded['discarded_data_y']/discarded['discarded_data_x'])*100
discarded = discarded.drop(columns=['key_0', 'discarded_data_x'])
discarded.rename(columns={"discarded_data_y": "datapoints_remaining"}, inplace=True)
discarded['percent_of_dataset'] = discarded['datapoints_remaining']/all.shape[0]*100
discarded.to_csv(r'data\numbers_discarded.csv', index=False)
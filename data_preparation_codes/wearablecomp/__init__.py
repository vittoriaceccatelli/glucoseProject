import numpy as np
import pandas as pd
import datetime as datetime
from scipy.signal import find_peaks, peak_prominences
from scipy.interpolate import interp1d
from scipy import signal
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

'''
    Feature Engineering of Wearable Sensors:
     
    Metrics computed:
        Mean Heart Rate Variability
        Median Heart Rate Variability
        Maximum Heart Rate Variability
        Minimum Heart Rate Variability
        SDNN (HRV)
        RMSSD (HRV)
        NNx (HRV)
        pNNx (HRV)
        HRV Frequency Domain Metrics:
            PowerVLF
            PowerLF
            PowerHF
            PowerTotal
            LF/HF
            PeakVLF
            PeakLF
            PeakHF
            FractionLF
            FractionHF
        EDA Peaks
        Activity Bouts
        Interday Summary: 
            Interday Mean
            Interday Median
            Interday Maximum 
            Interday Minimum 
            Interday Quartile 1
            Interday Quartile 3
        Interday Standard Deviation 
        Interday Coefficient of Variation 
        Intraday Standard Deviation (mean, median, standard deviation)
        Intraday Coefficient of Variation (mean, median, standard deviation)
        Intraday Mean (mean, median, standard deviation)
        Daily Mean
        Intraday Summary:
            Intraday Mean
            Intraday Median
            Intraday Minimum
            Intraday Maximum
            Intraday Quartile 1
            Intraday Quartile 3
        TIR (Time in Range of default 1 SD)
        TOR (Time outside Range of default 1 SD)
        POR (Percent outside Range of default 1 SD)
        MASE (Mean Amplitude of Sensor Excursions, default 1 SD)
        Hours from Midnight (circadian rhythm feature)
        Minutes from Midnight (ciracadian rhythm feature)

        
    '''


def e4import(filepath, sensortype, format_id, Starttime='NaN', Endtime='NaN', window='5min'): #window is in seconds
    """
        brings in an empatica compiled file **this is not raw empatica data**
        Args:
            filepath (String): path to file
            sensortype (Sting): Options: 'EDA', 'HR', 'ACC', 'TEMP', 'BVP'
            Starttime (String): (optional, default arg = 'NaN') format '%Y-%m-%d %H:%M:%S.%f', if you want to only look at data after a specific time
            Endtime (String): (optional, default arg = 'NaN') format '%Y-%m-%d %H:%M:%S.%f', if you want to only look at data before a specific time
            window (String): default '5min'; this is the window your data will be resampled on.
        Returns:
            (pd.DataFrame): dataframe of data with Time, Mean, Std columns
    """
    
    if sensortype == 'ACC':
        data = pd.read_csv(filepath, names=["Time", "x", "y", "z"], dtype=str)
        data.columns = data.columns.str.strip()

        # Drop first row if it’s not a timestamp
        if not data["Time"].iloc[0].startswith(("202")):
            data = data.iloc[1:].reset_index(drop=True) 

        #drop if time column contains na
        data["Time"] = pd.to_datetime(data["Time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        data = data.dropna(subset=["Time"])

        data[['x', 'y', 'z']] = data[['x', 'y', 'z']].apply(pd.to_numeric, errors='coerce')
        data['Var'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
        data = data.drop(columns=['x', 'y', 'z'])   
    else:
        data = pd.read_csv(filepath, names=['Time', 'Var'], dtype=str)
        data.columns = data.columns.str.strip()

        # Drop first row if it’s not a timestamp
        if not data["Time"].iloc[0].startswith(("202")): 
            data = data.iloc[1:].reset_index(drop=True) 

        #participant 001 and sensors hr and acc have different datetime formats
        if str(format_id) == '001' and sensortype in ['ACC', 'HR']:
            data["Time"] = pd.to_datetime(data["Time"], format="%m/%d/%y %H:%M", errors="coerce")
        elif sensortype == 'TEMP' or sensortype == 'EDA':
            data["Time"] = pd.to_datetime(data["Time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        else:
            data["Time"] = pd.to_datetime(data["Time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        #drop if time column contains na
        data = data.dropna(subset=["Time"])

    data['Time'] =  pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S.%f')
    
    if Starttime != 'NaN':
        VarData = data.loc[data.loc[:, 'Time'] >= Starttime, :]
        if Endtime != 'NaN':
            VarData = VarData.loc[VarData.loc[:, 'Time'] <= Endtime, :]
    else:
        VarData = data
    
    Data = pd.DataFrame()
    numeric_cols = VarData.select_dtypes(include=['object']).columns 

    VarData[numeric_cols] = VarData[numeric_cols].apply(pd.to_numeric, errors='coerce')

    VarData = VarData.dropna()
    VarData['Time'] = pd.to_datetime(VarData['Time'], errors='coerce')
    VarData.set_index('Time', inplace=True)

    #if sensor is ACC: calculate features on 5 min rolling basis and on 2hr rolling basis 
    if sensortype == 'ACC':
        Data[[(sensortype + '_Mean')]] = VarData.resample(window).mean()
        Data[[(sensortype + '_Std')]] = VarData.resample(window).std()
        Data[[(sensortype + '_Min')]] = VarData.resample(window).min()
        Data[[(sensortype + '_Max')]] = VarData.resample(window).max()
        Data[[(sensortype + '_q1G')]] = VarData.resample(window).quantile(0.25)
        Data[[(sensortype + '_q3G')]] = VarData.resample(window).quantile(0.75)
        Data[[(sensortype + '_skew')]] = VarData.resample(window).apply(lambda x: (np.sum((x - np.mean(x)) ** 3) / ((len(x) - 1) * (np.std(x, ddof=1) ** 3)))if len(x) > 1 and np.std(x, ddof=1) != 0 else 0)

        Data_2hr = pd.DataFrame()
        Data_2hr[[(sensortype + '_Mean2hrs')]] = Data[[(sensortype + '_Mean')]].rolling('2h').mean()
        Data_2hr[[(sensortype + '_Max2hrs')]] = Data[[(sensortype + '_Max')]].rolling('2h').max()
        Data = Data.reset_index().rename(columns={'index': 'Time'})
        return Data, Data_2hr
    #if sensor is not ACC: calculate features only on 5 min rolling basis 
    else:
        Data[[(sensortype + '_Mean')]] = VarData.resample(window).mean()
        Data[[(sensortype + '_Std')]] = VarData.resample(window).std()
        Data[[(sensortype + '_Min')]] = VarData.resample(window).min()
        Data[[(sensortype + '_Max')]] = VarData.resample(window).max()
        Data[[(sensortype + '_q1G')]] = VarData.resample(window).quantile(0.25)
        Data[[(sensortype + '_q3G')]] = VarData.resample(window).quantile(0.75)
        Data[[(sensortype + '_skew')]] = VarData.resample(window).apply(lambda x: (np.sum((x - np.mean(x)) ** 3) / ((len(x) - 1) * (np.std(x, ddof=1) ** 3)))if len(x) > 1 and np.std(x, ddof=1) != 0 else 0)
        Data = Data.reset_index().rename(columns={'index': 'Time'})
        return Data

def HRV(IBI, ibimultiplier = 1000):
    """
        computes Heart Rate Variability metrics
        Args:
            time (pandas.DataFrame column or pandas series): time column
            IBI (pandas.DataFrame column or pandas series): column with inter beat intervals
            ibimultiplier (IntegerType): defualt = 1000; transforms IBI to milliseconds. If data is already in ms, set as 1
        Returns:
            maxHRV (FloatType): maximum HRV
            minHRV (FloatType): minimum HRV
            meanHRV (FloatType): mean HRV
            medianHRV(FloatType): median HRV
    """

    ibi = IBI*ibimultiplier
    maxHRV = round(max(ibi) * 10) / 10
    minHRV = round(min(ibi) * 10) / 10
    meanHRV = round(np.mean(ibi) * 10) / 10
    medianHRV = round(np.median(ibi) * 10) / 10
    
    return maxHRV, minHRV, meanHRV, medianHRV


def SDNN(IBI, ibimultiplier=1000):
    """
        computes Heart Rate Variability metric SDNN
        Args:
            time (pandas.DataFrame column or pandas series): time column
            IBI (pandas.DataFrame column or pandas series): column with inter beat intervals
            ibimultiplier (IntegerType): defualt = 1000; transforms IBI to milliseconds. If data is already in ms, set as 1
        Returns:
            SDNN (FloatType): standard deviation of NN intervals 
    """
    ibi = IBI*ibimultiplier
    SDNN = round(np.sqrt(np.var(ibi, ddof=1)) * 10) / 10 
    
    return SDNN


def RMSSD(IBI, ibimultiplier=1000):
    """
        computes Heart Rate Variability metric RMSSD
        Args:
            time (pandas.DataFrame column or pandas series): time column
            IBI (pandas.DataFrame column or pandas series): column with inter beat intervals
            ibimultiplier (IntegerType): defualt = 1000; transforms IBI to milliseconds. If data is already in ms, set as 1
        Returns:
            RMSSD (FloatType): root mean square of successive differences
            
    """
    ibi = IBI*ibimultiplier
    
    differences = abs(np.diff(ibi))
    rmssd = np.sqrt(np.sum(np.square(differences)) / len(differences))
    
    return round(rmssd * 10) / 10


def NNx(IBI, ibimultiplier=1000, x=50):
    """
        computes Heart Rate Variability metrics NNx and pNNx
        Args:
            time (pandas.DataFrame column or pandas series): time column
            IBI (pandas.DataFrame column or pandas series): column with inter beat intervals
            ibimultiplier (IntegerType): defualt = 1000; transforms IBI to milliseconds. If data is already in ms, set as 1
            x (IntegerType): default = 50; set the number of times successive heartbeat intervals exceed 'x' ms
        Returns:
            NNx (FloatType): the number of times successive heartbeat intervals exceed x ms
            pNNx (FloatType): the proportion of NNx divided by the total number of NN (R-R) intervals. 
    """
    ibi = IBI*ibimultiplier
    
    differences = abs(np.diff(ibi))
    n = np.sum(differences > x)
    p = (n / len(differences)) * 100
    
    return (round(n * 10) / 10), (round(p * 10) / 10)

def PeaksEDA(eda, time):
    """
        calculates peaks in the EDA signal
        Args:
            eda (pandas.DataFrame column or pandas series): eda column
            time (pandas.DataFrame column or pandas series): time column
        Returns:
            countpeaks (IntegerType): the number of peaks total 
            peakdf (pandas.DataFrame): a pandas dataframe with time and peaks to easily integrate with your data workflow
    """  
    
    time = pd.to_datetime(time, errors="coerce")
    EDAy = np.array(eda.to_numpy(), dtype=float)

    peaks, _ = find_peaks(EDAy, height=0, distance=4, prominence=0.3)

    peakdf = pd.DataFrame({"Time": time.iloc[peaks], "Peak": 1})
    
    peakdf.set_index("Time", inplace=True)

    #resample every 5min
    peak_summary_5min = peakdf.resample("5T").sum()
    peak_summary_5min.rename(columns={"Peak": "Peak_eda_5min_sum"}, inplace=True)

    #rolling 2hr overlapping window, features: mean and sum
    peak_summary_5min["Peak_eda_2hr_sum"] = peak_summary_5min["Peak_eda_5min_sum"].rolling(window="2h").sum()
    peak_summary_5min["Peak_eda_2hr_mean"] = peak_summary_5min["Peak_eda_2hr_sum"] / 24

    return peak_summary_5min


def exercisepts(acc, hr, time): #acc and hr must be same length, acc must be magnitude
    """
        calculates activity bouts using accelerometry and heart rate
        Args:
            acc (pandas.DataFrame column or pandas series): accelerometry column
            hr (pandas.DataFrame column or pandas series): heart rate column
            time (pandas.DataFrame column or pandas series): time column
        Returns:
            countbouts (IntegerType): the number of acitvity bouts total
            returndf (pandas.DataFrame): a pandas dataframe with time and activity bouts (designated as a '1') to easily integrate with your data workflow
    """  
    
    exercisepoints = []
    for z in range(len(acc)):
        if acc.loc[z, "ACC_Mean"] > np.mean(acc.loc[0:z, "ACC_Mean"]):
            if hr.loc[z, "HR_Mean"] > np.mean(hr.loc[0:z, "HR_Mean"]):
                exercisepoints.append(1)
            else:
                exercisepoints.append(0)
        else:
            exercisepoints.append(0)
            
    returndf = pd.DataFrame()
    returndf['Time'] = time
    exercisepoints = exercisepoints[:len(returndf)]
    returndf['Activity Bouts'] = exercisepoints
    
    return returndf


def interdaycv(column):
    """
        computes the interday coefficient of variation on pandas dataframe Sensor column
        Args:
            column (pandas.DataFrame column or pandas series): column that you want to calculate over
        Returns:
            cvx (IntegerType): interday coefficient of variation 
    """
    cvx = (np.std(column) / (np.nanmean(column)))*100
    return cvx


def interdaysd(column):
    """
        computes the interday standard deviation of pandas dataframe Sensor column
        Args:
            column (pandas.DataFrame column or pandas series): column that you want to calculate over
        Returns:
            interdaysd (IntegerType): interday standard deviation 
    """
    interdaysd = np.std(column)
    return interdaysd

def wake_time(acc, hr):
    #merge acc and hr data on time
    hr["Time"] = pd.to_datetime(hr["Time"], errors="coerce")
    acc["Time"] = pd.to_datetime(acc["Time"], errors="coerce")
    data = pd.merge(hr, acc, on="Time", how="outer")
    data["Time"] = pd.to_datetime(data["Time"], errors="coerce")
    data.set_index("Time", inplace=True)

    #Extract date and calculate daily averages, ensure data is sampled at 5min frequency
    data = data.asfreq("5min")
    data["Date"] = data.index.date  
    daily_avg = data.groupby("Date")[["HR_Mean", "HR_Std", "ACC_Mean", "ACC_Std"]].transform("mean")

    #calculate daily metrics
    daily_metrics = data.groupby("Date").agg(
        HR_CV=('HR_Mean', interdaycv),
        HR_SD=('HR_Std', interdaysd),
        ACC_CV=('ACC_Mean', interdaycv),
        ACC_SD=('ACC_Std', interdaysd),
    )
    data = data.merge(daily_metrics, left_on="Date", right_index=True, how="left")

    #calculate points as described in paper (less than historical average)
    data["Points"] = (
        (data["HR_Mean"] < daily_avg["HR_Mean"]).astype(int) +
        (data["HR_Std"] < daily_avg["HR_Std"]).astype(int) +
        (data["ACC_Mean"] < daily_avg["ACC_Mean"]).astype(int) +
        (data["ACC_Std"] < daily_avg["ACC_Std"]).astype(int)
    )

    #Determine sleep state
    data["Sleep_State"] = (data["Points"] >= 2).astype(int)
    data["Sleep_State"] = data["Sleep_State"].fillna(0)

    # Smooth sleep state using a 3-hour rolling window
    data["Sleep_State_Smoothed"] = data["Sleep_State"].rolling("3h", min_periods=1).mean()

    # Calculate slope
    data["Slope"] = data["Sleep_State_Smoothed"].diff()

    # Identify wake time candidates
    wake_candidates = data[
        (data["Slope"] > 0) 
        & (data["Sleep_State_Smoothed"].rolling("25min", min_periods=1, closed="right").mean() > 0.5) 
        & (data["Sleep_State_Smoothed"].rolling("75min", min_periods=1, closed="right").mean() > 0.5) 
    ]
    data["Wake_Time"] = np.nan
    data.loc[wake_candidates.index, "Wake_Time"] = 1
    data["Wake_Time"] = data["Wake_Time"].fillna(0)
    
    data.drop(columns=[
        "Sleep_State", "Sleep_State_Smoothed", "Slope", "Points", 'HR_Mean', 
        'HR_Std', 'HR_Min', 'HR_Max', 'HR_q1G', 'HR_q3G', 'HR_skew', 'ACC_Mean', 
        'ACC_Std', 'ACC_Min', 'ACC_Max', 'ACC_q1G', 'Date', 'ACC_q3G','ACC_skew', 
        'HR_CV', 'HR_SD', 'ACC_CV', 'ACC_SD'], inplace=True)
    return data

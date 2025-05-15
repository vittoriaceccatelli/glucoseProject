import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_gender(data, what):
    grouped = data.groupby(['Gender', 'PersStatus']).size().unstack(fill_value=0)

    # Improved color palette
    colors = ['#4E79A7', '#F28E2B', '#76B7B2']

    fig, ax = plt.subplots(figsize=(10, 6))
    v = []
    bottom = [0] * len(grouped.index)

    for i, status in enumerate(grouped.columns):
        values = grouped[status].values
        bars = ax.bar(grouped.index, values, bottom=bottom, label=status, color=colors[i])
        v.append([a.tolist() for a in values])

        for bar, count in zip(bars, values):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_y() + bar.get_height()/2,
                    str(count), ha='center', va='center', fontsize=7, color='black')

        bottom = [b + v for b, v in zip(bottom, values)]

    # Aesthetic enhancements
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("PersStatus Distribution by Gender", fontsize=14)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11, frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', length=0)

    plt.tight_layout()
    #plt.show()
    plt.savefig(rf"D:\Vittoria\Code\data\plots\gender_issue\gender_plot_{what}")
    return v


data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features.csv")
demo = pd.read_csv(r"D:\Vittoria\Code\data\labeled_demographics.csv")
data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")
counts = plot_gender(data, "non_balanced")
all_females = counts[0][0] + counts[1][0] + counts[2][0]
all_males = counts[0][1] + counts[1][1] + counts[2][1]

percentage_male_female = all_males/all_females*100

percentage_norms_in_female = counts[2][0]/all_females*100
percentage_high_in_female = counts[0][0]/all_females*100
percentage_low_in_female = counts[1][0]/all_females*100

percentage_norms_in_male = counts[2][1]/all_males*100
percentage_high_in_male = counts[0][1]/all_males*100
percentage_low_in_male = counts[1][1]/all_males*100

print(percentage_male_female)
print(percentage_norms_in_female, percentage_high_in_female, percentage_low_in_female)
print(percentage_norms_in_male, percentage_high_in_male, percentage_low_in_male)

data = pd.read_csv(r"D:\Vittoria\Code\data\balanced_dataset_random_Labeled.csv")
demo = pd.read_csv(r"D:\Vittoria\Code\data\labeled_demographics.csv")
data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")
counts = plot_gender(data, "random")

data = pd.read_csv(r"D:\Vittoria\Code\data\oversampled_Labeled.csv")
demo = pd.read_csv(r"D:\Vittoria\Code\data\labeled_demographics.csv")
data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")
counts = plot_gender(data, "oversample")


'''data = pd.read_csv(r"D:\Vittoria\Code\data\gender_balanced_dataset_random_Labeled.csv")
counts = plot_gender(data, "balanced")
all_females = counts[0][0] + counts[1][0] + counts[2][0]
all_males = counts[0][1] + counts[1][1] + counts[2][1]
percentage_male_female = all_males/all_females*100

percentage_norms_in_female = counts[2][0]/all_females*100
percentage_high_in_female = counts[0][0]/all_females*100
percentage_low_in_female = counts[1][0]/all_females*100

percentage_norms_in_male = counts[2][1]/all_males*100
percentage_high_in_male = counts[0][1]/all_males*100
percentage_low_in_male = counts[1][1]/all_males*100

print(percentage_male_female)
print(percentage_norms_in_female, percentage_high_in_female, percentage_low_in_female)
print(percentage_norms_in_male, percentage_high_in_male, percentage_low_in_male)

data = pd.read_csv(r"D:\Vittoria\Code\data\gender_balanced_dataset_random_NoCorrelation.csv")
counts = plot_gender(data, "balanced")
all_females = counts[0][0] + counts[1][0] + counts[2][0]
all_males = counts[0][1] + counts[1][1] + counts[2][1]
percentage_male_female = all_males/all_females*100

percentage_norms_in_female = counts[2][0]/all_females*100
percentage_high_in_female = counts[0][0]/all_females*100
percentage_low_in_female = counts[1][0]/all_females*100

percentage_norms_in_male = counts[2][1]/all_males*100
percentage_high_in_male = counts[0][1]/all_males*100
percentage_low_in_male = counts[1][1]/all_males*100

print(percentage_male_female)
print(percentage_norms_in_female, percentage_high_in_female, percentage_low_in_female)
print(percentage_norms_in_male, percentage_high_in_male, percentage_low_in_male)'''


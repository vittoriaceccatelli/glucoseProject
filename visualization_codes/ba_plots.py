import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from collections import Counter

# Function to create a Bland-Altman plot
def bland_altman_plot(y_true, y_pred, title="Bland-Altman Plot - Population TabPFN model - Without Food Log Features", units="mg/dL"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diffs = y_true - y_pred  # Differences between true and predicted values
    md = np.mean(diffs)      # Mean difference (bias)
    sd = np.std(diffs, ddof=1)  # Sample standard deviation of differences
    loa_upper = md + 1.96 * sd  # Upper limit of agreement
    loa_lower = md - 1.96 * sd  # Lower limit of agreement

    # Plotting the differences
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_true, y=diffs, s=25, alpha=0.6)

    # Add lines for mean and limits of agreement
    plt.axhline(md, color='blue', linestyle='-', label=f"Mean Diff = {md:.2f} {units}")
    plt.axhline(loa_upper, color='orange', linestyle='--', label=f"+1.96 SD = {loa_upper:.2f}")
    plt.axhline(loa_lower, color='orange', linestyle='--', label=f"-1.96 SD = {loa_lower:.2f}")

    plt.xlabel(f'Reference Glucose Values ({units})')
    plt.ylabel(f'Difference (Reference - Predicted) ({units})')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# Function to classify prediction-reference pairs into Clarke zones
def get_zone(ref_values, pred_values):
    if (ref_values <= 70 and pred_values <= 70) or (pred_values <= 1.2*ref_values and pred_values >= 0.8*ref_values):
        return 'A'
    elif (ref_values >= 180 and pred_values <= 70) or (ref_values <= 70 and pred_values >= 180):
        return 'E'
    elif ((ref_values >= 70 and ref_values <= 290) and pred_values >= ref_values + 110) or ((ref_values >= 130 and ref_values <= 180) and (pred_values <= (7/5)*ref_values - 182)):
        return 'C'
    elif (ref_values >= 240 and (pred_values >= 70 and pred_values <= 180)) or (ref_values <= 175/3 and pred_values <= 180 and pred_values >= 70) or ((ref_values >= 175/3 and ref_values <= 70) and pred_values >= (6/5)*ref_values):
        return 'D'
    else:
        return 'B'

# Function to plot Clarke Error Grid with zone breakdown
def clarke_error_grid(ref_values, pred_values):
    zones = [get_zone(r, p) for r, p in zip(ref_values, pred_values)]  # Assign zones to each point
    zone_counts = Counter(zones)  # Count of each zone
    total = len(zones)
    zone_percentages = {zone: 100 * count / total for zone, count in zone_counts.items()}  # Percentages

    # Print zone distribution
    for zone in sorted(['A', 'B', 'C', 'D', 'E']):
        pct = zone_percentages.get(zone, 0)
        print(f"Zone {zone}: {pct:.2f}%")

    # Define color for each zone
    color_map = {
        'A': 'green',
        'B': 'yellow',
        'C': 'orange',
        'D': 'red',
        'E': 'purple'
    }
    colors = [color_map[z] for z in zones]

    # Scatter plot of predictions with zone-based color
    plt.scatter(ref_values, pred_values, marker='o', c=colors, s=8)
    plt.title("Clarke Error Grid - Population TabPFN model - Without Food Log Features")
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("Prediction Concentration (mg/dl)")
    plt.xticks(np.arange(0, 401, 50))
    plt.yticks(np.arange(0, 401, 50))
    plt.gca().set_facecolor('white')
    plt.xlim([0, 400])
    plt.ylim([0, 400])
    plt.gca().set_aspect('equal')

    # Plot zone boundaries
    plt.plot([0, 400], [0, 400], ':', c='black')
    plt.plot([0, 175/3], [70, 70], '-', c='black')
    plt.plot([175/3, 400/1.2], [70, 400], '-', c='black')
    plt.plot([70, 70], [84, 400], '-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    plt.plot([70, 290], [180, 400], '-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black') 
    plt.plot([70, 400], [56, 320], '-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 400], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180], '-', c='black')
    plt.plot([240, 400], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')

    # Add zone labels
    plt.text(30, 15, "A", fontsize=15)
    plt.text(370, 260, "B", fontsize=15)
    plt.text(280, 370, "B", fontsize=15)
    plt.text(160, 370, "C", fontsize=15)
    plt.text(160, 15, "C", fontsize=15)
    plt.text(30, 140, "D", fontsize=15)
    plt.text(370, 120, "D", fontsize=15)
    plt.text(30, 370, "E", fontsize=15)
    plt.text(370, 15, "E", fontsize=15)

# Filepaths for different prediction models
person_nofood = rf"D:\Vittoria\Code\ba-plots\predictions\personalised_nofood.csv"
person_food = rf"D:\Vittoria\Code\ba-plots\predictions\personalised_yesfood.csv"
population_nofood = rf"D:\Vittoria\Code\ba-plots\predictions\population_nofood.csv"
population_food = rf"D:\Vittoria\Code\ba-plots\predictions\population_yesfood.csv"
tabpfn_person_nofood = rf"D:\Vittoria\Code\ba-plots\predictions\tabpfn_personalised_nofood.csv"
tabpfn_person_food = rf"D:\Vittoria\Code\ba-plots\predictions\tabpfn_personalised_yesfood.csv"
tabpfn_population_nofood = rf"D:\Vittoria\Code\ba-plots\predictions\tabpfn_population_nofood.csv"
tabpfn_population_food = rf"D:\Vittoria\Code\ba-plots\predictions\tabpfn_population_yesfood.csv"

# Load prediction CSV for evaluation
data = pd.read_csv(tabpfn_population_food).drop(columns=["Unnamed: 0"])

# Generate Bland-Altman plot
bland_altman_plot(data["True"], data["Prediction"])
plt.close()

# Generate Clarke Error Grid
clarke_error_grid(data["True"], data["Prediction"])
plt.close()
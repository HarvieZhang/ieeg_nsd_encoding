import pickle
import matplotlib.pyplot as plt
import numpy as np

import re

# Define the dictionary to store the extracted data
results = {}

# Open the text file and read its contents
with open("output_30", "r") as file:
    for line in file:
        # Use regular expressions to extract coordinates and validation correlation
        coord_match = re.search(r"Coordinates: \((\d+),(\d+)\)", line)
        corr_match = re.search(r"Best Validation Correlation: (-?\d+\.\d+)", line)

        # If both matches are found, store the data in the dictionary
        if coord_match and corr_match:
            x, y = int(coord_match.group(1)), int(coord_match.group(2))
            correlation = float(corr_match.group(1))
            results[(x, y)] = correlation




# Extract data from the results dictionary
times = [coord[1] * 2 for coord in results.keys()]  # y from data as time in ms
correlations = list(results.values())

# Sort data based on times
sorted_indices = np.argsort(times)
times = np.array(times)[sorted_indices]
correlations = np.array(correlations)[sorted_indices]

# Create a 2D array with correlations
heatmap_data = correlations[np.newaxis, :]

# Plot histogram
plt.figure(figsize=(10, 5))
plt.bar(times, correlations, width=1.6, color='blue', alpha=0.7)


# Plot histogram of correlations
#plt.figure(figsize=(10, 5))
#plt.hist(correlations, bins=30, color='blue', alpha=0.7, edgecolor='black')
#plt.xlabel('Correlation')
#plt.ylabel('Number of Points')
#plt.title('Distribution of Prediction Correlations')
#plt.tight_layout()

# Mark "present image" at x=960
plt.axvline(x=480, color='red', linestyle='--')
plt.annotate('present image', xy=(480, 1), xytext=(960, 1.05),
             arrowprops=dict(facecolor='red', shrink=0.05),
             horizontalalignment='center', verticalalignment='top',
             color='red')

plt.xlabel('Time')
plt.ylabel('Correlation')
plt.title('Prediction Correlation in 81.6404Hz')
plt.tight_layout()

# Create the color bar visualization
#plt.figure(figsize=(10, 2))
#plt.imshow(heatmap_data, cmap='viridis', aspect='auto', extent=[times.min(), times.max(), 0, 1])
#plt.colorbar(label='Correlation')
#plt.xlabel('Time')
#plt.yticks([])
#plt.title('Prediction Correlation in 81.6404Hz')
#plt.tight_layout()

plt.savefig("corr_wavelet_distribution.png", dpi=300, bbox_inches='tight')
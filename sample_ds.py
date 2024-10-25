import numpy as np
import matplotlib.pyplot as plt

# Data
datasets = [
    "Walmart Sales",
    "Admission Predict",
    "Car Details",
    "Customer Purchasing Behaviors",
    "Employee Data",
    "Gold Price",
    "Laptop Price",
    "Student Performance",
    "Book Read",
    "Rounded Hours Student Scores"
]

brutal_nmse = [
    0.0343, 
    0.2239, 
    0.6546, 
    0.00192, 
    0.6201, 
    0.0158, 
    0.3511, 
    0.0158, 
    0.0119, 
    0.2002
]

non_brutal_nmse = [
    0.0362, 
    0.2159, 
    0.7230, 
    0.00220, 
    0.5946, 
    0.0159, 
    0.3501, 
    0.0159, 
    0.0261, 
    0.2406
]

# Calculate performance as 1/NMSE
brutal_performance = [1/nmse for nmse in brutal_nmse]
non_brutal_performance = [1/nmse for nmse in non_brutal_nmse]

# Calculate relative performance (non_brutal / brutal)
relative_performance = [nb / b for nb, b in zip(non_brutal_performance, brutal_performance)]

# Number of variables
num_vars = len(datasets)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Complete the loop for the plot
brutal_baseline = [1] * (num_vars + 1)
relative_performance += relative_performance[:1]
angles += angles[:1]

# Create the radar chart
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], datasets, size=8)

# Plot data
ax.plot(angles, brutal_baseline, color='blue', linewidth=2, label='Brutal (Baseline)')
ax.plot(angles, relative_performance, color='red', linewidth=2, label='Non-Brutal (Relative)')
ax.fill(angles, relative_performance, color='red', alpha=0.25)

# Add a title and a legend
plt.title('Relative Performances of A4ML for LLM-Native vs Brutal Preprocessing Across Datasets', size=15, pad=30)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Set y-axis limits
ax.set_ylim(0, 2)  # Adjust as needed
ax.set_yticks([0.5, 1, 1.5])

# Adjust layout to bring labels closer
plt.tight_layout()
plt.show()
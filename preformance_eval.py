import pandas as pd
import matplotlib.pyplot as plt

d_accuracy = {
    'BPIC2011':[0.37, 0.95, 0.95],
    'BPIC2012':[0.93, 0.67, 0.67],
    'BPIC2015_1':[0.89, 0.88, 0.89],
    'BPIC2015_2':[0.95, 0.91, 0.92],
    'BPIC2015_3':[0.87, 0.91, 0.91],
    'BPIC2015_4':[0.89, 0.88, 0.88],
    'BPIC2015_5':[0.90, 0.89, 0.89],
    'BPIC2017':[0.86, 0.81, 0.81]
}

d_f1 = pd.DataFrame({
    'BPIC2011':[0.32, 0.76, 0.89],
    'BPIC2012':[0.9, 0.42, 0.51],
    'BPIC2015_1':[0.85, 0.73, 0.62],
    'BPIC2015_2':[0.93, 0.74, 0.75],
    'BPIC2015_3':[0.82, 0.74, 0.75],
    'BPIC2015_4':[0.84, 0.67, 0.64],
    'BPIC2015_5':[0.87, 0.77, 0.77],
    'BPIC2017':[0.81, 0.64, 0.64]
})

# Transpose the DataFrame to have models as columns and datasets as rows
d_f1_transposed = d_f1.T

# Set the colors for each model
colors = ['#888888', '#555555', '#222222']

# Get the model names
models = d_f1_transposed.index.tolist()

# Plot the grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.2  # Width of the bars

for i, col in enumerate(d_f1_transposed.columns):
    x = [j + width*i for j in range(len(models))]
    ax.bar(x, d_f1_transposed[col], width, label=col, color=colors[i])

# Set the x-axis labels to be the model names
ax.set_xticks([j + 0.2 for j in range(len(models))])
ax.set_xticklabels(models)

# Set plot labels and title
plt.xlabel('Dataset')
plt.ylabel('F1 Score')
# plt.title('F1 Scores for the different Datasets and Models')

# Add a legend to differentiate Transformer, Random Forest, and XGBoost
plt.legend(["Transformer", "Random Forest", "XGBoost"])

# Show the plot
plt.tight_layout()

# Show the plot
plt.savefig("f1.png",  dpi=300)
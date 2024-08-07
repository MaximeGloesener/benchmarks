"""
We want to give a score to each model based on the metrics and then rank them.
"""

import pandas as pd
import numpy as np

# Convert the data to a DataFrame
df = pd.read_csv('output_file.csv', header=None)
df.columns = ['Model', 'FPS CPU', 'FPS GPU', 'Nombre de paramètres (M)', 'Taille modèle (MB)', 'Nombre de MACs (M)', 'Max memory used (MB)', 'TOP 1 ACC', 'TOP 5 ACC']

# Normalize the metrics
for col in df.columns[1:]:
    if col in ['FPS CPU', 'FPS GPU', 'TOP 1 ACC', 'TOP 5 ACC']:
        df[col] = df[col] / df[col].max()
    else:
        df[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Define weights
weights = [1, 2.5, 0.5, 0.5, 0.5, 1, 20, 5]

# Calculate the weighted score
df['Score'] = df.iloc[:, 1:].mul(weights).sum(axis=1)

# Sort by score in descending order
df_sorted = df.sort_values('Score', ascending=False)

# Display the results
print(df_sorted[['Model', 'Score']])

# Save the results to a new CSV file
df_sorted[['Model', 'Score']].to_csv('ranking.csv', index=False)

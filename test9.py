import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm

# Sample similarity scores between images (replace this with your actual similarity scores)
similarity_scores = np.array([
    [1.0, 0.9, 0.8, 0.85, 0.87, 0.88, 0.83, 0.91, 0.89, 0.86],
    [0.9, 1.0, 0.88, 0.84, 0.86, 0.89, 0.82, 0.90, 0.87, 0.83],
    [0.8, 0.88, 1.0, 0.87, 0.82, 0.85, 0.81, 0.89, 0.84, 0.80],
    [0.85, 0.84, 0.87, 1.0, 0.92, 0.93, 0.91, 0.88, 0.90, 0.89],
    [0.87, 0.86, 0.82, 0.92, 1.0, 0.94, 0.89, 0.87, 0.93, 0.91],
    [0.88, 0.89, 0.85, 0.93, 0.94, 1.0, 0.92, 0.90, 0.91, 0.95],
    [0.83, 0.82, 0.81, 0.91, 0.89, 0.92, 1.0, 0.86, 0.88, 0.90],
    [0.91, 0.90, 0.89, 0.88, 0.87, 0.90, 0.86, 1.0, 0.94, 0.92],
    [0.89, 0.87, 0.84, 0.90, 0.93, 0.91, 0.88, 0.94, 1.0, 0.95],
    [0.86, 0.83, 0.80, 0.89, 0.91, 0.95, 0.90, 0.92, 0.95, 1.0]
])

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=2, random_state=42)
embedded = tsne.fit_transform(similarity_scores)

# Generate colors based on similarity scores
norm = plt.Normalize(vmin=similarity_scores.min(), vmax=similarity_scores.max())
cmap = cm.get_cmap('viridis')

# Plot the similarity plot
plt.figure(figsize=(8, 6))
for i, (x, y) in enumerate(embedded):
    plt.scatter(x, y, c=[cmap(norm(score))], marker='o', label=f'Image {i+1} - Similarity: {score:.2f}')
plt.title('Image Similarity Plot')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.grid(True)
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label='Similarity Score')
plt.show()

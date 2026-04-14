import matplotlib.pyplot as plt
import seaborn as sns

def visualize_matrix(matrix, title, cmap='viridis'):
  plt.figure(figsize=(10, 8))

  sns.heatmap(matrix,
              annot=True,
              cmap=cmap,
              fmt=".2f",
              vmin=-1, vmax=1,
              square=True)

  plt.title(title)
  plt.show()
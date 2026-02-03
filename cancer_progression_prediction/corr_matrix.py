import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'prepared_data.csv'

df = pd.read_csv(file_path)

correlation_matrix = df.corr()
#print(correlation_matrix)

plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Корреляционная матрица')
plt.tight_layout()
plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = './Online_game.csv'
data = pd.read_csv(file_path)

year_data = data['Year'].dropna()

plt.figure(figsize=(12, 6))
sns.histplot(year_data, kde=True, bins=40)
plt.title('Distribution of Game Release Years')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.show()
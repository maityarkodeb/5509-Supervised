import pandas as pd

file_path = './Online_game.csv'
data = pd.read_csv(file_path)

sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
correlation_matrix = data[sales_columns].corr()

print(correlation_matrix)
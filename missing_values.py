import pandas as pd

file_path = './Online_game.csv'
data = pd.read_csv(file_path)

year_missing = data['Year'].isnull().sum()
publisher_missing = data['Publisher'].isnull().sum()

print("# missing values for year: ", year_missing)
print("# missing values for publisher: ", publisher_missing)


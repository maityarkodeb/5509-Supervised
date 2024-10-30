import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

file_path = './Online_game.csv'
data = pd.read_csv(file_path)

# handle missing values
data['Year'].fillna(data['Year'].median(), inplace=True)
data['Publisher'].fillna(data['Publisher'].mode()[0], inplace=True)

label_encoders = {}
categorical_columns = ['Platform', 'Genre', 'Publisher']
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data.drop(columns=['Global_Sales', 'Name', 'Rank'])
y = data['Global_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))

decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train, y_train)

decision_tree_predictions = decision_tree_model.predict(X_test)
decision_tree_rmse = np.sqrt(mean_squared_error(y_test, decision_tree_predictions))

print("Linear Regression - Root mean squared error: ", linear_rmse)
print("Decision Tree - Root mean squared error: ", decision_tree_rmse)


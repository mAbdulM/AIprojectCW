import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# Load CSV file
data = pd.read_csv("/path", encoding='UTF-8')
df = pd.DataFrame(data)

# Remove dots and dollar signs, and convert to float
data['value'] = data['value'].replace('[^\d]', '', regex=True).astype(float)

data.drop('marking', axis=1, inplace=True)

print(data.shape)

# Encode categorical columns to integers if present
categorical_columns = ['player', 'country', 'club']
for col in categorical_columns:
    if data[col].dtype == 'object':
        data[col] = LabelEncoder().fit_transform(data[col])

# Separate features and target
features = data.drop('value', axis=1)  
target = data['value']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=9)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=55, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = rf_model.predict(X_test)

# Perform k-fold cross-validation (k=5)
num_folds = 5
cv_scores = cross_val_score(rf_model, features, target, cv=num_folds, scoring='neg_mean_squared_error')

# Convert negative MSE to positive values
cv_scores = -cv_scores

# Print the cross-validation scores
print(f"Cross-Validation MSE Scores: {cv_scores}")
print(f"Mean MSE: {cv_scores.mean()}")
print(f"Standard Deviation of MSE: {cv_scores.std()}")

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Actual values in blue
plt.scatter(y_test, y_test, alpha=0.5, color='blue', label='Actual')

# Predicted values in orange
plt.scatter(y_test, predictions, alpha=0.5, color='orange', label='Predicted')

plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()



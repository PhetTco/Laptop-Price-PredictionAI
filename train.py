import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the dataset
laptop_data = pd.read_csv('laptop.csv')

# Clean and preprocess data
laptop_data['Ghz'] = laptop_data['Ghz'].str.extract(r'(\d+\.?\d*)').astype(float)
laptop_data['RAM'] = laptop_data['RAM'].str.extract(r'(\d+)').astype(float)
laptop_data['SSD'] = laptop_data['SSD'].str.extract(r'(\d+)').astype(float)
laptop_data['Display'] = laptop_data['Display'].str.extract(r'(\d+\.?\d*)').astype(float)
laptop_data['Battery_Life'] = laptop_data['Battery_Life'].str.extract(r'(\d+\.?\d*)').astype(float)
laptop_data['HDD'] = laptop_data['HDD'].apply(lambda x: 1 if 'HDD' in x else 0)

# Fill missing values
laptop_data.fillna({
    'Ghz': laptop_data['Ghz'].median(),
    'RAM': laptop_data['RAM'].median(),
    'SSD': laptop_data['SSD'].median(),
    'Display': laptop_data['Display'].median(),
    'Battery_Life': laptop_data['Battery_Life'].median(),
    'HDD': laptop_data['HDD'].mode()[0],
}, inplace=True)

# Select features and target
features = ['Brand', 'Processor_Name', 'Processor_Brand', 'RAM_Expandable', 'RAM', 'RAM_TYPE', 
            'Ghz', 'Display_type', 'Display', 'GPU', 'GPU_Brand', 'SSD', 'HDD', 'Battery_Life']
target = 'Price'

X = laptop_data[features]
y = laptop_data[target]

# Preprocessing: one-hot encode categorical features
numeric_features = ['Ghz', 'RAM', 'SSD', 'Display', 'Battery_Life']
categorical_features = [col for col in features if col not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create a Random Forest Regressor pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the grid of hyperparameters to search
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2]
}

# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Save the best model to a file
joblib.dump(best_model, 'best_laptop_price_model.pkl')











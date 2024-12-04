

Original file is located at
    https://colab.research.google.com/drive/1ipvfF6TGVaFeV3nY0pJhhqXldV7GZOJ8
"""

# Basic Libraries
import pandas as pd
import numpy as np

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Clustering and Recommendation Libraries
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Saving Models
import joblib

data =pd.read_csv('/content/cleaned_data.csv')
df.dtypes

print(df.info())

df['date'] = pd.to_datetime(df['date'])

# Label Encoding for categorical variables
label_encoder = LabelEncoder()
categorical_cols = ['branch', 'city', 'category', 'payment_method']
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['unit_price', 'quantity', 'rating', 'profit_margin']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Extract additional time-related features
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)

df = df.drop(columns=['date'])


print(df.head())

# Use KMeans for customer segmentation
customer_features = df[['rating', 'quantity', 'unit_price', 'profit_margin']]
kmeans = KMeans(n_clusters=3, random_state=42)
df['customer_segment'] = kmeans.fit_predict(customer_features)


from sklearn.cluster import KMeans


customer_segments = df['customer_segment']

segment_counts = customer_segments.value_counts()
print("Customer Segment Counts:\n", segment_counts)

# For instance, you can calculate the mean values of other features for each segment.


segment_means = df.groupby('customer_segment')['unit_price'].mean()
print("\nMean Unit Price per Segment:\n", segment_means)

# Visualize the segments
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_features, x='unit_price', y='profit_margin', hue=df['customer_segment'], palette='viridis')
plt.title("Customer Segmentation")
plt.xlabel("Unit Price")
plt.ylabel("Profit Margin")
plt.show()

joblib.dump(kmeans, 'customer_segmentation_model.pkl')

# Pivot table for recommendation 
user_item_matrix = df.pivot_table(index='branch', columns='category', values='rating', aggfunc='mean').fillna(0)

similarity_matrix = cosine_similarity(user_item_matrix.T)  # Transpose the matrix to focus on categories

np.save('category_similarity_matrix.npy', similarity_matrix)

category_index = 0  # Replace with actual category index
similar_categories = similarity_matrix[category_index]
recommended_categories = np.argsort(similar_categories)[::-1][1:6]  # Top 5 similar categories
print("Top 5 Recommended Categories:", recommended_categories)



#  features (X) and target (y)
X = df.drop(columns=['profit_margin'])
y = df['profit_margin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)

X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

print(X_train.isnull().sum())
print(X_val.isnull().sum())
print(y_train.isnull().sum())
print(y_val.isnull().sum())

print(X_train.columns)
print(X_val.columns)
.



rf_model = RandomForestRegressor(random_state=42)
xgb_model = xgb.XGBRegressor(random_state=42)
lgb_model = lgb.LGBMRegressor(random_state=42)


rf_param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

xgb_param_dist = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

lgb_param_dist = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [20, 31, 50, 100],
    'min_data_in_leaf': [10, 20, 30, 50],
    'min_split_gain': [0.0, 0.1, 0.2],
    'max_depth': [-1, 5, 10, 15]
}

rf_random_search = RandomizedSearchCV(rf_model, param_distributions=rf_param_dist, n_iter=10,
                                      scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=1)
xgb_random_search = RandomizedSearchCV(xgb_model, param_distributions=xgb_param_dist, n_iter=10,
                                       scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=1)
lgb_random_search = RandomizedSearchCV(lgb_model, param_distributions=lgb_param_dist, n_iter=10,
                                       scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=1)

rf_random_search.fit(X_train, y_train)
xgb_random_search.fit(X_train, y_train)
lgb_random_search.fit(X_train, y_train)

best_rf = rf_random_search.best_estimator_
best_xgb = xgb_random_search.best_estimator_
best_lgb = lgb_random_search.best_estimator_

rf_predictions = best_rf.predict(X_val)
xgb_predictions = best_xgb.predict(X_val)
lgb_predictions = best_lgb.predict(X_val)

rf_rmse = mean_squared_error(y_val, rf_predictions, squared=False)
xgb_rmse = mean_squared_error(y_val, xgb_predictions, squared=False)
lgb_rmse = mean_squared_error(y_val, lgb_predictions, squared=False)

print(f"Random Forest RMSE: {rf_rmse:.4f}")
print(f"XGBoost RMSE: {xgb_rmse:.4f}")
print(f"LightGBM RMSE: {lgb_rmse:.4f}")

rf_cv_rmse = np.mean(cross_val_score(best_rf, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error'))
xgb_cv_rmse = np.mean(cross_val_score(best_xgb, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error'))
lgb_cv_rmse = np.mean(cross_val_score(best_lgb, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error'))

print(f"Random Forest CV RMSE: {rf_cv_rmse:.4f}")
print(f"XGBoost CV RMSE: {xgb_cv_rmse:.4f}")
print(f"LightGBM CV RMSE: {lgb_cv_rmse:.4f}")

def calculate_mase(y_true, y_pred, y_train):
    # Calculate naive forecast (previous value)
    naive_forecast = np.roll(y_true, 1)  # Use y_true instead of y_train
    naive_forecast[0] = y_true[0]  # Set first value to be the same for simplicity
    mae = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true - naive_forecast))
    return mae / mae_naive
rf_mase = calculate_mase(y_val, rf_predictions, y_train)
xgb_mase = calculate_mase(y_val, xgb_predictions, y_train)
lgb_mase = calculate_mase(y_val, lgb_predictions, y_train)

print(f"Random Forest MASE: {rf_mase:.4f}")
print(f"XGBoost MASE: {xgb_mase:.4f}")
print(f"LightGBM MASE: {lgb_mase:.4f}")

best_model = None
best_rmse = min(rf_rmse, xgb_rmse, lgb_rmse)

if best_rmse == rf_rmse:
    best_model = 'Random Forest'
elif best_rmse == xgb_rmse:
    best_model = 'XGBoost'
else:
    best_model = 'LightGBM'

print(f"Best model based on RMSE: {best_model}")


joblib.dump(best_xgb, 'best_xgb_model.pkl')









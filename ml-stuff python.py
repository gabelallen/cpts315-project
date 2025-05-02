from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas

# Load the dataset.
cleanedData_df = pandas.read_csv("USvideos_clean.csv")

# We need to tell it what to predict. We want to predict number of views.
y = cleanedData_df["views"]

# Features: all data except the target
# need to differentiate between numeric, categorical, and text features.
# possibly encode stuff like one-hot encoding for categorical variables
numeric = ["likes", "dislikes", "comment_total"]
categorical = ["channel_title", "category_id"]
text = "title"

# Transformers
preprocessor = ColumnTransformer(transformers=[
    ("num", "passthrough", numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("txt", TfidfVectorizer(), text)
])

# Model pipeline
model = Pipeline([
    ("features", preprocessor),
    ("regressor", RandomForestRegressor())
])

# Split and train
best_score = -1
best_test_size = None

for test_size in [0.1, 0.2, 0.3, 0.4, 0.5]:
  X_train, X_test, y_train, y_test = train_test_split(cleanedData_df, y, test_size=test_size, random_state=42)
  model.fit(X_train, y_train)
  score = model.score(X_test, y_test)
  print(f"Test size: {test_size}, Model score: {score}")
  
  if score > best_score:
    best_score = score
    best_test_size = test_size

print(f"Best test size: {best_test_size}, Best score: {best_score}")

# Evaluate
preds = model.predict(X_test)

print("Predictions:", preds[:5])
print("Actual:", y_test[:5].values)
# The model score represents the coefficient of determination R^2 of the prediction.
# It indicates how well the model predicts the target variable.
# A score of 1.0 means perfect prediction, while a score of 0.0 means the model does no better than predicting the mean of the target variable.
print("Model score:", model.score(X_test, y_test))
print("Mean absolute error:", mean_absolute_error(y_test, preds))
print("Root mean squared error:", mean_squared_error(y_test, preds, squared=False))

print("Feature importances:")
print(model.named_steps["regressor"].feature_importances_)
print("Model parameters:")
print(model.get_params())
print("Pipeline steps:")
print(model.steps)
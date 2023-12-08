import pandas as pd
import numpy as np
import pickle
import operator
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

def calculate_distance(row):
  """
  This function calculates ellipsoidal distance between pairs or coordinates
  in the same row of a data frame
  """
  start_point = (row['slat'], row['slon'])
  end_point = (row['elat'], row['elon'])
  return geodesic(start_point, end_point).kilometers  # Distance in kilometers
  
def miles_to_kilometers(miles):
  """
  This function does a simple conversion of a single value from miles to km
  """
  return miles * 1.60934
  
def yards_to_kilometers(yards):
  """
  This function does a simple conversion of a single value from yards to km
  """
  return yards * 0.0009144

def calculate_tornado_area(row, len_col = "len_km", wid_col = "wid_km" ):
  """
  This function does a simple conversion of a single value from miles to km
  """
  return row[len_col] * row[wid_col]

def time_based_train_test_split(df, perc = 0.95, year_col = 'yr', pred_col = 'fat'):
  """
  # Train Test Split function based on percentual time
  """  
  yr_max = df[year_col].min() + int(perc * (df[year_col].max() - df[year_col].min()))
  df_train = df[df[year_col] <= yr_max]
  df_test = df[df[year_col] > yr_max]
  
  X_train = df_train.drop(columns=[pred_col])
  y_train = df_train[pred_col]
  X_test = df_test.drop(columns=[pred_col])
  y_test = df_test[pred_col]
  
  return X_train, X_test, y_train, y_test

def transform_value(value):
  """
  Transform data for classifier - boolean
  """
  return '1' if value > 0 else '0'
  
def tune_regressor(X_train, y_train, X_val, y_val, model_file = 'best_model_regr.pkl', metric = 'neg_mean_squared_error'):
  
  models = [

    {
      'model': RandomForestRegressor(),
      'param_grid': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        }
        },
    {
      'model': SVR(),
      'param_grid': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3, 4]
        }
    },
    {
      'model': GradientBoostingRegressor(),
      'param_grid': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10]
        }
     }
    ]

  # Perform model comparison and hyperparameter tuning
  best_models = []
  for model_info in models:
    grid_search = GridSearchCV(estimator=model_info['model'], 
    param_grid=model_info['param_grid'], cv=2, scoring=metric, verbose = 1)
    grid_search.fit(X_val, y_val)
    best_model = grid_search.best_estimator_
    best_models.append({'model': best_model, 'score': grid_search.best_score_})

  # Evaluate the best models on the test set
  best_models = sorted(best_models, key=operator.itemgetter('score'), reverse=True)
  
  # Make Prediction on Validation Data Set
  best_model = best_models[0]['model']
  print(best_model)
        
  # Save best model as pickle file
  with open(model_file, 'wb') as file:
    pickle.dump(best_model, file)
    
  return(best_model)

def tune_classifier(X_train, y_train, X_val, y_val, model_file = 'best_model_class.pkl', metric = 'matthews_corrcoef'):

  models = [
    {
      'model': LogisticRegression(),
      'param_grid': {
        'penalty': [None, 'l2'],
        'intercept_scaling': [1, 2, 5, 10],
        'multi_class': ['multinomial']
        }
      },
    {
      'model': RandomForestClassifier(),
      'param_grid': {
        'n_estimators': [3, 5, 10, 20, 50],
        'max_depth': [10, 15, 20],
        'min_samples_split': [200, 500, 1000],
        'min_samples_leaf': [100, 200, 500]
        }
      }
    ]

  # Perform model comparison and hyperparameter tuning
  best_models = []
  for model_info in models:
    grid_search = GridSearchCV(estimator=model_info['model'], 
    param_grid=model_info['param_grid'], cv=2, scoring=metric, verbose = 1)
    grid_search.fit(X_val, y_val)
    best_model = grid_search.best_estimator_
    best_models.append({'model': best_model, 'score': grid_search.best_score_})
    
  # Evaluate the best models on the test set
  best_models = sorted(best_models, key=operator.itemgetter('score'), reverse=True)
  
  # Make Prediction on Validation Data Set
  best_model = best_models[0]['model']
  print(best_model)
        
  # Save best model as pickle file
  with open(model_file, 'wb') as file:
    pickle.dump(best_model, file)
    
  return(best_model)

def plt_error_map(y_train, y_val, y_test, y_train_pred, y_val_pred, y_test_pred):
  """
  Plot Error Map
  """
  plt.clf()
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  
  # Train
  # m = int(np.expm1(np.concatenate([y_train, y_val, y_test])).max())
  m = int(np.concatenate([y_train, y_val, y_test]).max())
  
  # axes[0].scatter(np.expm1(y_train), np.expm1(y_train_pred), alpha=0.5)
  axes[0].scatter(y_train, y_train_pred, alpha=0.5)

  axes[0].set_xlabel("Real Values")
  axes[0].set_ylabel("Prediction")
  axes[0].set_xlim(0, m)
  axes[0].set_ylim(0, m)
  axes[0].plot([0, m], [0, m], linestyle='--', color='red')
  axes[0].set_title("Error Map of Train Set")
  
  # Validation
  # axes[1].scatter(np.expm1(y_val), np.expm1(y_val_pred), alpha=0.5)
  axes[1].scatter(y_val, y_val_pred, alpha=0.5)
  axes[1].set_xlabel("Real Values")
  axes[1].set_ylabel("Prediction")
  axes[1].set_xlim(0, m)
  axes[1].set_ylim(0, m)
  axes[1].plot([0, m], [0, m], linestyle='--', color='red')
  axes[1].set_title("Error Map of Val Set")
  
  # Test
  # axes[2].scatter(np.expm1(y_test), np.expm1(y_test_pred), alpha=0.5)
  axes[2].scatter(y_test, y_test_pred, alpha=0.5)
  axes[2].set_xlabel("Real Values")
  axes[2].set_ylabel("Prediction")
  axes[2].set_xlim(0, m)
  axes[2].set_ylim(0, m)
  axes[2].plot([0, m], [0, m], linestyle='--', color='red')
  axes[2].set_title("Error Map of Test Set")
  
  # Adjust layout
  plt.tight_layout()
  
  # Save the figure
  plt.savefig("error-map.png")
  
  # Show the plots
  plt.show()
  
def plt_error_kernel(y_train, y_val, y_test, y_train_pred, y_val_pred, y_test_pred):
    """
    Plot Error Map with Kernel Density Estimates
    """
    plt.clf()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Calculate maximum value for axis limits
    # m = int(np.expm1(np.concatenate([y_train, y_val, y_test])).max())
    m = int(np.concatenate([y_train, y_val, y_test]).max())

    # Plot for Train Set
    # dt = pd.DataFrame({'Real': np.expm1(np.array(y_train)), 'Pred': np.expm1(y_train_pred)})
    dt = pd.DataFrame({'Real': np.array(y_train), 'Pred': y_train_pred})
    
    sns.jointplot(data=dt, x="Real", y="Pred", kind="hex")
    plt.savefig("error-kernel-train.png")
    plt.show()
    
    # Plot for Validation Set
    # dt = pd.DataFrame({'Real': np.expm1(np.array(y_val)), 'Pred': np.expm1(y_val_pred)})
    dt = pd.DataFrame({'Real': np.array(y_val), 'Pred': y_val_pred})
    
    sns.jointplot(data=dt, x="Real", y="Pred", kind="hex")
    plt.savefig("error-kernel-val.png")
    plt.show()

    # Plot for Test Set
    # dt = pd.DataFrame({'Real': np.expm1(np.array(y_test)), 'Pred': np.expm1(y_test_pred)})
    dt = pd.DataFrame({'Real': np.array(y_test), 'Pred': y_test_pred})

    sns.jointplot(data=dt, x="Real", y="Pred", kind="hex")
    plt.savefig("error-kernel-test.png")
    plt.show()

# Inputs
data_file = 'input/us_tornado_dataset_1950_2021.csv'

# Data wrangling ###############################################################
print("Wrangling the data... \n")
df = pd.read_csv(data_file)
# df.loc[df.mag==-9, 'mag'] = 0 #unknown magnitude when -9, so 0
df.loc[df.elat == 0.0, 'elat'] = df['slat'] #unknown final lat/long --> = to initial ones
df.loc[df.elon == 0.0, 'elon'] = df['slon']

df['traveled_d_km'] = df.apply(calculate_distance, axis=1)
df['len_km'] = df['len'].apply(miles_to_kilometers)
df['wid_km'] = df['wid'].apply(yards_to_kilometers)
df['area_sq_km'] = df.apply(calculate_tornado_area, axis=1)

# Format 'mo' and 'dy' columns with leading zeros
df['mo'] = df['mo'].apply(lambda x: str(x).zfill(2))
df['dy'] = df['dy'].apply(lambda x: str(x).zfill(2))

# Create a new 'date' column in the "year-month-day" format
df['date'] = pd.to_datetime(df[['yr', 'mo', 'dy']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d')
df['yr'] = df['yr'].astype(int)
df['mo'] = df['mo'].astype(int)
df['dy'] = df['dy'].astype(int)

# Extract the day of the week and create a new 'day_of_week' column
df['day_of_week'] = df['date'].dt.dayofweek

# Keeping only relevant columns
df = df[['yr', 'mo', 'dy', 'fat', 'traveled_d_km','area_sq_km', 'day_of_week']]

# Outlier Removal
df = df[df['fat'] < 30] # histogram based
df = df[df['traveled_d_km'] < 50] # histogram based
df = df[df['area_sq_km'] < 20] # histogram based

# Filtering death
df_zero = df[df.fat==0]

# Filtering death 
df_g = df[df.fat>0]

# replication factor to balance death
rep_f = int(len(df_zero) / len(df_g))

# oversample of death
df_gg = pd.concat([df_g] * rep_f, ignore_index=True)

# combining dfs again
frames = [df_zero, df_gg]
df = pd.concat(frames, ignore_index=True)

######## Classification ########################################################
print("Classifying... \n")
df.loc[:, 'has_fat'] = df['fat'].apply(transform_value)
df['has_fat'] = df['has_fat'].astype('category')

X = df.drop(columns = ['has_fat', 'fat'])
y = df.has_fat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Model Training and/or hyperparameter tuning ##################################
# Check if there's an available model
model_file = 'best_model_class.pkl'
try:
  with open(model_file, 'rb') as file:
      best_classifier = pickle.load(file)
except:
  best_classifier = tune_classifier(X_train, y_train, X_val, y_val, model_file = model_file)

print(best_classifier)

# Calculate MCC for the training set
y_train_pred = best_classifier.predict(X_train)
mcc_train = matthews_corrcoef(y_train, y_train_pred).round(3)
acc_train = accuracy_score(y_train, y_train_pred).round(3)

# Calculate MCC for the validation set
y_val_pred = best_classifier.predict(X_val)
mcc_val = matthews_corrcoef(y_val, y_val_pred).round(3)
acc_val = accuracy_score(y_val, y_val_pred).round(3)

# Calculate MCC for the test set
y_test_pred = best_classifier.predict(X_test)
mcc_test = matthews_corrcoef(y_test, y_test_pred).round(3)
acc_test = accuracy_score(y_test, y_test_pred).round(3)

print(f"Training MCC: {mcc_train}")
print(f"Validation MCC: {mcc_val}")
print(f"Test MCC: {mcc_test} \n")
print(f"Training Accuracy: {acc_train}")
print(f"Validation Accuracy: {acc_val}")
print(f"Test Accuracy: {acc_test} \n")

# Confusion Matrix
# Training Confusion Matrix
print("Training Confusion Matrix \n")
conf_matrix = confusion_matrix(y_train, y_train_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
print(conf_matrix_df)

# Validation Confusion Matrix
print("Validation Confusion Matrix \n")
conf_matrix = confusion_matrix(y_val, y_val_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
print(conf_matrix_df)

# Testing Confusion Matrix
print("Testing Confusion Matrix \n")
conf_matrix = confusion_matrix(y_test, y_test_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
print(conf_matrix_df)

# Classification Feature Importance
try:
  coefficients = best_classifier.coef_[0]
  feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})
  feature_importance = feature_importance.sort_values('Importance', ascending=False)
  feature_importance.plot.bar(x='Feature', y='Importance', figsize=(10, 6))
  plt.xlabel('Feature')
  plt.ylabel('Importance')
  plt.title('Feature Importance')
  plt.savefig('classifier-feat-importance.png')
  plt.show()
except:
  coefficients = best_classifier.feature_importances_
  feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})
  feature_importance = feature_importance.sort_values('Importance', ascending=False)
  feature_importance.plot.bar(x='Feature', y='Importance', figsize=(12, 12))
  plt.xticks(rotation=45, ha='right')
  plt.xlabel('Feature')
  plt.ylabel('Importance')
  plt.title('Feature Importance')
  plt.savefig('classifier-feat-importance.png')
  plt.show()
  
####### Regression #############################################################
print("Regression... \n")
# Use the classified tornadoes as test for regression
df_g = df_g[['yr', 'mo', 'dy', 'fat', 'traveled_d_km','area_sq_km', 'day_of_week']].drop_duplicates()
X = df_g.drop(columns=['fat'])
y = df_g['fat']

################################################################################
# df_g['fat'] = np.log1p(df_g['fat'])
################################################################################

# dft = df_g.groupby('yr')['fat'].sum().reset_index().drop_duplicates()
# plt.bar(dft['yr'], dft['fat'], color='blue', alpha=0.7)
# plt.xlabel('Year')
# plt.ylabel('Fatalities')
# plt.title('Fatalities Over Time')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Model Training and/or hyperparameter tuning ##################################
# Check if there's an available model
model_file = 'best_model_regr.pkl'
try:
  with open(model_file, 'rb') as file:
    best_regressor = pickle.load(file)
except:
  best_regressor = tune_regressor(X_train, y_train, X_val, y_val, model_file = model_file)
  
print(best_regressor)

# Calculate MSE for the training set
y_train_pred = best_regressor.predict(X_train)
# mse_train = mean_squared_error(np.expm1(y_train).astype(int), np.expm1(y_train_pred).astype(int)).round(3)
# r2_train = r2_score(np.expm1(y_train).astype(int), np.expm1(y_train_pred).astype(int)).round(3)
mse_train = mean_squared_error(y_train.astype(int), y_train_pred.astype(int)).round(3)
r2_train = r2_score(y_train.astype(int), y_train_pred.astype(int)).round(3)

# Calculate MSE for the validation set
y_val_pred = best_regressor.predict(X_val)
# mse_val = mean_squared_error(np.expm1(y_val).astype(int), np.expm1(y_val_pred).astype(int)).round(3)
# r2_val = r2_score(np.expm1(y_val).astype(int), np.expm1(y_val_pred).astype(int)).round(3)
mse_val = mean_squared_error(y_val.astype(int), y_val_pred.astype(int)).round(3)
r2_val = r2_score(y_val.astype(int), y_val_pred.astype(int)).round(3)

# Calculate MSE for the test set
y_test_pred = best_regressor.predict(X_test)
# mse_test = mean_squared_error(np.expm1(y_test).astype(int), np.expm1(y_test_pred).astype(int)).round(3)
# r2_test = r2_score(np.expm1(y_test).astype(int), np.expm1(y_test_pred).astype(int)).round(3)
mse_test = mean_squared_error(y_test.astype(int), y_test_pred.astype(int)).round(3)
r2_test = r2_score(y_test.astype(int), y_test_pred.astype(int)).round(3)

print(f"Training MSE: {mse_train}")
print(f"Validation MSE: {mse_val}")
print(f"Test MSE: {mse_test} \n")
print(f"Training R²: {r2_train}")
print(f"Validation R²: {r2_val}")
print(f"Test R²: {r2_test} \n")

# Regressor Feature Importance
# As the best model can differ, we use a try. Not always it will be simple to get the feature importance
try:
  coefficients = best_regressor.feature_importances_
  feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': np.abs(coefficients)})
  feature_importance = feature_importance.sort_values('Importance', ascending=False)
  feature_importance.plot.bar(x='Feature', y='Importance', figsize=(12, 12))
  plt.xticks(rotation=45, ha='right')
  plt.xlabel('Feature')
  plt.ylabel('Importance')
  plt.title('Feature Importance')
  plt.savefig('regression-feat-importance.png')
  plt.show()
except:
  coefficients = best_regressor.coef_
  feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': np.abs(coefficients)})
  feature_importance = feature_importance.sort_values('Importance', ascending=False)
  feature_importance.plot.bar(x='Feature', y='Importance', figsize=(12, 12))
  plt.xticks(rotation=45, ha='right')
  plt.xlabel('Feature')
  plt.ylabel('Importance')
  plt.title('Feature Importance')
  plt.savefig('regression-feat-importance.png')
  plt.show()

# todo exlude least correlated columns and retrain
# sd = np.std(np.expm1(df_g.fat)).round(1)
# m = np.mean(np.expm1(df_g.fat)).round(1)

sd = np.std(df_g.fat).round(1)
m = np.mean(df_g.fat).round(1)

print(f"SD of Fatalities when existent: {sd}")
print(f"Mean of Fatalities when existent: {m}")

# Graph with Error
plt_error_map(y_train, y_val, y_test, y_train_pred, y_val_pred, y_test_pred)
plt_error_kernel(y_train, y_val, y_test, y_train_pred, y_val_pred, y_test_pred)

print("Done!")

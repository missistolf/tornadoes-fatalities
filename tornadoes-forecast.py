import pandas as pd
import numpy as np
import pickle
import operator
from typing import Dict
from typing import Any
from typing import Union
from typing import Tuple
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
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import d2_pinball_score
from sklearn.metrics import mean_absolute_percentage_error

def calculate_distance(row: Dict[str, float]) -> float:
    """
    This function calculates ellipsoidal distance between pairs of coordinates
    in the same row of a data frame.

    Parameters:
        row (Dict[str, float]): A dictionary containing float values for
                               'slat', 'slon', 'elat', and 'elon'.

    Returns:
        float: The calculated ellipsoidal distance in kilometers.
    """
    start_point = (row['slat'], row['slon'])
    end_point = (row['elat'], row['elon'])
    return geodesic(start_point, end_point).kilometers
  
  
def miles_to_kilometers(miles: float) -> float:
    """
    This function does a simple conversion of a single value from miles to km.

    Parameters:
        miles (float): The distance value in miles.

    Returns:
        float: The converted distance value in kilometers.
    """
    return miles * 1.60934

  
def yards_to_kilometers(yards: float) -> float:
    """
    This function does a simple conversion of a single value from yards to km.

    Parameters:
        yards (float): The distance value in yards.

    Returns:
        float: The converted distance value in kilometers.
    """
    return yards * 0.0009144


def calculate_tornado_area(row: Dict[str, float], len_col: str = "len_km", wid_col: str = "wid_km") -> float:
    """
    This function calculates the area of a tornado given the length and width values in kilometers.

    Parameters:
        row (Dict[str, float]): A dictionary containing float values for 'len_km' and 'wid_km'.
        len_col (str): The key representing the length in kilometers in the 'row' dictionary. Default is "len_km".
        wid_col (str): The key representing the width in kilometers in the 'row' dictionary. Default is "wid_km".

    Returns:
        float: The calculated area of the tornado in square kilometers.
    """
    return row[len_col] * row[wid_col]


def time_based_train_test_split(df: pd.DataFrame, perc: float = 0.95, year_col: str = 'yr', pred_col: str = 'fat') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train Test Split function based on percentual time

    Parameters:
        df (DataFrame): The input DataFrame.
        perc (float): The percentage of data to include in the training set. Default is 0.95.
        year_col (str): The column name representing the year in the DataFrame. Default is 'yr'.
        pred_col (str): The column name representing the target variable in the DataFrame. Default is 'fat'.

    Returns:
        Tuple[DataFrame, DataFrame, Series, Series]: A tuple containing X_train, X_test, y_train, and y_test.
    """
    yr_max = df[year_col].min()+int(perc*(df[year_col].max()-df[year_col].min()))
    df_train = df[df[year_col] <= yr_max]
    df_test = df[df[year_col] > yr_max]

    X_train = df_train.drop(columns=[pred_col])
    y_train = df_train[pred_col]
    X_test = df_test.drop(columns=[pred_col])
    y_test = df_test[pred_col]

    return X_train, X_test, y_train, y_test


def transform_value(value: float) -> str:
    """
    Transform data for classifier - boolean

    Parameters:
        value (float): The numeric value to be transformed.

    Returns:
        str: The transformed value as a string, '1' if greater than 0, '0' otherwise.
    """
    return '1' if value > 0 else '0'
  

def tune_regressor(X_train: Any, y_train: Any, X_val: Any, y_val: Any, model_file: str = 'model/best_model_regr.pkl', metric: str = 'neg_mean_squared_error') -> Any:
    """
    Tune regressor models using grid search and save the best model.

    Parameters:
        X_train (Any): The features of the training set.
        y_train (Any): The target variable of the training set.
        X_val (Any): The features of the validation set.
        y_val (Any): The target variable of the validation set.
        model_file (str): The file path to save the best model as a pickle file. Default is 'model/best_model_regr.pkl'.
        metric (str): The scoring metric for grid search. Default is 'neg_mean_squared_error'.

    Returns:
        Any: The best-tuned regressor model.
    """
    models = [
        {
            'model': RandomForestRegressor(),
            'param_grid': {
                'n_estimators': [10, 20, 50],
                'max_depth': [None, 10, 20, 50],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [5, 10]
            }
        },
        {
            'model': SVR(),
            'param_grid': {
                'C': [0.1, 1],
                'kernel': ['rbf', 'poly'],
                'degree': [1, 2, 3],
            }
        },
        {
            'model': GradientBoostingRegressor(),
            'param_grid': {
                'n_estimators': [10, 20, 50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [10, 20, 50],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [5, 10]
            }
        }
    ]

    # Perform model comparison and hyperparameter tuning
    best_models = []
    for model_info in models:
        grid_search = GridSearchCV(estimator=model_info['model'], param_grid=model_info['param_grid'], cv=2, scoring=metric, verbose=1)
        grid_search.fit(X_val, y_val)
        best_model = grid_search.best_estimator_
        best_models.append({'model': best_model, 'score': grid_search.best_score_})

    # Evaluate the best models on the test set
    best_models = sorted(best_models, key=operator.itemgetter('score'), reverse=True)

    # Make Prediction on Validation Data Set
    best_model = best_models[0]['model']
    print(best_model)

    # Save the best model as a pickle file
    with open(model_file, 'wb') as file:
        pickle.dump(best_model, file)

    return best_model


def tune_classifier(X_train: Any, y_train: Any, X_val: Any, y_val: Any, model_file: str = 'model/best_model_class.pkl', metric: str = 'matthews_corrcoef') -> Any:
    """
    Tune classifier models using grid search and save the best model.

    Parameters:
        X_train (Any): The features of the training set.
        y_train (Any): The target variable of the training set.
        X_val (Any): The features of the validation set.
        y_val (Any): The target variable of the validation set.
        model_file (str): The file path to save the best model as a pickle file. Default is 'model/best_model_class.pkl'.
        metric (str): The scoring metric for grid search. Default is 'matthews_corrcoef'.

    Returns:
        Any: The best-tuned classifier model.
    """
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
        grid_search = GridSearchCV(estimator=model_info['model'], param_grid=model_info['param_grid'], cv=2, scoring=metric, verbose=1)
        grid_search.fit(X_val, y_val)
        best_model = grid_search.best_estimator_
        best_models.append({'model': best_model, 'score': grid_search.best_score_})

    # Evaluate the best models on the test set
    best_models = sorted(best_models, key=operator.itemgetter('score'), reverse=True)

    # Make Prediction on Validation Data Set
    best_model = best_models[0]['model']
    print(best_model)

    # Save the best model as a pickle file
    with open(model_file, 'wb') as file:
        pickle.dump(best_model, file)

    return best_model
  
def plt_error_map(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, y_train_pred: np.ndarray, y_val_pred: np.ndarray, y_test_pred: np.ndarray, output_path: str) -> None:
    """
    Plot Error Map

    Parameters:
        y_train (np.ndarray): The true values of the target variable for the training set.
        y_val (np.ndarray): The true values of the target variable for the validation set.
        y_test (np.ndarray): The true values of the target variable for the test set.
        y_train_pred (np.ndarray): The predicted values for the training set.
        y_val_pred (np.ndarray): The predicted values for the validation set.
        y_test_pred (np.ndarray): The predicted values for the test set.
        output_path (str): The file path to save the error map plot.
    """
    plt.clf()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Train
    m = int(np.concatenate([y_train, y_val, y_test]).max())
    axes[0].scatter(y_train, y_train_pred, alpha=0.5)
    axes[0].set_xlabel("Real Values")
    axes[0].set_ylabel("Prediction")
    axes[0].set_xlim(0, m)
    axes[0].set_ylim(0, m)
    axes[0].plot([0, m], [0, m], linestyle='--', color='red')
    axes[0].set_title("Error Map of Train Set")

    # Validation
    axes[1].scatter(y_val, y_val_pred, alpha=0.5)
    axes[1].set_xlabel("Real Values")
    axes[1].set_ylabel("Prediction")
    axes[1].set_xlim(0, m)
    axes[1].set_ylim(0, m)
    axes[1].plot([0, m], [0, m], linestyle='--', color='red')
    axes[1].set_title("Error Map of Val Set")

    # Test
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
    plt.savefig(output_path)

    # Show the plots
    plt.show()
    
    
def plt_error_kernel(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, y_train_pred: np.ndarray, y_val_pred: np.ndarray, y_test_pred: np.ndarray) -> None:
    """
    Plot Error Map with Kernel Density Estimates

    Parameters:
        y_train (np.ndarray): The true values of the target variable for the training set.
        y_val (np.ndarray): The true values of the target variable for the validation set.
        y_test (np.ndarray): The true values of the target variable for the test set.
        y_train_pred (np.ndarray): The predicted values for the training set.
        y_val_pred (np.ndarray): The predicted values for the validation set.
        y_test_pred (np.ndarray): The predicted values for the test set.
    """
    plt.clf()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Calculate maximum value for axis limits
    m = int(np.concatenate([y_train, y_val, y_test]).max())

    # Plot for Train Set
    dt_train = pd.DataFrame({'Real': y_train, 'Pred': y_train_pred})
    sns.jointplot(data=dt_train, x="Real", y="Pred", kind="hex")
    plt.savefig("output/error-kernel-train.png")
    plt.show()

    # Plot for Validation Set
    dt_val = pd.DataFrame({'Real': y_val, 'Pred': y_val_pred})
    sns.jointplot(data=dt_val, x="Real", y="Pred", kind="hex")
    plt.savefig("output/error-kernel-val.png")
    plt.show()

    # Plot for Test Set
    dt_test = pd.DataFrame({'Real': y_test, 'Pred': y_test_pred})
    sns.jointplot(data=dt_test, x="Real", y="Pred", kind="hex")
    plt.savefig("output/error-kernel-test.png")
    plt.show()
    
# Main     
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
X_train, X_test, y_train, y_test = time_based_train_test_split(df.drop(columns = ['fat']), perc=0.95, pred_col = "has_fat")
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Save for later... save for regression testing
train_rm, X_test_regr, y_train_rm, y_test_regr = time_based_train_test_split(df.drop(columns = ['has_fat']), perc=0.95, pred_col = "fat")

# Model Training and/or hyperparameter tuning ##################################
# Check if there's an available model
model_file = 'model/best_model_class.pkl'
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
  plt.savefig('output/classifier-feat-importance.png')
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
  plt.savefig('output/classifier-feat-importance.png')
  plt.show()
  
####### Regression #############################################################
print("Regression... \n")
# Use the classified tornadoes as test for regression
X_test_regr['class'] = y_test_pred
X_test_regr['fat'] = y_test_regr
X_test_final = X_test_regr[X_test_regr['class'] == '1'].drop(columns = ['class'])
# X_test_final = X_test_regr[X_test_regr['fat'] > 0].drop(columns = ['class'])
X_test_final = X_test_final.drop_duplicates()

# df_g = df_g[['yr', 'mo', 'dy', 'fat', 'traveled_d_km','area_sq_km', 'day_of_week']].drop_duplicates()
df_g = df_g[['yr', 'mo', 'dy', 'fat', 'traveled_d_km','area_sq_km', 'day_of_week']]

# df_g.loc[:, 'fat'] = np.log1p(df_g['fat'])
plt.hist(df_g.fat, bins=30, edgecolor='k')  # You can adjust the number of bins as needed
plt.show()

################################################################################
# df_g['fat'] = np.log1p(df_g['fat'])
################################################################################

# dft = df_g.groupby('yr')['fat'].sum().reset_index().drop_duplicates()
# plt.bar(dft['yr'], dft['fat'], color='blue', alpha=0.7)
# plt.xlabel('Year')
# plt.ylabel('Fatalities')
# plt.title('Fatalities Over Time')
# plt.show()

X_train, X_test, y_train, y_test = time_based_train_test_split(df = df_g, perc=0.95, year_col="yr", pred_col='fat')
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Overwrite test dataset with the one classifed
X_test = X_test_final.drop(columns = ['fat'])
# y_test = np.log1p(X_test_final['fat'])
y_test = X_test_final['fat']

# Model Training and/or hyperparameter tuning ##################################
# Check if there's an available model
model_file = 'model/best_model_regr.pkl'
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
msle_train = mean_squared_log_error(y_train.astype(int), y_train_pred.astype(int)).round(3)
# pinball_train = d2_pinball_score(y_train.astype(int), y_train_pred.astype(int)).round(3)
# mape_train = mean_absolute_percentage_error(y_train.astype(int), y_train_pred.astype(int)).round(3)

# Calculate MSE for the validation set
y_val_pred = best_regressor.predict(X_val)
# mse_val = mean_squared_error(np.expm1(y_val).astype(int), np.expm1(y_val_pred).astype(int)).round(3)
# r2_val = r2_score(np.expm1(y_val).astype(int), np.expm1(y_val_pred).astype(int)).round(3)
mse_val = mean_squared_error(y_val.astype(int), y_val_pred.astype(int)).round(3)
msle_val = mean_squared_log_error(y_val.astype(int), y_val_pred.astype(int)).round(3)
# pinball_val = d2_pinball_score(y_val.astype(int), y_val_pred.astype(int)).round(3)
# mape_val = mean_absolute_percentage_error(y_val.astype(int), y_val_pred.astype(int)).round(3)

# Calculate MSE for the test set
y_test_pred = best_regressor.predict(X_test)
# mse_test = mean_squared_error(np.expm1(y_test).astype(int), np.expm1(y_test_pred).astype(int)).round(3)
# r2_test = r2_score(np.expm1(y_test).astype(int), np.expm1(y_test_pred).astype(int)).round(3)
mse_test = mean_squared_error(y_test.astype(int), y_test_pred.astype(int)).round(3)
msle_test = mean_squared_log_error(y_test.astype(int), y_test_pred.astype(int)).round(3)
# pinball_test = d2_pinball_score(y_test.astype(int), y_test_pred.astype(int)).round(3)
mape_test = mean_absolute_percentage_error(y_test.astype(int), y_test_pred.astype(int)).round(3)

print(f"Training MSE: {mse_train}")
print(f"Validation MSE: {mse_val}")
print(f"Test MSE: {mse_test} \n")
print(f"Training MSLE: {msle_train}")
print(f"Validation  MSLE: {msle_val}")
print(f"Test  MSLE: {msle_test} \n")

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
  plt.savefig('output/regression-feat-importance.png')
  plt.show()
except:
  try:
    coefficients = best_regressor.coef_
    feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': np.abs(coefficients.squeeze())})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    feature_importance.plot.bar(x='Feature', y='Importance', figsize=(12, 12))
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.savefig('output/regression-feat-importance.png')
    plt.show()
  except:
    print("No Feature Importance Possible for this Model")
    
# todo exlude least correlated columns and retrain
# sd = np.std(np.expm1(df_g.fat)).round(1)
# m = np.mean(np.expm1(df_g.fat)).round(1)

sd = np.std(df_g.fat).round(1)
m = np.mean(df_g.fat).round(1)

print(f"SD of Fatalities when existent: {sd}")
print(f"Mean of Fatalities when existent: {m}")

# Graph with Error
plt_error_map(y_train, y_val, y_test, y_train_pred, y_val_pred, y_test_pred, output_path = "output/map-error.png")
plt_error_kernel(y_train, y_val, y_test, y_train_pred, y_val_pred, y_test_pred)

df_train = pd.DataFrame({'pred': y_train_pred, 'real': y_train})
df_train = df_train.groupby('real')['pred'].mean().reset_index()
df_val = pd.DataFrame({'pred': y_val_pred, 'real': y_val})
df_val = df_val.groupby('real')['pred'].mean().reset_index()
df_test = pd.DataFrame({'pred': y_test_pred, 'real': y_test})
df_test = df_test.groupby('real')['pred'].mean().reset_index()

plt_error_map(df_train.real, df_val.real, df_test.real, df_train.pred, df_val.pred, df_test.pred, output_path = "output/map-error-linear.png")

print("Done!")

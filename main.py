import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tabulate import tabulate
# install pandas, sklearn, statsmodels, xgboost, tabulate

# DATA PREPROCESSING
# read in the raw data
raw_data = pd.read_csv(os.getcwd() + '\insurance_claims.csv')
print(raw_data.head())
# replace all missing data with NAN
raw_data.replace('?', np.nan, inplace = True)

# the columns that are missing data are collision_type, property_damage and police_report_available
# taking a look at the data, collision_type is missing for all rows where the incident_type is either vehicle theft or parked car
# this makes sense because in both cases, the collision type cannot be confirmed as the policy holder was not present to witness the collision
# for this column, we will replace all nan values with 'unknown'
print(raw_data.loc[raw_data['collision_type'].isna()]['incident_type'].unique())
incident_type_no_witness = ['Vehicle Theft', 'Parked Car']
print(raw_data.loc[raw_data['incident_type'].isin(incident_type_no_witness)]['collision_type'].unique())
raw_data['collision_type'].replace(np.nan, 'Unknown', inplace = True)
print(raw_data['collision_type'].head())

# looking at the data, there doesn't seem to be a connection between the columns of entries in which the property_damage or police_report_available data is missing
# common sense would dictate that if property damage and a police report were present in the accident, it would be less likely to be a case of fraud
# however, these details not being present is not a good enough indicator of fraud
# these do not seem like values that we can impute with accuracy, so we will also change these missing values to 'unknown'
# maybe there is some correlation in the missing values that was not perceived
raw_data['property_damage'].replace(np.nan, 'Unknown', inplace = True)
raw_data['police_report_available'].replace(np.nan, 'Unknown', inplace = True)
print(raw_data['property_damage'].unique())
print(raw_data['property_damage'].unique())

# EVALUATING A BASELINE MODEL
# the baseline model will only use the data of numerical columns
# all other columns are defined as non_numerical_cols and dropped
non_numerical_cols = ['policy_csl', 'insured_sex', 'insured_education_level', 'insured_relationship', 'incident_type', 'collision_type', 'incident_severity', 'authorities_contacted',
'property_damage', 'police_report_available', 'policy_bind_date', 'insured_zip', 'insured_hobbies', 'incident_date', 'incident_city', 'incident_location', 'auto_make', 'auto_model',
 'policy_state', 'insured_occupation', 'incident_state']
baseline_X = raw_data.copy()
baseline_X.drop(non_numerical_cols, inplace=True, axis=1)
# the target variable fraud_reported is also dropped and the values are mapped to 1 & 0 corresponding to whether the instance was a case of fraud
baseline_X = baseline_X.drop('fraud_reported', axis=1)
y = raw_data['fraud_reported'].map({'Y': 1, 'N': 0})
# the data is divided into a 70/30 split
train_bl_X, val_bl_X, train_bl_y, val_bl_y = train_test_split(baseline_X, y, test_size=0.25)
# the baseline model is defined, fit and the accuracy score is recorded to be referenced later
baseline_model = DecisionTreeClassifier(random_state=0)
baseline_model.fit(train_bl_X, train_bl_y)
val_predictions = baseline_model.predict(val_bl_X)
print('\n\n\nACCURACY SCORE:')
baseline_score = accuracy_score(val_bl_y, val_predictions)
count_correct_predictions = np.count_nonzero(val_bl_y == val_predictions)
print(baseline_score)
print(count_correct_predictions)

# FEATURE ENGINEERING
# the columns of interest upon first glance are divided into numerical and categorical columns
numerical_cols = ['months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium', 'umbrella_limit', 'capital_gains', 'capital_loss', 'incident_hour_of_the_day',
'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim']
categorical_cols = ['policy_csl', 'insured_sex', 'insured_education_level', 'insured_relationship', 'incident_type', 'collision_type', 'incident_severity', 'authorities_contacted',
'property_damage', 'police_report_available']

# the categorical columns do not contain an excessive number of unique values, so they can be converted into indicator variables
print()
for col in categorical_cols:
    print(col)
    print(raw_data[col].unique())
    print('\n')

# the categorical columns are converted into indicator variables using pd.get_dummies, dropping the first to help reduce the extra columns created,
# thus reducing the correlations created among dummy variables
# all of the relevant data is combined into X
categorical_df = pd.get_dummies(raw_data[categorical_cols], drop_first=True)
numerical_df = raw_data[numerical_cols]
X = pd.concat([numerical_df, categorical_df], axis=1)
print(X.head())

# IDENTIFYING MULTICOLLINEARITY
def get_VIF(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    return vif_data

# from looking at the VIF scores, we can see that:
# age is highly correlated to months:as_customer 
# total_claim_amount, vehicle_claim, property_claim, injury_claim are all highly correlated as total_claim_amount is a sum of the rest
# collision_type_unknown is highly correlated to incident_type_Vehicle Theft and incident_type_Parked Car because the collison type is always unknown in those two incident types
# incident_type_Single Vehicle Collision is highly correlated to number_of_vehicles_involved
# policy_annual_premium and vehicle_claim are highly correlated to other variables because the are determined depending on the other variables
vif_before = get_VIF(X)
print(vif_before.sort_values('VIF'))

# the columns with high VIF scores are dropped, reducing the multicollinearity within the model
X.drop(['age', 'total_claim_amount', 'collision_type_Unknown', 'incident_type_Single Vehicle Collision', 'policy_annual_premium', 'vehicle_claim'], inplace=True, axis=1)
vif_after = get_VIF(X)
print(vif_after.sort_values('VIF'))

#train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.25)

# define the models and their associated parameters for hypertuning
dt_model = DecisionTreeClassifier(random_state=0)
rf_model = RandomForestClassifier(random_state=0)
gb_model = GradientBoostingClassifier(random_state=0)
xgb_model = XGBClassifier(random_state=0)
knn_model = KNeighborsClassifier()
all_models = {'Decision Tree': dt_model, 'Random Forest': rf_model, 'Gradient Boost': gb_model, 'XG Boost': xgb_model, 'K Nearest Neighbours': knn_model}

dt_params = {'max_depth': [3, 5, 10, 15, 20],
             'min_samples_leaf': [5, 10, 20, 50, 100]}
rf_params = {'n_estimators': [10, 25, 50, 75, 100],
             'max_depth': [3, 5, 10, 15, 20],
             'min_samples_leaf': [5, 10, 20, 50, 100]}
gb_params = {'learning_rate': [0.1, 0.25, 0.5, 1, 5],
             'n_estimators': [10, 25, 50, 75, 100],
             'max_depth': [3, 5, 10, 15, 20],
             'min_samples_leaf': [5, 10, 20, 50, 100]}
xgb_params = {'learning_rate': [0.1, 0.25, 0.5, 1, 5],
             'n_estimators': [10, 25, 50, 75, 100],
             'max_depth': [3, 5, 10, 15, 20]}
knn_params = {'n_neighbors': [10, 20, 30, 40, 50],
              'leaf_size': [10, 20, 30, 40, 50]}

all_models = {'Decision Tree': [dt_model, dt_params], 'Random Forest': [rf_model, rf_params], 'Gradient Boost': [gb_model, gb_params],
               'XG Boost': [xgb_model, xgb_params], 'K Nearest Neighbours': [knn_model, knn_params]}

# define the folds to be used for cross validation
k_folds = KFold(n_splits=5, shuffle=True, random_state=0)


def evaluate_grid_search_cv(estimator, params, folds, X, y):
    grid_search = GridSearchCV(estimator=estimator, param_grid=params, cv=folds, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    return [best_model, best_score]


def evaluate_model(model, train_X, train_y, val_X, val_y):
    model.fit(train_X, train_y)
    model_predictions = model.predict(val_X)
    model_accuracy = accuracy_score(val_y, model_predictions)
    return model_accuracy


print("-----------------------------------------------------")
# loop through all of the models and their parameters, evaluating each on the training data and storing the highest scoring model/parameters for each algorithm
# the best scoring model is then used to make predicitons for the validation data and the accuracy is stored
for key, items in all_models.items():
    print("Training " + key + " Model...")
    model = items[0]
    params = items[1]
    [best_model, best_score] = evaluate_grid_search_cv(model, params, k_folds, X, y)
    items.extend([str(best_model), best_score])

# display each algorithm with its best performing model, training accuracy and validation accuracy
best_score_df = pd.DataFrame.from_dict(data=all_models, orient='index', columns=['Model', 'Parameters', 'Best Model', 'Best Average Accuracy'])
best_score_df.loc["Baseline Model"] = [str(baseline_model), None, str(baseline_model), baseline_score]
print(tabulate(best_score_df[['Best Model', 'Best Average Accuracy']].sort_values(by='Best Average Accuracy', ascending=False), headers='keys', tablefmt='fancy_grid'))

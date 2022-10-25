# Auto Insurance Fraud Claim Detection
This Jupyter Notebook details the creation of 925 predictive models over five machine learning algorithms to predict whether an auto insurance claim is fraudulent. It then ranks the best performing model of each algorithm in order of their average accuracy and uses the optimal model to estimate the total dollar amount paid to fraudulent claims.

# Additional Note
You can attempt to define a model with a greater accuracy than the optimal model outlined in this notebook by altering the code in the Model Initialization section. Some alterations you may make include:
  * Changing the existing values of the existing parameter dictionaries
  * The addition of new parameters to be used in the tuning of existing models (Ex. adding min_samples_split to rf_params)
  * The addition of a new type of model (in which case you would need to define the baseline model and its associated parameters, then add these to the dicitonary     all_models

# Auto Insurance Fraud Claim Detection

## Introduction
With the vast amount of auto insurance claims filed everyday, an extensive amount of resources are required to validate these claims. Despite these efforts, fraudulent claims slip by uncaught all too often. Analyzing 1000 claims, this Jupyter Notebook outlines the process of creating a predictive model aimed to identify fraudulent auto insurance claims. Using hyper-parameter tuning, 925 predictive models are created over five different machine learning algorithms. The best overall performing model is then used to estimate the total dollar amount paid to fraudulent claims, which is then compared to the amount due to real fraudulent claims. This predictive model can help reduce the resources required to validate auto insurance claims by automatically highlighting potentially fraudulent claims, which would then be looked into further by a human.

## Additional Note
You can attempt to define a model with a greater accuracy than the optimal model outlined in this notebook by altering the code in the Model Initialization section. Some alterations you may make include:
  * Changing the existing values of the existing parameter dictionaries
  * The addition of new parameters to be used in the tuning of existing models (Ex. adding min_samples_split to rf_params)
  * The addition of a new type of model (in which case you would need to define the baseline model and its associated parameters, then add these to the dicitonary all_models

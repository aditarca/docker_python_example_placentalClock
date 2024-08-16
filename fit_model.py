import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Make sure the training data files provided are in the current directory

# Read annotation data for all training samples
meta = pd.read_csv("Sample_annotation.csv")
meta.set_index('Sample_ID', inplace=True)

# Read feature data for the first 100 features only and all samples (for speed reasons)
# Samples are columns and rows are features
X = pd.read_csv("Beta_raw_subchallenge1.csv", nrows=100, index_col=0)

# Transpose the feature data
X = X.T

# Extract the target gestational age and merge it with feature data
X['GA'] = meta.loc[X.index, 'GA']

# Fit a simple linear model that predicts GA using all 100 methylation features
# Prepare the input data
y = X.pop('GA')
model = LinearRegression().fit(X, y)

# Save the model needed for docker submission
with open("model_test_SC1.pkl", 'wb') as f:
    pickle.dump(model, f)


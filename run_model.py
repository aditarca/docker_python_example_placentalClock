import pandas as pd
import numpy as np
import argparse
import os
import pickle

# Set up the argument parser
parser = argparse.ArgumentParser(description="Predict gestational age using a pre-trained model.")
parser.add_argument("--input", type=str, default="/input", help="Input directory [default=/input]")
parser.add_argument("--output", type=str, default="/output", help="Output directory [default=/output]")
args = parser.parse_args()

# Read the test data
test_data = pd.read_csv(os.path.join(args.input, "Leaderboard_beta_subchallenge1.csv"))
test_data.set_index(test_data.columns[0], inplace=True)
Sample_IDs = test_data.columns

# Transpose the data
test_data = test_data.T

# Load the pre-trained model
with open("/usr/local/bin/model_test_SC1.pkl", 'rb') as f:
    model = pickle.load(f)
# Ensure test data columns are in the correct order
expected_columns = model.feature_names_in_
test_data = test_data[expected_columns]

# Make predictions
ga = model.predict(test_data)

# Ensure predictions are within the valid range
ga = np.clip(ga, 5, 44)

# Prepare output
output_df = pd.DataFrame({
    "ID": Sample_IDs,
    "GA_prediction": ga
})

# Save the predictions to a CSV file
output_df.to_csv(os.path.join(args.output, "predictions.csv"), index=False)


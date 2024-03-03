import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

data = pd.read_csv("Ratio_teste_ML.csv", delimiter=";")
print(data.head())

from sklearn.model_selection import train_test_split
x = np.array(data[["med_pixels"]])
y = np.array(data[["Score"]])
    
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                    test_size=0.33, 
                                                    random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

new_data = pd.read_csv("ratio_med.csv", delimiter=";")

# Extract features (assuming same column names as the original data)
new_x = np.array(new_data[["med_pixels"]])

# Make predictions using the trained model
predicted_scores = model.predict(new_x)

# Add predicted scores to the new data frame (optional)
new_data["Predicted_Score"] = predicted_scores

# Print the predicted scores
print(predicted_scores)
df = pd.DataFrame(predicted_scores)
print(df)
df.to_csv("my_data.csv", index=False)
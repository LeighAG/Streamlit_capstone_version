import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

# Title for the Streamlit app
st.title("Machine Learning NWSL Tool")

# Upload CSV file
matches = pd.read_csv("dataNWSL20232024.csv", index_col=0)

# Display dataset info
# st.subheader("Dataset Overview")
# st.write(matches.head())

# # Display basic statistics
# st.subheader("Dataset Description")
# st.write(matches.describe())

# # Check for missing data
# st.subheader("Missing Data")
# missing_data = matches.isnull().sum()
# missing_data = missing_data[missing_data > 0]
# st.write(missing_data)

# # Calculate Pearson correlation coefficient
# st.subheader("Pearson Correlation Coefficient")
# c = pearsonr(matches['A'], matches['KP'])
# st.write(c)

# Create categorical variables for position played
matches["Pos"] = matches["P"].astype("category").cat.codes

# Create categorical variables for team names
matches["Teams"] = matches["Team"].astype("category").cat.codes

# Create weighted aggregate goal contribution score for the dependent variable
matches['GCS'] = (((matches['G'] * 2) + matches['A'])).astype(int)

# Utilize interquartile range method to identify outliers in the dependent variable and create variable
Q1 = matches['GCS'].quantile(0.25)
Q3 = matches['GCS'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = matches['GCS'][(matches['GCS'] < lower_bound) | (matches['GCS'] > upper_bound)]

# Display the calculated columns and outliers
# st.subheader("Processed Data")
# st.write(matches.head())

# st.subheader("Outliers in GCS")
# st.write(outliers)

# Clean dataset of outliers and remove all rows for player position of goalie
matches_cleaned = matches[~matches.index.isin(outliers.index) & (matches['P'] != 'G') & (matches['Pos'] != 2)]

    # Display cleaned dataset
# st.subheader("Cleaned Dataset")
# st.write(matches_cleaned.head())

 # Create dataset for independent variables, dropping goals, assists, attacking assists and GCS
x = matches_cleaned.drop(columns=["G", "A", "KP", "GCS"])

    # Create dataset for GCS
y = matches_cleaned["GCS"]

    # Create training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=11, test_size=0.2)

    # Identify predictor variables
predictors = ["S", "SOT", "Tackles", "FC", "FS", "GP"]

# st.subheader("Training and Test Data Shapes")
# st.write("X_train shape:", x_train.shape)
# st.write("X_test shape:", x_test.shape)
# st.write("Y_train shape:", y_train.shape)
# st.write("Y_test shape:", y_test.shape)

# st.subheader("Predictor Variables")
# st.write(predictors)

#create random forest model
rf=RandomForestClassifier(
    n_estimators=150,         
    min_samples_split=6,     
    bootstrap=True,           
    oob_score=True,           
    random_state=1            
)

#fit the training data to the model
rf.fit(x_train[predictors].values, y_train)

#test the model with predictions using the test data of independent variables
preds = rf.predict(x_test[predictors].values)

# rsq = r2_score(y_test, preds)
# rsq

 # Get the OOB score
oob_score = rf.oob_score_

    # Calculate the OOB error score
oob_error = 1 - oob_score

    # Display OOB results
# st.subheader("Out-of-Bag (OOB) Score and Error")
# st.write(f"OOB Score: {oob_score}")
# st.write(f"OOB Error Score: {oob_error}")

# mse = mean_squared_error(y_test, preds)
# mse

text = "Predicting NWSL Player Goal Contribution Scores with Machine Learning"

# Display centered text with styling in Streamlit
st.markdown(f"""
    <div style="text-align: center; font-size: 28px; font-weight: bold;">
        {text}
    </div>
""", unsafe_allow_html=True)

# Display application description
st.write("""
This application is a predictive tool designed for evaluating and predicting a player's Goal Contribution Score (GCS) 
in the National Women’s Soccer League (NWSL). The GCS measures a player’s overall impact based on goals and assists. 
By entering key player statistics such as shots taken, tackles, and fouls, users can gain insights into a player’s 
projected performance within the league. 

Below, users can explore visualizations about the data used for developing this model and then use an interactive tool. 
This tool is perfect for NWSL fans, coaches, and analysts.
""")

# Kernel Density Estimate Plot
st.subheader("1. A histogram showing the distribution of the GCS in the data.")

plt.figure(figsize=(8, 6))
sns.histplot(matches_cleaned['GCS'], kde=True, bins=20, color='skyblue')

# Add titles and labels
plt.title('Distribution of Goal Contribution Score', fontsize=16)
plt.xlabel('Goal Contribution Score', fontsize=14)
plt.ylabel('Density', fontsize=14)

# Display the plot in Streamlit
st.pyplot(plt)

# Feature names corresponding to your predictors
feature_names = [
    'Shots Taken',
    'Shots On Target',
    'Tackles',
    'Fouls Committed',
    'Fouls Suffered',
    'Games Played'
]

# Ensure 'rf' (RandomForest model) is already fitted
feature_importances = rf.feature_importances_

# Display description
st.subheader("2. A pie chart illustrating the feature importance percentages in the model")

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(feature_importances, labels=feature_names, autopct='%1.1f%%', startangle=140)

# Add a title
plt.title('Feature Importance in the Predictor Model')

# Display the pie chart in Streamlit
st.pyplot(plt)

# Ensure 'rf' (RandomForest model) is already fitted and 'predictors' is defined
predicted_gcs = rf.predict(x_test[predictors].values)

# Display description
st.subheader("3. A scatter plot showing predicted and actual GCS values with the model regression line.")

# Create scatter plot with regression line
plt.figure(figsize=(10, 6))

plt.scatter(y_test, predicted_gcs, alpha=0.6, color='blue', edgecolor='k', label='Predicted vs Actual')

sns.regplot(x=y_test, y=predicted_gcs, scatter=False, color='red', label='Regression Line')

# Add plot labels and legend
plt.title('Actual vs Predicted GCS Scores', fontsize=16)
plt.xlabel('Actual GCS Scores', fontsize=14)
plt.ylabel('Predicted GCS Scores', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Display the plot in Streamlit
st.pyplot(plt)

# Display description
st.subheader("4. A scatter plot showing the proportion of shots on target over shots taken and GCS score. This was created as a visualization of interest because shots on target and shots taken have the highest percentage of feature predictability.")

# Create a copy of the cleaned dataset to avoid modifying the original
subset = matches_cleaned.copy()

# Calculate the proportion of shots on target
subset['On_Target_Proportion'] = subset['SOT'] / subset['S']

# Drop rows where scoring attempts ('S') are zero to avoid division by zero errors
subset = subset[subset['S'] > 0]

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(
    subset['On_Target_Proportion'],
    subset['GCS'],
    alpha=0.6,
    edgecolors='k'
)

# Add titles and labels
plt.title('Proportion of Shots on Target out of Shots Taken and GCS Score', fontsize=14)
plt.xlabel('Proportion of Shots on Target', fontsize=12)
plt.ylabel('Goal Contribution Score', fontsize=12)

# Display the plot in Streamlit
st.pyplot(plt)

#INTERACTIVE MODEL AND DISPLAY

# Define GCS statistics
mean = 2.13
std = 2.56
min_score = 0.00
q25 = 0.00
q50 = 1.00
q75 = 4.00
max_score = 10.00


# Function to interpret the GCS score
def interpret_score(score):
    interpretation = f"""
    **How to Interpret the Score**  

    - **Average Score (Mean):** 2.13 – If the score is near this value, the player’s contribution is average.  
    - **Standard Deviation:** 2.56 – Scores far from this range indicate exceptional or below-average performance.  
    - **Minimum Score:** 0.00 – Indicates no goals or assists.  
    - **25th Percentile (Q1):** 0.00 – 25% of players have this score or lower.  
    - **Median (Q2/50th Percentile):** 1.00 – Half of the players score below and half score above this point.  
    - **75th Percentile (Q3):** 4.00 – 75% of players have this score or lower, meaning 25% are higher achievers.  
    - **Maximum Score:** 10.00 – The highest observed score in the dataset.  
    """
    return interpretation



# Display instructions
st.write("Input data into the below boxes and press submit to find out the player score.")
st.write("For inspiration, go to: [NWSL Stats](https://www.nwslsoccer.com/stats/players). It is recommended to use completed NWSL seasons.")

# 
# Title of the App
st.title("⚽ Predicting NWSL Player Goal Contribution Score")

st.write("""
This application predicts a player's **Goal Contribution Score (GCS)** in the National Women's Soccer League (NWSL).
Enter the player's statistics below and click **Predict GCS** to see the projected performance.
""")

# Collecting user input in the main page
st.header("📊 Input Player Statistics")

shots_taken = st.number_input("Shots Taken", min_value=0, step=1, value=0)
shots_on_target = st.number_input("Shots On Target", min_value=0, step=1, value=0)
tackles = st.number_input("Tackles", min_value=0, step=1, value=0)
fouls_committed = st.number_input("Fouls Committed", min_value=0, step=1, value=0)
fouls_suffered = st.number_input("Fouls Suffered", min_value=0, step=1, value=0)
games_played = st.number_input("Games Played", min_value=0, step=1, value=0)

# Prediction function
def predict_gcs():
    # Collect user inputs in a DataFrame (match model input structure)
    input_data = pd.DataFrame([[shots_taken, shots_on_target, tackles, fouls_committed, fouls_suffered, games_played]],
                              columns=["S", "SOT", "Tackles", "FC", "FS", "GP"])
    # Predict using the trained RandomForest model
    prediction = rf.predict(input_data)[0]
    return prediction

# Submit button and output
if st.button("🚀 Predict GCS"):
    predicted_score = predict_gcs()
    
    # Display predicted GCS
    st.subheader(f"🏆 Predicted Goal Contribution Score: {predicted_score}")

    # Interpretation of the prediction
    if predicted_score < 1:
        st.write("🔵 **Low Contribution:** This player contributes minimally to goals and assists.")
    elif predicted_score < 4:
        st.write("🟡 **Average Contribution:** This player has an average contribution compared to others.")
    else:
        st.write("🔥 **High Contribution:** This player significantly impacts the team's goal-scoring efforts!")
    
    st.write(interpret_score(predicted_score))
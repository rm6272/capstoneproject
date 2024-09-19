#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:11:52 2024

@author: ruim
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

random.seed(a="N11034601")

df = pd.read_csv("/Users/ruim/Downloads/spotify52kData.csv")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Set the aesthetic style of the plots
sns.set(style='whitegrid')

# Create a figure with a 2x5 grid of subplots
plt.figure(figsize=(20, 10))  # Adjust the size as needed

# List of your features
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Create a histogram for each feature
for i, feature in enumerate(features):
    plt.subplot(2, 5, i+1)  # Creates subplot in a 2x5 grid
    sns.histplot(df[feature], kde=False, color='blue')  # kde=False turns off the Kernel Density Estimate
    plt.title(feature.capitalize())  # Capitalize the title for better presentation
    plt.xlabel('')
    plt.ylabel('')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# Set the aesthetic style of the plots
sns.set(style='whitegrid')

# Create the scatter plot
plt.figure(figsize=(10, 6))  # Set the figure size
sns.scatterplot(data=df, x='duration', y='popularity', alpha=0.5, edgecolor=None, s=15)

# Adding titles and labels
plt.title('Relationship Between Song Duration and Popularity')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity')

# Show the plot
plt.show()

correlation = df['duration'].corr(df['popularity'])
spearman_corr = df['duration'].corr(df['popularity'], method='spearman')

print(f'Correlation coefficient: {correlation:.2f}')
print(f' spearman Correlation coefficient: {spearman_corr:.2f}')


df_true = df[df['explicit'] == True]

# DataFrame where explicit is False
df_false = df[df['explicit'] == False]

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Histogram for the True subset
sns.histplot(df_true['popularity'], kde=True, color='blue', ax=axs[0])  
axs[0].set_title('Histogram of Popularity for Explicit Songs')
axs[0].set_xlabel('Popularity')
axs[0].set_ylabel('Frequency')

# Histogram for the False subset
sns.histplot(df_false['popularity'], kde=True, color='red', ax=axs[1]) 
axs[1].set_title('Histogram of Popularity for Non-Explicit Songs')
axs[1].set_xlabel('Popularity')
axs[1].set_ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

from scipy.stats import mannwhitneyu

stat, p_value = mannwhitneyu(df_true['popularity'], df_false['popularity'], alternative='greater')
print('Statistics=%.3f, p-value=%.10f' % (stat, p_value))


median_explicit = df_true['popularity'].median()
median_nonexplicit = df_false['popularity'].median()

print(f"Median Popularity of explicit: {median_explicit}")
print(f"Median Popularity of non-explicit: {median_nonexplicit}")

if median_explicit > median_nonexplicit:
    print("Group explicit is more popular on average than Group non-explicit.")
elif median_explicit < median_nonexplicit:
    print("Group nonexplicit is more popular on average than Group explicit.")
else:
    print("Both groups have equal popularity on average, despite significant differences in distribution.")


#4
df_minor = df[df['mode'] == 0]
df_major = df[df['mode'] == 1]

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Histogram for the True subset for visualization of the distribution
sns.histplot(df_minor['popularity'], kde=True, color='blue', ax=axs[0]) 
axs[0].set_title('Histogram of Popularity for minor Subset')
axs[0].set_xlabel('Popularity')
axs[0].set_ylabel('Frequency')

# Histogram for the False subset
sns.histplot(df_major['popularity'], kde=True, color='red', ax=axs[1]) 
axs[1].set_title('Histogram of Popularity for major Subset')
axs[1].set_xlabel('Popularity')
axs[1].set_ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

stat, p_value = mannwhitneyu(df_major['popularity'], df_minor['popularity'], alternative='greater')
print('Statistics=%.3f, p-value=%.10f' % (stat, p_value))

median_major = df_major['popularity'].median()
median_minor = df_minor['popularity'].median()

print(f"Median Popularity of major: {median_major}")
print(f"Median Popularity of minor: {median_minor}")

if median_major > median_minor:
    print("major is more popular on average than minor.")
elif median_major < median_minor:
    print("minor is more popular on average than major.")
else:
    print("Both groups have equal popularity on average, despite significant differences in distribution.")

correlation = df['loudness'].corr(df['energy'])
correlation_sp = df['loudness'].corr(df['energy'], method='spearman')

print(f'The Pearson correlation coefficient between loudness and energy is: {correlation:.2f}')
print(f'The Spearman correlation coefficient between loudness and energy is: {correlation_sp:.2f}')


sns.scatterplot(data=df, x='loudness', y='energy', s = 10)
plt.title('Relationship Between Loudness and Energy of Songs')
plt.xlabel('Loudness (dB)')
plt.ylabel('Energy')
plt.show()

import pandas as pd

# Assuming you have loaded your DataFrame as df
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Calculate correlation with popularity
correlations = df[features + ['popularity']].corr()['popularity'].sort_values()

# Display the correlations
print(correlations)

from sklearn.metrics import r2_score

# Function to perform regression and calculate R^2 score
def regress_and_score(feature):
    X = df[['instrumentalness']]  # Feature matrix
    y = df['popularity']  # Target variable

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate R^2 score
    score = r2_score(y_test, y_pred)
    return score

# Evaluate each feature
scores = {feature: regress_and_score(feature) for feature in features}

# Display the scores
print(scores)

#5 correlation linear... 
# ols, 
#ksiaiser 3, elbow 1, 3 PCA, based on each of the three graphs and choose. 
 
#7, train test split, 

from sklearn.metrics import mean_squared_error, r2_score

# Features you've decided to use
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Target variable
target = 'popularity'

# Check for any remaining missing values and drop them (or impute, depending on your choice)
df.dropna(subset=features + [target], inplace=True)

# Define the feature matrix (X) and the target vector (y)
X = df[features]
y = df[target]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("multiple linear regression: ")
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Optionally, display regression coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Q8 PCA
from sklearn.preprocessing import StandardScaler

X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA
import numpy as np

# Initialize PCA
pca = PCA()

# Fit PCA to the scaled data
pca.fit(X_scaled)



# Display explained variance ratio for each component
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Typically, you might choose the number of components that sum to 0.85 or more of total variance
explained_variance = pca.explained_variance_ratio_

# Loadings (contribution of each feature to each component)
loadings = pca.components_

explained_variance_first_component = pca.explained_variance_ratio_[0]

# Print the explained variance for the first principal component
print(f"Explained variance for the first principal component: {explained_variance_first_component:.4f}")

eigenvalues = pca.explained_variance_

# Creating a bar graph of the eigenvalues
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, color='b', alpha=0.6)
plt.axhline(y=1, color='r', linestyle='--')  # Adding a horizontal line at y=1
plt.title('Scree Plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.xticks(range(1, len(eigenvalues) + 1))
plt.grid(True)
plt.show()

cumulative_variance = np.cumsum(explained_variance)

# Threshold for explained variance
threshold = 90

# Number of components needed to reach the threshold
num_components = np.where(cumulative_variance >= threshold/100)[0][0] + 1

# Printing the results
print('Number of factors to account for at least 90% variance:', num_components)

explained_variance_second_component = pca.explained_variance_ratio_[1]
explained_variance_third_component = pca.explained_variance_ratio_[2]

print("explained 3 components: ", explained_variance_second_component+explained_variance_first_component+explained_variance_third_component)

cumulative_explained_variance = np.cumsum(explained_variance)
sum_explained_variance_7 = cumulative_explained_variance[6] if len(explained_variance) >= 7 else "Not enough components"
print(f'Sum of explained variance up to 7 components:', sum_explained_variance_7)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Prepare data for logistic regression
# Assuming 'mode' as 1 for major and 0 for minor and 'valence' is already in the dataset
X = df[['valence']]  # Predictor
y = df['mode']  # Target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the logistic regression model:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Display the logistic regression model coefficients and intercept
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
    

pc_index = 0  # Indexing starts at 0 for the first component

# Plot the loadings for the selected principal component
plt.figure(figsize=(12, 6))
plt.bar(features, loadings[pc_index]*-1, color='skyblue')
plt.xlabel('Features')
plt.ylabel('Loading Value')
plt.title(f'Loadings on Principal Component {pc_index + 1}')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(features, loadings[1]*-1, color='skyblue')
plt.xlabel('Features')
plt.ylabel('Loading Value')
plt.title(f'Loadings on Principal Component 2')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
plt.figure(figsize=(12, 6))
plt.bar(features, loadings[2]*-1, color='skyblue')
plt.xlabel('Features')
plt.ylabel('Loading Value')
plt.title(f'Loadings on Principal Component 3')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
    
#9
from sklearn.metrics import roc_auc_score

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove non-predictor columns if necessary, e.g., ID columns or the target column itself
numerical_features.remove('mode')  # adjust as necessary if 'mode' is included in the numerical list

# Dictionary to hold each feature's performance
feature_performance = {}

# Analyze each numerical feature
for feature in numerical_features:
    # Prepare the data
    X = df[[feature]].dropna()
    y = df.loc[X.index, 'mode']  # Ensure y is aligned with X after dropna

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict on the testing set
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # get probabilities for the positive class

    # Calculate AUC-ROC as the performance metric
    auc_score = roc_auc_score(y_test, y_pred_prob)
    feature_performance[feature] = auc_score

# Print the performance of each feature
for feature, auc in feature_performance.items():
    print(f"{feature}: AUC-ROC = {auc:.3f}")
    

X = df[['speechiness']].dropna()  # Select and drop any missing values
y = df.loc[X.index, 'mode']  # Align target variable with cleaned predictor

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Generate a sequence of speechiness values for prediction
speechiness_range = np.linspace(X['speechiness'].min(), X['speechiness'].max(), 300)

# Predict probabilities for these values
probabilities = model.predict_proba(speechiness_range.reshape(-1, 1))[:, 1]

# Plotting the logistic regression results
plt.figure(figsize=(10, 6))
plt.scatter(X_train['speechiness'], y_train, color='black', label='Training data', alpha=0.7)
plt.scatter(X_test['speechiness'], y_test, color='red', label='Test data', alpha=0.5)
plt.plot(speechiness_range, probabilities, color='blue', label='Logistic Regression Curve')
plt.title('Logistic Regression on Speechiness Predicting Mode')
plt.xlabel('Speechiness')
plt.ylabel('Probability of Major Mode')
plt.legend()
plt.grid(True)
plt.show()

#10 Which is a better predictor of whether a song is classical music 

df['is_classical'] = (df['track_genre'] == 'classical').astype(int)
X_duration = df[['duration']]
y = df['is_classical']
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)  # This is the correct PCA scores array

# Split data for the duration-based model
X_train_dur, X_test_dur, y_train_dur, y_test_dur = train_test_split(df[['duration']], y, test_size=0.2, random_state=42)

# Split data for the PCA-based model
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model_dur = LogisticRegression()
model_dur.fit(X_train_dur, y_train_dur)
y_pred_dur = model_dur.predict(X_test_dur)
print("Duration-based Model Accuracy:", accuracy_score(y_test_dur, y_pred_dur))
print("Classification Report for Duration Model:\n", classification_report(y_test_dur, y_pred_dur))
y_pred_prob_dur = model_dur.predict_proba(X_test_dur)[:, 1]  # Probabilities for the positive class
auc_score_dur = roc_auc_score(y_test_dur, y_pred_prob_dur)
print("AUC-ROC for Duration Model:", auc_score_dur)

# Train Logistic Regression model with PCA components
model_pca = LogisticRegression()
model_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = model_pca.predict(X_test_pca)
print("PCA-based Model Accuracy:", accuracy_score(y_test_pca, y_pred_pca))
print("Classification Report for PCA Model:\n", classification_report(y_test_pca, y_pred_pca))
y_pred_prob_pca = model_pca.predict_proba(X_test_pca)[:, 1]  # Probabilities for the positive class
auc_score_pca = roc_auc_score(y_test_pca, y_pred_prob_pca)
print("AUC-ROC for PCA Model:", auc_score_pca)

#Extra credit
# Analyze the distribution of beats per measure
beats_per_measure_distribution = df['time_signature'].value_counts().sort_index()

# Plot the distribution of beats per measure
plt.figure(figsize=(12, 6))
sns.barplot(x=beats_per_measure_distribution.index, y=beats_per_measure_distribution.values)
plt.title('Distribution of Beats per Measure')
plt.xlabel('Beats per Measure')
plt.ylabel('Number of Songs')
plt.show()

# Identify the most common time signature
most_common_time_signature = beats_per_measure_distribution.idxmax()
print(f'The most common time signature in the dataset is: {most_common_time_signature}')
X = df[['time_signature']]
y = df['tempo']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional, not necessary for time signature as it is categorical)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-Squared: {r2}')





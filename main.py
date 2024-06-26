from flask import Flask, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tpot import TPOTRegressor

app = Flask(__name__, template_folder='templates/html/')

def preprocess_data(df):
    df.dropna(inplace=True)
    le = LabelEncoder()
    categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
    df[categorical_cols] = df[categorical_cols].apply(le.fit_transform)
    scaler = StandardScaler()
    numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def calculate_inertia(data, max_k=10):
    inertia_values = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)
    return inertia_values

def get_correlation_heatmap(df):
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.index,
                    y=corr_matrix.columns,
                    color_continuous_scale='sunset')
    return fig.to_json()

def get_clustered_data(df, numerical_cols):
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[numerical_cols])
    fig = px.scatter(df, x='area', y='price', color='cluster', title='Clustered Data')
    return fig.to_json()

def get_elbow_plot(df, numerical_cols):
    inertia_values = calculate_inertia(df[numerical_cols])
    k_values = np.arange(1, len(inertia_values) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_values, y=inertia_values, mode='lines+markers'))
    fig.update_layout(title='Elbow Method for Optimal K',
                      xaxis_title='Number of Clusters (K)',
                      yaxis_title='Inertia')
    return fig.to_json()

# Function to train Gaussian Naive Bayes regressor, make predictions, and plot results
def train_and_plot_model(df):
    # Split the data into features and target variable
    X = df.drop(['price'], axis=1)
    y = df['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a Gaussian Naive Bayes regressor
    gnb = GaussianNB()
    gnb.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = gnb.predict(X_test_scaled)

    # Evaluate the model using RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Plot actual vs predicted prices using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted Prices'))
    fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Actual Prices', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Actual Prices vs Predicted Prices',
                      xaxis_title='Actual Prices',
                      yaxis_title='Predicted Prices')
    plot_json = fig.to_json()

    return rmse, plot_json


# Function to train Random Forest regressor, make predictions, and plot results
def rtrain_and_plot_model(df):
    # Split the data into features and target variable
    X = df.drop(['price'], axis=1)
    y = df['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest regressor
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_regressor.predict(X_test)

    # Evaluate the model using RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Get feature importances from the trained model
    importances = rf_regressor.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importances
    fig_importance = go.Figure()
    fig_importance.add_trace(go.Bar(x=importance_df['Feature'], y=importance_df['Importance'],
                                    marker_color='skyblue'))
    fig_importance.update_layout(title='Feature Importance',
                                 xaxis_title='Features',
                                 yaxis_title='Importance')

    # Plot actual vs predicted prices using Plotly
    fig_predictions = go.Figure()
    fig_predictions.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted Prices'))
    fig_predictions.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Actual Prices',
                                         line=dict(color='red', dash='dash')))
    fig_predictions.update_layout(title='Actual Prices vs Predicted Prices',
                                  xaxis_title='Actual Prices',
                                  yaxis_title='Predicted Prices')

    return rmse, fig_predictions.to_json(), fig_importance.to_json()

def automate_model(df):
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a TPOT AutoML instance
    tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
    
    # Fit TPOT on the training data
    tpot.fit(X_train, y_train)
    
    # Evaluate the model on the test data
    y_pred = tpot.predict(X_test)
    
    # Evaluate the model using RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return rmse

@app.route('/')
def index():
    df = pd.read_csv("Housing.csv")
    df = preprocess_data(df)

    numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

    heatmap_json = get_correlation_heatmap(df)
    clustered_data_json = get_clustered_data(df, numerical_cols)
    elbow_plot_json = get_elbow_plot(df, numerical_cols)
    rmse, naive_bayes_plot_json = train_and_plot_model(df)
    rmsee, a, importance_plot_json = rtrain_and_plot_model(df)

    # Automate model training and evaluation
    rmse_automl = automate_model(df)

    return render_template('index.html', heatmap_json=heatmap_json, clustered_data_json=clustered_data_json,
                           elbow_plot_json=elbow_plot_json, naive_bayes_plot_json=naive_bayes_plot_json,rmse=rmse,
                           importance_plot_json=importance_plot_json, rmse_automl=rmse_automl)

if __name__ == '__main__':
    app.run(debug=True)
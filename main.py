from flask import Flask, render_template, jsonify, request
import pandas as pd
import ast 
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__, template_folder='templates/html/')

def calculate_inertia(data, max_k=10):
    inertia_values = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)
    return inertia_values

@app.route('/')
def index():
    df = pd.read_csv("Housing.csv")            
    # Data preprocessing
    df.dropna(inplace=True)
    # Create a label encoder object
    le = LabelEncoder()

    # Apply label encoding for each categorical column
    df['mainroad'] = le.fit_transform(df['mainroad'])
    df['guestroom'] = le.fit_transform(df['guestroom'])
    df['basement'] = le.fit_transform(df['basement'])
    df['hotwaterheating'] = le.fit_transform(df['hotwaterheating'])
    df['airconditioning'] = le.fit_transform(df['airconditioning'])
    df['prefarea'] = le.fit_transform(df['prefarea'])
    df['furnishingstatus'] = le.fit_transform(df['furnishingstatus'])
    # Calculate correlation matrix

    scaler = StandardScaler()
    numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    corr_matrix = df.corr()

    # Create a heatmap using Plotly Express
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.index,
                    y=corr_matrix.columns,
                    color_continuous_scale='sunset')

    # Convert the Plotly figure to JSON format
    heatmap_json = fig.to_json()


    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[numerical_cols])

    # Visualize clustered data
    fig = px.scatter(df, x='area', y='price', color='cluster', title='Clustered Data')
    clustered_data_json = fig.to_json()


    #Calculate inertia values for different K values
    inertia_values = calculate_inertia(df[numerical_cols])

    # Plot elbow curve
    k_values = np.arange(1, len(inertia_values) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_values, y=inertia_values, mode='lines+markers'))
    fig.update_layout(title='Elbow Method for Optimal K',
                      xaxis_title='Number of Clusters (K)',
                      yaxis_title='Inertia')
    elbow_plot_json = fig.to_json()
    
    return render_template('index.html', heatmap_json=heatmap_json, clustered_data_json=clustered_data_json,
                           elbow_plot_json=elbow_plot_json)

if __name__ == '__main__':
    app.run(debug=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
import os

# Flask is a web framework for Python used for building web applications
# It allows us to create a web interface where users can interact with machine learning models
app = Flask(__name__, template_folder="templates_97_CSM_241016")
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

uploaded_data = {}  # Dictionary to store processed data globally (basic setup)


# Function to generate target values with Gaussian noise to remove systematic bias
# This ensures real-world variability in the dataset
# The noise level is scaled based on dataset size to simulate real-life data variations
def apply_standard_deviation(x, size):
    std_dev = 0.3 * (size / 1000)  # Increasing noise for larger datasets to remove bias
    return np.sin(2 * np.pi * x) + np.random.normal(0, std_dev, len(x))

# Function to preprocess the dataset before applying ML models
# 1. Standardizes the input values using StandardScaler to remove scaling bias
# 2. Splits the dataset into 80% training and 20% testing to ensure fair evaluation
def preprocess_data(x, y):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.reshape(-1, 1))
    return train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Function to determine the best model for the given dataset
def select_best_model(x_train, x_test, y_train, y_test, size):
    models = {
        "Linear Regression": LinearRegression(),
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "K-Means Clustering": KMeans(n_clusters=2, random_state=42),
        "SVM": svm.SVC()
    }
    
    best_model = None
    best_score = float('-inf')
    best_model_name = ""

    for name, model in models.items():
        if name == "Linear Regression":
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            score = -mean_squared_error(y_test, y_pred)  # Lower MSE is better, negate for max
        
        elif name in ["Logistic Regression", "Decision Tree", "SVM"]:
            threshold = np.percentile(y_train, 50)  # Median split for balanced classes
            y_train_cls = (y_train > threshold).astype(int)
            y_test_cls = (y_test > threshold).astype(int)
            model.fit(x_train, y_train_cls)
            y_pred = model.predict(x_test)
            score = accuracy_score(y_test_cls, y_pred)
        
        elif name == "K-Means Clustering":
            model.fit(x_train)
            y_pred = model.predict(x_test)
            score = silhouette_score(x_test, y_pred)  # Measures clustering quality
        
        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name
    
    best_model.fit(x_train, y_train if best_model_name == "Linear Regression" else (y_train > np.percentile(y_train, 50)).astype(int))
    y_pred = best_model.predict(x_test)
    
    # Save scatter plot of actual vs predicted values with user-defined name
    filename = f"static/32_Samaita_{size}.png"
    os.makedirs("static", exist_ok=True)
    plt.scatter(x_test, y_test, label="Actual", color='blue' if size < 10000 else 'green')
    plt.scatter(x_test, y_pred, label="Predicted", color='red' if size < 10000 else 'orange')
    plt.title(f"Best Model: {best_model_name} - Dataset Size {size}")
    plt.legend()
    plt.savefig(filename)
    plt.clf()
    
    return best_model_name, filename

# Route in Flask: Defines the webpage URL and behavior when accessed
# Here the Flask route decorator defines the URL endpoint that is the homepage of the web application
# Also specifies that when a user visits the root URL (/) the function below it (index()) will be executed 
# GET is used when a user first visits the webpage a GET request is sent which typically loads the page
# POST is used When a user submits a form a POST request is sent allowing the server to process data
# ------------------- Flask Routes -------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['dataset']
    if file:
        filename = file.filename
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Create folder if it doesn't exist
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)  # Save the file

        df = pd.read_csv(file_path)  # Now read from disk

        if df.shape[1] < 2:
            return render_template('index.html', error="Dataset must have at least two columns.")

        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        size = len(x)

        y_noisy = apply_standard_deviation(x, size)
        x_train, x_test, y_train, y_test = preprocess_data(x, y_noisy)

        uploaded_data['x_train'] = x_train
        uploaded_data['x_test'] = x_test
        uploaded_data['y_train'] = y_train
        uploaded_data['y_test'] = y_test
        uploaded_data['size'] = size

        return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train():
    if not uploaded_data:
        return render_template('index.html', error="Please upload a dataset first.")

    x_train = uploaded_data['x_train']
    x_test = uploaded_data['x_test']
    y_train = uploaded_data['y_train']
    y_test = uploaded_data['y_test']
    size = uploaded_data['size']

    best_model_name, filename = select_best_model(x_train, x_test, y_train, y_test, size)

    uploaded_data['last_plot'] = filename  # Save for visualize
    uploaded_data['last_model'] = best_model_name

    return render_template('index.html')

@app.route('/visualize', methods=['GET'])
def visualize():
    if 'last_plot' not in uploaded_data:
        return render_template('index.html', error="Train the model first to visualize results.")
    
    output = [(uploaded_data['size'], uploaded_data['last_model'], uploaded_data['last_plot'])]
    return render_template('index.html', output=output)

# -------------------

if __name__ == '__main__':
    app.run(debug=True)

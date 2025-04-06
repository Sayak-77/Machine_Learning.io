import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets, tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC


class SyntheticDatasetGenerator:
    def __init__(self, n, noise_std, learning_rate, iterations):
        self.n = n  # Number of data points = 10
        self.noise_std = noise_std  # 0.03
        self.learning_rate = learning_rate #0.05  
        self.iterations = iterations  #50
        self.x_values = None
        self.y_values = None
        self.synthetic_dataset = None  # Linear Regression (regression dataset)
        self.synthetic_dataset_logistic = None # Logistic Regression (classification dataset)
        self.y_binary = None
        self.train_dataset = None
        self.test_dataset = None
    
    def generate_dataset_Linear(self):
        self.x_values = np.random.uniform(0, 1, self.n)
        true_y_values = np.sin(2 * np.pi * self.x_values) # using sin(2πx)
        noise = np.random.normal(0, self.noise_std, self.n) # adding noise
        self.y_values = true_y_values + noise

        self.synthetic_dataset = pd.DataFrame({'x': self.x_values, 'y': self.y_values}) # Linear Regression dataset creation
    
    def generate_dataset_Logistic(self):
        self.x_values = np.random.uniform(0, 1, self.n)
        true_y_values = np.sin(2 * np.pi * self.x_values) # using sin(2πx)
        noise = np.random.normal(0, self.noise_std, self.n) # adding noise
        self.y_values = true_y_values + noise
         
        self.y_binary = (self.y_values > 0).astype(int) # Converting into Binary: 1 (if y>0) else 0 for Logistic Regression
        self.synthetic_dataset_logistic = pd.DataFrame({'x': self.x_values, 'y': self.y_binary}) # Logistic Regression dataset creation

    def save_dataset_Linear(self, filename_linear):
        self.synthetic_dataset.to_csv(filename_linear, index=False)
        print(f"Dataset saved to {filename_linear}")
    
    def save_dataset_Logistic(self, filename_logistic):
        self.synthetic_dataset_logistic.to_csv(filename_logistic, index=False)
        print(f"Dataset saved to {filename_logistic}")
    
    def display_dataset_Linear(self):
        print("\nSynthetic Dataset Linear Regression:")
        print(self.synthetic_dataset)
    
    def display_dataset_Logistic(self):
        print("\nSynthetic Dataset Logistic Regression:")
        print(self.synthetic_dataset_logistic)
    
    def split_dataset(self, train_frac, random_state=42):
        # Split the dataset into training and test sets
        self.train_dataset = self.synthetic_dataset.sample(frac=train_frac, random_state=random_state)
        self.test_dataset = self.synthetic_dataset.drop(self.train_dataset.index)
        
        #Saving
        train_output_path = 'training_set.csv'
        test_output_path = 'test_set.csv'
        self.train_dataset.to_csv(train_output_path, index=False)
        self.test_dataset.to_csv(test_output_path, index=False)
        
        # Displaying 
        print("\nTraining Dataset:")
        print(self.train_dataset)
        print("\nTest Dataset:")
        print(self.test_dataset)
    
    def fit_curve(self, x, y, learning_rate, iterations):
        # Initialize w and b randomly for each curve
        w = np.random.randn()
        b = np.random.randn()
        errors = []

        # Gradient descent loop
        for _ in range(iterations):
            y_pred = w * np.sin(2 * np.pi * x) + b
            error = y_pred - y
            errors.append(np.mean(np.abs(error)))
        
            # Calculate gradients
            w_grad = np.sum(2 * error * np.sin(2 * np.pi * x)) / len(x)
            b_grad = np.sum(2 * error) / len(x)

            # Update w and b
            w -= learning_rate * w_grad
            b -= learning_rate * b_grad
        
        mean_error = np.mean(errors)
        return w, b, mean_error
    
    def fit_multiple_curves_training_set(self, num_curves):
        # Fit multiple curves and store their parameters
        curves = []
        mean_errors = []
        x_train = self.train_dataset['x'].values
        y_train = self.train_dataset['y'].values
        
        for _ in range(num_curves):
            w, b, mean_error = self.fit_curve(x_train, y_train, self.learning_rate, self.iterations)
            curves.append((w, b))
            mean_errors.append(mean_error)
        return curves, mean_errors
    
    def fit_multiple_curves_test_set(self, num_curves):
        # Fit multiple curves and store their parameters
        curves = []
        mean_errors = []
        x_test = self.test_dataset['x'].values
        y_test = self.test_dataset['y'].values
        
        for _ in range(num_curves):
            w, b, mean_error = self.fit_curve(x_test, y_test, self.learning_rate, self.iterations)
            curves.append((w, b))
            mean_errors.append(mean_error)
        return curves, mean_errors

    def plot_curves(self, curves): # Linear Regression
        # Plot the synthetic data points and fitted curves
        plt.scatter(self.x_values, self.y_values, color='blue', label='Synthetic Data')

        # Plot each fitted curve
        x_curve = np.linspace(0, 1, 100)
        for i, (w, b) in enumerate(curves):
            y_curve = w * np.sin(2 * np.pi * x_curve) + b
            plt.plot(x_curve, y_curve, label=f'Curve {i+1}')
        
        # Customize the plot
        plt.title("Synthetic Data and Fitted Curves")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    def train_logistic_regression(self):
        X = self.synthetic_dataset_logistic[['x']].values 
        y = self.synthetic_dataset_logistic['y'].values 

        # Using Leave-One-Out Cross-Validation (LOO-CV) due to small dataset
        loo = LeaveOneOut()
        accuracies = []

        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))

        avg_accuracy = np.mean(accuracies)
        print("\nMean Accuracy for the Logistic Model")
        print(f"Leave-One-Out Cross-Validation Accuracy: {avg_accuracy:.3f}")

        # Train final model on entire dataset
        self.model = LogisticRegression()
        self.model.fit(X, y)

    def plot_logistic_decision_boundary(self):
        X = np.linspace(0, 1, 100).reshape(-1, 1)
        probs = self.model.predict_proba(X)[:, 1]  # Probability of class 1

        plt.scatter(self.synthetic_dataset_logistic['x'], self.synthetic_dataset_logistic['y'], c=self.synthetic_dataset_logistic['y'], cmap='coolwarm', edgecolors='k')
        plt.plot(X, probs, label="Logistic Regression Curve", color='black')
        plt.xlabel('x')
        plt.ylabel('P(y=1)')
        plt.title('Logistic Regression Decision Boundary')
        plt.legend()
        plt.show()  
    
    def decision_tree_classification(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1234) #training the model

        clf = DecisionTreeClassifier(random_state=1234)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print(f'Mean Squared Error (MSE): {mse:.4f}')
        print(f'Accuracy Score: {accuracy:.4f}')

        # Plot the decision tree
        plt.figure(figsize=(9, 7.5))
        tree.plot_tree(clf,
                feature_names=iris.feature_names,
                class_names=iris.target_names,
                filled=True)
        plt.show()

    def kmeans_clustering(self, n_clusters=3):
        iris = datasets.load_iris()
        X = iris.data  

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_kmeans = kmeans.fit_predict(X)

        sse = kmeans.inertia_  # Inertia is the SSE
        mse = sse / len(X)

        # Reduce dimensionality to 2D using PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Plot clusters
        plt.figure(figsize=(8, 6))
        for i in range(n_clusters):
            plt.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1], label=f'Cluster {i}')

        # Plot cluster centroids
        centroids_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                    s=300, c='red', marker='X', edgecolors='black', label='Centroids')

        plt.title('K-Means Clustering on Iris Dataset')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

        print(f"Sum of Squared Errors (SSE): {sse:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")

    def support_vector_machines(self, step_size):
        iris = datasets.load_iris()
        X = iris.data[:, :2]  # Only first two features for visualization
        y = iris.target

        # Splitting data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training the SVM Linearly
        clf_linear = SVC(kernel='linear')
        clf_linear.fit(X_train, y_train)

        # Predict on test set
        y_pred = clf_linear.predict(X_test)

        # Calculate Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        # Creating mesh grid for decision boundary visualization
        h = step_size  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        Z_linear = clf_linear.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_linear = Z_linear.reshape(xx.shape)

        # Plotting the Hyperplane and decision boundary
        plt.contourf(xx, yy, Z_linear, cmap=plt.cm.Paired, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
        plt.title('Linear SVM')
        plt.xlabel('Sepal Length')
        plt.ylabel('Sepal Width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()
    

dataset_generator = SyntheticDatasetGenerator(n=10, noise_std=0.3, learning_rate=0.05, iterations=50)

inp =  input("\nChoose your Machine Learning Algorithm\ni.Enter 1 for Linear Regression\nii.Enter 2 for Logistic Regression\niii.Enter 3 for Decision Tree Model\niv.Enter 4 for K-Means Clustering\nv.Enter 5 for Support Vector Machines\n")

match inp:
    case "1":
        n = False
        while not n:
            user_input = input("\ni.Enter 1 for Synthetic Dataset\nii.Enter 2 for Binary Dataset\n")
            if user_input.lower() == '1':
                dataset_generator.generate_dataset_Linear()
                dataset_generator.save_dataset_Linear('synthetic_dataset_linear.csv')
                dataset_generator.display_dataset_Linear()
                # #Train and Test Dataset
                dataset_generator.split_dataset(train_frac=0.8)

                #Gradient Descent
                curves_training_set, error_training_set = dataset_generator.fit_multiple_curves_training_set(num_curves=8)
                curves_test_set, error_test_set = dataset_generator.fit_multiple_curves_test_set(num_curves=2)
                print("\nFitted Parameters for Training Dataset:")
                for i, (w, b) in enumerate(curves_training_set):
                    print(f"Curve {i+1}: w = {w:.4f}, b = {b:.4f}")

                print("\nMean Errors for Interactions for Training Dataset")
                for i, error in enumerate(error_training_set):
                        print(f"Mean error for Curve {i+1}: {error:.4f}")

                print("\nFitted Parameters for Test Dataset:")
                for i, (w, b) in enumerate(curves_test_set):
                    print(f"Curve {i+1}: w = {w:.4f}, b = {b:.4f}")

                print("\nMean Errors for Interactions for Test Dataset")
                for i, error in enumerate(error_test_set): 
                        print(f"Mean error for Curve {i+1}: {error:.4f}")

                dataset_generator.plot_curves(curves_training_set)
                dataset_generator.plot_curves(curves_test_set)
                n = True
            else:
                print("\nWrong Dataset..!!")
    case "2":
        n = False
        while not n:
            user_input = input("\ni.Enter 1 for Synthetic Dataset\nii.Enter 2 for Binary Dataset\n")
            if user_input.lower() == '2':
                dataset_generator.generate_dataset_Logistic()
                dataset_generator.save_dataset_Logistic('synthetic_dataset_logistic.csv')
                dataset_generator.display_dataset_Logistic()
                dataset_generator.train_logistic_regression()   #Training Model Logistic Regression
                dataset_generator.plot_logistic_decision_boundary()   #Plotting
                n = True
            else:
                print("\nWrong Dataset..!!")
    case "3":
        print("\nMean Error & Accuracy for Decision Tree Model(using IRIS Dataset)")
        dataset_generator.decision_tree_classification()   #Decision Tree Training & Plotting
    case "4":
        print("\nK-Means Clustering & Plotting the 3 Clusters with their Centroids(using IRIS Dataset)")
        dataset_generator.kmeans_clustering(n_clusters=3)   #K-Means Clustering Training & Plotting
    case "5":
        print("\nMean Errors & Accuracy for Support Vector Machines(using IRIS Dataset)")
        dataset_generator.support_vector_machines(step_size=0.02)
    case _:
        print("\nWrong Input.Please Choose Again..!!")






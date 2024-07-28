# evaluate_neural_network.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class EvaluateNeuralNetwork:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.clf = MLPClassifier()

    def train(self):
        self.clf.fit(self.X_train, self.y_train)

    def predict(self):
        return self.clf.predict(self.X_test)

    def evaluate(self, y_pred):
        print("Confusion Matrix: ", confusion_matrix(self.y_test, y_pred))
        print("Accuracy : ", accuracy_score(self.y_test, y_pred)*100)
        print("Report : ", classification_report(self.y_test, y_pred))

    def tune_hyperparameters(self):
        param_grid = {
            'hidden_layer_sizes': [(10,), (20,), (30,)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd', 'lbfgs'],
            'learning_rate_init': [0.01, 0.001, 0.0001]
        }
        grid_search = GridSearchCV(self.clf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Accuracy: ", grid_search.best_score_)

    def random_search(self):
        param_grid = {
            'hidden_layer_sizes': [(10,), (20,), (30,)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd', 'lbfgs'],
            'learning_rate_init': [0.01, 0.001, 0.0001]
        }
        random_search = RandomizedSearchCV(self.clf, param_grid, cv=5, scoring='accuracy', n_iter=10)
        random_search.fit(self.X_train, self.y_train)
        print("Best Parameters: ", random_search.best_params_)
        print("Best Accuracy: ", random_search.best_score_)

    def cross_validation(self):
        scores = cross_val_score(self.clf, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print("Accuracy Scores: ", scores)
        print("Mean Accuracy: ", scores.mean())

    def handle_imbalanced_data(self):
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(self.X_train, self.y_train)
        rus = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = rus.fit_resample(X_train_res, y_train_res)
        return X_train_res, y_train_res

    def feature_scaling(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

def main():
    # Load dataset
    df = pd.read_csv('balance-scale.csv')

    # Split dataset into features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create evaluate neural network object
    enn = EvaluateNeuralNetwork(X_train, y_train, X_test, y_test)

    # Handle imbalanced data
    X_train_res, y_train_res = enn.handle_imbalanced_data()
    enn.X_train = X_train_res
    enn.y_train = y_train_res

    # Feature scaling
    enn.feature_scaling()

    # Train neural network
    enn.train()

    # Make predictions
    y_pred = enn.predict()

    # Evaluate neural network
    enn.evaluate(y_pred)

    # Tune hyperparameters
    enn.tune_hyperparameters()

    # Random search
    enn.random_search()

    # Cross validation
    enn.cross_validation()

if __name__ == '__main__':
    main()

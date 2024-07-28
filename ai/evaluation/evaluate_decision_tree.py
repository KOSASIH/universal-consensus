# evaluate_decision_tree.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class EvaluateDecisionTree:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.clf = DecisionTreeClassifier()

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
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
        }
        grid_search = GridSearchCV(self.clf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Accuracy: ", grid_search.best_score_)

    def random_search(self):
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
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

    # Create evaluate decision tree object
    edt = EvaluateDecisionTree(X_train, y_train, X_test, y_test)

    # Handle imbalanced data
    X_train_res, y_train_res = edt.handle_imbalanced_data()
    edt.X_train = X_train_res
    edt.y_train = y_train_res

    # Feature scaling
    edt.feature_scaling()

    # Train decision tree
    edt.train()

    # Make predictions
    y_pred = edt.predict()

    # Evaluate decision tree
    edt.evaluate(y_pred)

    # Tune hyperparameters
    edt.tune_hyperparameters()

    # Random search
    edt.random_search()

    # Cross validation
    edt.cross_validation()

if __name__ == '__main__':
    main()

# decision_tree.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)

    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def evaluate(self, y_test, y_pred):
        print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
        print("Accuracy : ", accuracy_score(y_test, y_pred)*100)
        print("Report : ", classification_report(y_test, y_pred))

    def plot_tree(self, feature_names, class_names):
        plt.figure(figsize=(15, 10))
        tree.plot_tree(self.clf, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
        plt.show()

def main():
    # Load dataset
    df = pd.read_csv('balance-scale.csv')

    # Split dataset into features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create decision tree object
    dt = DecisionTree(criterion='gini', max_depth=None)

    # Train decision tree
    dt.train(X_train, y_train)

    # Make predictions
    y_pred = dt.predict(X_test)

    # Evaluate decision tree
    dt.evaluate(y_test, y_pred)

    # Plot decision tree
    dt.plot_tree(X.columns, ['L', 'B', 'R'])

if __name__ == '__main__':
    main()

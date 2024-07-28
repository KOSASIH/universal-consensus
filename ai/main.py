# main.py
import argparse
import logging
import os
import sys
from datetime import datetime
from utils.helpers import load_data, split_data, scale_data, handle_imbalanced_data, get_class_weights, get_metrics, plot_confusion_matrix
from evaluate_neural_network import EvaluateNeuralNetwork
from evaluate_decision_tree import EvaluateDecisionTree

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Advanced Machine Learning Project')
    parser.add_argument('--data_file', type=str, required=True, help='Path to data file')
    parser.add_argument('--model_type', type=str, required=True, choices=['neural_network', 'decision_tree'], help='Type of model to use')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for neural network training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for neural network training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for neural network training')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers for neural network')
    parser.add_argument('--num_hidden_units', type=int, default=128, help='Number of hidden units for neural network')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for neural network')
    parser.add_argument('--tree_depth', type=int, default=5, help='Depth of decision tree')
    parser.add_argument('--num_estimators', type=int, default=100, help='Number of estimators for decision tree')
    parser.add_argument('--max_features', type=str, default='auto', help='Maximum features for decision tree')
    parser.add_argument('--class_weight', type=str, default='balanced', help='Class weight for decision tree')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logger.info('Arguments: %s', args)

    # Load data
    data = load_data(args.data_file)
    logger.info('Data loaded')

    # Split data
    X_train, X_test, y_train, y_test = split_data(data.drop('target', axis=1), data['target'], test_size=args.test_size, random_state=args.random_state)
    logger.info('Data split')

    # Scale data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    logger.info('Data scaled')

    # Handle imbalanced data
    X_train_res, y_train_res = handle_imbalanced_data(X_train_scaled, y_train)
    logger.info('Imbalanced data handled')

    # Get class weights
    class_weights = get_class_weights(y_train_res)
    logger.info('Class weights calculated')

    # Train model
    if args.model_type == 'neural_network':
        model = EvaluateNeuralNetwork(X_train_res, y_train_res, X_test_scaled, y_test, num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, num_hidden_layers=args.num_hidden_layers, num_hidden_units=args.num_hidden_units, dropout_rate=args.dropout_rate)
    elif args.model_type == 'decision_tree':
        model = EvaluateDecisionTree(X_train_res, y_train_res, X_test_scaled, y_test, tree_depth=args.tree_depth, num_estimators=args.num_estimators, max_features=args.max_features, class_weight=args.class_weight)
    logger.info('Model trained')

    # Make predictions
    y_pred = model.predict()
    logger.info('Predictions made')

    # Evaluate model
    accuracy, precision, recall, f1_score = get_metrics(y_test, y_pred)
    logger.info('Model evaluated')

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    logger.info('Confusion matrix plotted')

    # Save results
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file = os.path.join(results_dir, f'results_{args.model_type}_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt')
    with open(results_file, 'w') as f:
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
                f.write(f'F1 Score: {f1_score:.4f}\n')
    logger.info('Results saved')

if __name__ == '__main__':
    main()

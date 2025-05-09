from BGShapTree.BGShapTree import ForestExplainer
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from datetime import datetime
import numpy as np



def ctime():
    """A formatter on current time used for printing running status."""
    ctime = "[" + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "]"
    return ctime


def load_data():
    training_path = 'diabetes-training.csv'
    test_path = 'diabetes-test.csv'
    training_X = []
    training_y = []
    test_X = []
    test_y = []
    class_number = 0
    print('{} Loading Training Set:'.format(ctime()))
    with open(training_path, 'r') as f:
        raw_data = f.readlines()
        for line in tqdm(raw_data):
            instance = line.split(' ')[:-1]
            data = instance[1:]
            for i in range(len(data)):
                data[i] = float(data[i])
            label = int(instance[0])
            if label + 1 > class_number:
                class_number = label + 1
            training_X.append(data)
            training_y.append(label)
    print('{} Loading Test Set'.format(ctime()))
    with open(test_path, 'r') as f:
        raw_data = f.readlines()
        for line in tqdm(raw_data):
            instance = line.split(' ')[:-1]
            data = instance[1:]
            for i in range(len(data)):
                data[i] = float(data[i])
            label = int(instance[0])
            test_X.append(data)
            test_y.append(label)
    dimension = len(test_X[0])
    return training_X, np.asarray(training_y), test_X, test_y, dimension, class_number


def run():
    # Load data from diabetes dataset
    print("{} Loading Data:".format(ctime()))
    training_X, training_y, test_X, test_y, dimension, class_number = load_data()

    # Train the target random forests
    target_model = RandomForestClassifier(n_estimators=10)
    target_model.fit(training_X, training_y)
    target_instance = test_X[0]
    target_category = target_model.predict([target_instance])[0]

    # Initialize the explainer
    explainer = ForestExplainer(model=target_model, data_set=training_X)

    # Calculate the importance values of d individual features
    feature_importance = explainer.shapley_value([target_instance])
    print('Importance of individual features:')
    print(feature_importance[0][target_category])

    # Calculate the importance value of given feature set
    target_set = [0, 1, 2]
    set_importance = explainer.set_valuation(target_instance, target_set)
    print('Importance of feature set ' + str(target_set) + ':')
    print(set_importance)

    # Search salient feature set of target size
    target_size = 5
    salient_set = explainer.search_for_salient_set(x=target_instance, tau=10, size=target_size)
    print('Salient feature set of size ' + str(target_size) + ':')
    print(salient_set)


if __name__ == '__main__':
    run()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_split_dataset():
    iris = load_iris()
    x = iris.data
    y = iris.target
    class_1 = 1  
    class_2 = 2 
    binary_x = x[(y == class_1) | (y == class_2)]
    binary_y = y[(y == class_1) | (y == class_2)]
    label_encoder = LabelEncoder()
    binary_y = label_encoder.fit_transform(binary_y)
    x_train, x_test, y_train, y_test = train_test_split(binary_x, binary_y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

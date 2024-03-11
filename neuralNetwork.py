import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


X_train = np.array([[1.0, 1.0], [9.4, 6.4], [2.5, 2.1], [8.0, 7.7], [0.5, 2.2], [7.9, 8.4], [7.0, 7.0], [2.8, 0.8], [1.2, 3.0], [7.8, 6.1]])
y_train = np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0])
X_test = np.array([[3.0, 2.8], [1.5, 0.3], [8.0, 6.7], [9.9, 5.2], [3.5, 3.5]])
y_test = np.array([1, 1, 0, 0, 1])

clf = MLPClassifier(activation='logistic', solver='sgd', learning_rate='constant', learning_rate_init=0.3, random_state=1).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Prediction:')
print(y_pred)
print('Actual:')
print(y_test)
print('Accuracy:')
print(accuracy_score(y_test, y_pred))

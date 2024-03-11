import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# Import data
data_train = pd.read_csv('data.csv')
X_train = data_train.drop('source', axis=1)
Y_train = data_train['source']
data_test = pd.read_csv('data_test.csv')
X_test = data_test.drop('source', axis=1)
Y_test = data_test['source']

# Create decision tree
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, Y_train)

# Show tree plot
plt.figure(figsize=(12, 10))
tree.plot_tree(clf, filled=True)
plt.show()

# Make predictions
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)

# Test accuracy
accuracy_train = accuracy_score(Y_train, pred_train)
print("Training Set Accuracy:", accuracy_train)
accuracy_test = accuracy_score(Y_test, pred_test)
print("Training Set Accuracy:", accuracy_test)

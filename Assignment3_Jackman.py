# Eric Jackman DSC311 Assignment 3
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


def question_6(x, y):
    print("Question 6:")

    # Linear regression
    linearReg = linear_model.LinearRegression()

    # Create forward selection object
    sfs1 = sfs(linearReg,
               k_features=9,
               forward=True,
               floating=False,
               verbose=2,
               scoring='neg_mean_squared_error')

    # Forward selection
    sfs1 = sfs1.fit(x, y)

    # Print the selected columns
    selected_col = list(sfs1.k_feature_names_)
    print("\nFeatures Selected:\n", selected_col)


def question_7(x, y):
    print("\nQuestion 7:")

    # 10-Fold Sampling
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Linear regression
    linearReg = linear_model.LinearRegression()
    linearReg.fit(x_train, y_train)

    # Print coefficients
    print("Coefficients:\n", linearReg.coef_)

    # Make predictions
    y_predictions = linearReg.predict(x_test)

    # Print mean squared error
    print("Mean squared error:\n", mean_squared_error(y_test, y_predictions))


if __name__ == "__main__":
    # Create dictionary of Table 8.5
    data = {"Age": [55, 43, 37, 82, 23, 46, 38, 50, 29, 42, 35, 38, 31, 71],
            "Educ. level": [1.0, 2.0, 5.0, 3.0, 3.2, 5.0, 4.2, 4.0, 4.5, 4.1, 4.5, 2.5, 4.8, 2.3],
            "Max. temp": [25, 31, 15, 20, 10, 12, 16, 26, 15, 21, 30, 13, 8, 12],
            "Weight": [77, 110, 70, 85, 65, 75, 75, 63, 55, 66, 95, 72, 83, 115],
            "Arabic": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            "Indian": [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
            "Mediterr.": [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
            "Oriental": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
            "Fast food": [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            "Height": [175, 195, 172, 180, 168, 173, 180, 165, 158, 163, 190, 172, 185, 192]}

    # Create dataframe from dictionary
    df = pd.DataFrame(data)

    # Split into x and y
    dfX = df.loc[:, "Age": "Fast food"]
    dfY = df["Height"]

    # Convert to numpy array
    X = dfX.to_numpy()
    Y = dfY.to_numpy()

    question_6(dfX, dfY)
    question_7(X, Y)

### Module imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

########
# Determine the optimal polynomial degree
# IDEA: given features and a target variable we determine the optimal degree
# STEP 1: split the data in train/validation/test data (given function parameters)
# STEP 2: create polynomial features of the train and validation data
# STEP 3: create/fit a linear model on the training data
# STEP 4: Predict values on validation data
# Iterate through steps  2 - 4 with different degree (default = 5)
# Plot the MSE and R^2 for the different degree values and annotate the best value
# Give the output of the MSE and R^2 on the test data

def poly_optimal_degree_determiner(x, y, train_size=0.6, validation_size=0.2, test_size=0.2, max_degree=5, graphical = True):
    # data split
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=0)
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, train_size=validation_size/(1 - train_size), random_state=0)

    # reshape data
    x_train = x_train.values.reshape(-1, 1)
    x_validation = x_validation.values.reshape(-1, 1)

    # Create model
    lr = LinearRegression(normalize=True)

    # lists with metrics
    mae_list = []
    mse_list = []
    r2_list = []
    for i in range(1, max_degree + 1):
        # polynomial feature creation
        poly = PolynomialFeatures(i)
        x_train_poly = poly.fit_transform(x_train)
        x_validation_poly = poly.fit_transform(x_validation)

        # Fit the model
        lr.fit(x_train_poly, y_train)

        # Make predictions
        y_pred = lr.predict(x_validation_poly)

        # Compute metrics
        mae = mean_absolute_error(y_validation, y_pred)
        mse = mean_squared_error(y_validation, y_pred)
        r2 = r2_score(y_validation, y_pred)

        # Add metrics to dictionary
        mae_list.append(mae)
        mse_list.append(mse)
        r2_list.append(r2)

    # Obtain best degree (with prioritization to mse)
    best_degree = mse_list.index(min(mse_list)) + 1
    best_mse = mse_list[best_degree - 1]
    best_r2 = r2_list[best_degree - 1]

    # graphical representation of the evolution
    if graphical:
        fig, ax = plt.subplots()
        ax.plot(list(range(1, max_degree + 1)), mse_list, marker = ".", color = "r", label = "MSE" )
        ax.set_xlabel("Degree values")
        ax.set_ylabel("MSE values")
        ax.set_xticks([i for i in range(1, max_degree + 1)])
        ax2 = ax.twinx()
        ax2.plot(list(range(1, max_degree + 1)), r2_list, marker = ".", color = "b", label = "R2")
        ax2.set_ylabel("R2 values")
        ax2.tick_params(color = "b")
        plt.axvline(best_degree)
        fig.legend(loc="upper right")
        plt.show()

    # Output
    print("The optimal degree is: {}".format(best_degree))
    print("With a respective mse of: {}".format(round(best_mse, 2)))
    print("And with a r2 of: {}".format(round(best_r2, 2)))

########
# Obtain the decision rules once the decision tree is obtained
# IDEA: print the decision rules given the decision tree object and the feature names
# Use If Then Else prints whereby indentations are used for rules within rules

def tree_to_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)




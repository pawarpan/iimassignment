import pickle as pt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X_train, y_train):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    pt.dump(model, open('decision_tree_model.pkl', 'wb'))
    # Save the model to a file

def predict(X_test):
    model = pt.load(open('decision_tree_model.pkl', 'rb'))
    # Load the model from the file
    model = pt.load(open('decision_tree_model.pkl', 'rb'))
    #Make predictions using the loaded model
    return model.predict(X_test)

def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Calculate Mean Squared Error and R-squared
    return mse, r2

  


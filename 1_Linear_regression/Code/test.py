import numpy as np
import Linear_regression_module as Linear_regression


def mean_squared_error(y_true, y_pred):
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


def main():
    # load dataset
    train_data = np.loadtxt("G:\github\MachineLearning_Code_Practice\Linear_regression\personal_version\\train_data.txt")
    test_data = np.loadtxt("G:\github\MachineLearning_Code_Practice\Linear_regression\personal_version\\test_data.txt")
    
    
    # Split train and test X,y
    X_train,y_train = train_data[:,:-1], train_data[:,-1]
    X_test,y_test = test_data[:,:-1], test_data[:,-1]
    
    # train model
    clf = Linear_regression.LinearRegression()
    clf.fit(X_train, y_train)

    # predict
    y_pred = clf.predict(X_test)

    print("MSE: ", mean_squared_error(y_test, y_pred))
    print(clf.w)
    
    

if __name__ == "__main__":
    main()           
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")


def test_all_model(train_X, train_y, models, seed, display=0):
    """ Test the performance of the data set in the given model,
        and output the results, evaluating the result with accuracy.
        Parameters
        ----------
            train_X: trainning set
            train_y: label set
            models: Selected model()
            display:Whether to show the result

        Returns
        -------
            scores_set : A list of pre-processed positive PANAS for each student,
                                    in the format of: [ accuracy score, accuracy score, ...]
            RMSE_set : A list of pre-processed negative PANAS for each student,
                                    in the format of: [ RMSE score, RMSE score, ...]
    """
    scores_set = []
    RMSE_set = []
    for clf_name, clf in models.items():
        kfold = KFold(n_splits=10, random_state=seed)
        MSE = cross_val_score(clf, train_X, train_y, cv=kfold, scoring='neg_mean_squared_error')
        scores = cross_val_score(clf, train_X, train_y, cv=kfold, scoring='accuracy')
        RMSE = np.sum(np.sqrt(-1 * MSE)) / len(MSE)
        scores_set.append(scores.mean())
        RMSE_set.append(RMSE)
        if (display == 1):
            print('Accuracy: {:.2f} ( RMSE:{:.2f}) [{}]'.format(scores.mean(), RMSE, clf_name))
    if (display == 1):
        print()
    return scores_set, RMSE_set


def find_chi_square_num(train_X, train_y, models, kfold, seed):

    """ finding the best chi square num.
        ----------
            train_X: trainning set
            train_y: label set
            models: Selected model()

        Returns
        -------
            max_temp : Maximum
            temp : best num for chi square
    """
    max_temp = 0
    temp = 0
    for num in range(1, len(train_X[0])):
        Transx = SelectKBest(chi2, k=num).fit_transform(train_X, train_y)
        scores_set, RMSE_set = test_all_model(Transx, train_y, models, seed)
        if sum(scores_set) > max_temp:
            max_temp = sum(scores_set)
            temp = Transx
    return max_temp, temp


def find_best_features(train_X, train_y, models, kfold, seed):
    """ finding the best features.
        ----------
            train_X: trainning set
            train_y: label set
            models: Selected model()

        Returns
        -------
            Trans_set["VarianceThreshold"] : best trainning set for VarianceThreshold
            Trans_set["tree_GBDT"] : best trainning set for tree_GBDT
            Trans_set["Chi_square"] : best trainning set for Chi_square
    """
    Trans_set = {}

    Trans_set["VarianceThreshold"] = (VarianceThreshold(threshold=0).fit_transform(train_X))
    Trans_set["tree_GBDT"] = (SelectFromModel(GradientBoostingClassifier()).fit_transform(train_X, train_y))
    x_value, x_new = find_chi_square_num(train_X, train_y, models, kfold, seed)
    Trans_set["Chi_square"] = x_new

    models_names = ["LR","KNN","NB","LDA","TREE","SVM"]
    b_width = 0.3
    x = list(range(len(models_names)))
    for key, value in Trans_set.items():
        print('method: {}'.format(key))
        print('len: {}'.format(len(value[0])))
        print(value)
        scores_set ,RMSE_set= test_all_model(value, train_y, models, seed, 1)

        plt.bar(x, scores_set, width=0.3, label=key, tick_label=models_names)
    plt.legend()
    plt.show()

    return Trans_set["VarianceThreshold"], Trans_set["tree_GBDT"], Trans_set["Chi_square"]


def test_logistic_regression(X1 , X2 , train_y, name):
    """ Using the new data set to train logistic regression ,and using GridSearch to find best hyper-parameters.
        ----------
            X1: original data set
            X2: data set processed by find_best_features
            train_y: label set
            name: name of label set

    """
    params = [{'penalty': ['l1', 'l2'],
                         'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                         'solver': ['liblinear']},
                        {'penalty': ['l2'],
                         'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                         'solver': ['newton-cg', 'sag', 'lbfgs']}]

    clf = GridSearchCV(LogisticRegression(tol=1e-6), params, cv=10)
    clf.fit(X1, train_y)
    print("Best score in first time:\n", clf.best_score_)
    clf.fit(X2, train_y)
    print("Best parameters set found:", clf.best_params_)
    print("Best score in second time:", clf.best_score_)
    print("model.best_estimator_", clf.best_estimator_)
    result_temp = []
    for p, s in zip(clf.cv_results_['params'],
                    clf.cv_results_['mean_test_score']):
        result_temp.append(s)
        print(p, s)
    plt.plot(range(0, len(result_temp)), result_temp, '-', label=name)
    plt.ylabel("accuracy_score")
    plt.xlabel("penalty, C, solver")
    plt.title("test_logistic_regression")
    plt.legend()
    clf2 = GridSearchCV(LogisticRegression(tol=1e-6), params, scoring='neg_mean_squared_error', cv=10)
    clf2.fit(X1, train_y)
    clf2.fit(X2, train_y)
    MSE = clf2.best_score_
    RMSE = np.sqrt(-1 * MSE)
    print("neg_mean_squared_error score :\n", RMSE)


def test_svm(X1, X2, train_y, name):
    """ Using the new data set to train SVM ,and using GridSearch to find best hyper-parameters.

            X1: original data set
            X2: data set processed by find_best_features
            train_y: label set
            name: name of label set

    """
    model = SVC(probability=True)

    params = [
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
        {'kernel': ['poly'], 'C': [1, 10], 'degree': [2, 3]},
        {'kernel': ['rbf'], 'C': [1, 10, 100, 1000],
         'gamma': [1, 0.1, 0.01, 0.001]}]
    clf = GridSearchCV(estimator=model, param_grid=params, cv=10)
    clf.fit(X1, train_y)
    print("Best score in first time:\n", clf.best_score_)
    clf.fit(X2, train_y)
    print("Best parameters set found:", clf.best_params_)
    print("Best score in second time", clf.best_score_)
    print("model.best_estimator_", clf.best_estimator_)
    result_temp = []
    for p, s in zip(clf.cv_results_['params'],
                    clf.cv_results_['mean_test_score']):
        result_temp.append(s)
        print(p, s)
    plt.plot(range(0, len(result_temp)), result_temp, '-', label=name)
    plt.ylabel("accuracy_score")
    plt.xlabel("kernel, C, degree, gamma")
    plt.title("test_svm")
    plt.legend()
    clf2 = GridSearchCV(estimator=model, param_grid=params, scoring='neg_mean_squared_error', cv=10)
    clf2.fit(X1, train_y)
    clf2.fit(X2, train_y)
    MSE = clf2.best_score_
    RMSE = np.sqrt(-1 * MSE)
    print("neg_mean_squared_error score :\n", RMSE)


def test_knn(X1, X2, train_y, name):
    """ Using the new data set to train SVM ,and using GridSearch to find best hyper-parameters.

            X1: original data set
            X2: data set processed by find_best_features
            train_y: label set
            name: name of label set

    """
    model = KNeighborsClassifier()
    params = [
        {'weights': ['uniform'],
            'n_neighbors': [i for i in range(1, 11)]},
        {'weights': ['distance'],
            'n_neighbors': [i for i in range(1, 11)],
            'p': [i for i in range(1, 6)]}]
    clf = GridSearchCV(estimator=model, param_grid=params, cv=10)
    clf.fit(X1, train_y)
    print("Best score in first time:\n", clf.best_score_)
    clf.fit(X2, train_y)
    print("Best parameters set found:", clf.best_params_)
    print("Best score in second time", clf.best_score_)
    print("model.best_estimator_", clf.best_estimator_)
    result_temp = []
    for p, s in zip(clf.cv_results_['params'],
                    clf.cv_results_['mean_test_score']):
        result_temp.append(s)
        print(p, s)
    plt.plot(range(0,len(result_temp)), result_temp, '-', label=name)
    plt.ylabel("accuracy_score")
    plt.xlabel("weights, n_neighbors, p")
    plt.title("test_knn")
    plt.legend()
    clf2 = GridSearchCV(estimator=model, param_grid=params, scoring='neg_mean_squared_error', cv=10)
    clf2.fit(X1, train_y)
    clf2.fit(X2, train_y)
    MSE = clf2.best_score_
    RMSE = np.sqrt(-1 * MSE)
    print("RMSE score :\n", RMSE)


def test_DT(X1, X2, train_y, name):
    """ Using the new data set to train DT ,and using GridSearch to find best hyper-parameters.

            X1: original data set
            X2: data set processed by find_best_features
            train_y: label set
            name: name of label set

    """
    model = DecisionTreeClassifier()
    params = {'max_depth': range(1, 21), 'criterion': np.array(['entropy', 'gini'])}
    clf = GridSearchCV(estimator=model, param_grid=params, cv=10)
    clf.fit(X1, train_y)
    print("Best score in first time:\n", clf.best_score_)
    clf.fit(X2, train_y)
    print("Best parameters set found:", clf.best_params_)
    print("Best score in second time", clf.best_score_)
    print("model.best_estimator_", clf.best_estimator_)
    clf2 = GridSearchCV(estimator=model, param_grid=params, scoring='neg_mean_squared_error', cv=10)
    clf2.fit(X1, train_y)
    clf2.fit(X2, train_y)
    MSE = clf2.best_score_
    RMSE = np.sqrt(-1 * MSE)
    print("RMSE score :\n", RMSE)


def min_max_normalize(X):
    for i in range(0, len(X[1])):
        v = X[:, i]
        X[:, i] = (v - v.min()) / (v.max() - v.min())
    return X


def run():
    # flourishing positive negative
    paper_features = np.genfromtxt("paper_features_1.csv", delimiter=",", skip_header=1)
    all_features = np.genfromtxt("all_features_1.csv", delimiter=",", skip_header=1)

    x_paper_features = paper_features[:, 1:-3]
    x_all_features = all_features[:, 1:-3]

    y_negative = paper_features[:, -1]
    y_positive = paper_features[:, -2]
    y_flourishing = paper_features[:, -3]

    # min-max normalize
    x_paper_features = min_max_normalize(x_paper_features)
    x_all_features = min_max_normalize(x_all_features)
    df = pd.DataFrame(x_all_features)
    df.to_csv("temp.csv", index=False)

    # models contains all models
    models = {}
    models["LogisticRegression"] = LogisticRegression()
    models["KNeighborsClassifier"] = KNeighborsClassifier()
    models["GaussianNB"] = GaussianNB()
    models["LinearDiscriminantAnalysis"] = LinearDiscriminantAnalysis()
    models["DecisionTreeClassifier"] = DecisionTreeClassifier()
    models["SVC"] = SVC()

    validation_size = 0.2
    seed = 7

    # models selection part

    print("test best model for paper_features\n")
    for x, y in zip(["negative", "positive", "flourishing"],
                 [y_negative, y_positive, y_flourishing]):

        print("y set is: {}".format(x))
        test_all_model(x_paper_features, y, models, seed, 1)
    print("test best model for all_features\n")
    for x, y in zip(["negative", "positive", "flourishing"],
            [y_negative, y_positive, y_flourishing]):

        print("y set is: {}".format(x))
        test_all_model(x_all_features, y, models, seed, 1)


    kfold = KFold(n_splits=10, random_state=seed)

    # features selection part

    print()
    print("test best features for all_features\n")
    good_set ={}
    for name1, y in zip(["negative", "positive", "flourishing"],
                    [y_negative, y_positive, y_flourishing]):
        print("y set is: {}".format(name1))
        print()
        X1, X2, X3 = find_best_features(x_all_features, y, models, kfold, seed)
        good_set[name1] = X3
    print()
    print("end of test best features for all_features\n")

    # final test for four models we have choosed

    for name1, y in zip(["negative", "positive", "flourishing"],
                    [y_negative, y_positive, y_flourishing]):
        print("y set is: {}\n".format(name1))

        print("test_logistic_regression\n")

        print("test_svm\n")

        print("test_knn\n")

        print("test_DT\n")

    plt.show()



if __name__ == '__main__':
    run()

import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay, confusion_matrix, f1_score, fbeta_score, make_scorer

# импорт и подготовка датасета
df = pd.read_csv('data/new_data.csv')

rs = {'Прогрессия'}
tr = set(df.columns)-rs

X = df[list(tr)]
Y = df[list(rs)]

def randomoversample(x, y):
    ros = RandomOverSampler(random_state=10)
    return ros.fit_resample(x, y)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
# smote = SMOTE(random_state=10)
# X_train_res, Y_train_res = randomoversample(X_train, Y_train)
# data = X_train, X_test, Y_train, Y_test

# smote = SMOTE(random_state=10)
# X_res, Y_res = smote.fit_resample(X, Y)

# X_train_res, X_test, Y_train_res, Y_test = train_test_split(X_res, Y_res, test_size=0.2)
# data = X_train_res, X_test, Y_train_res, Y_test

# smote = SMOTE(random_state=10)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)
# data = X_train_res, X_test, Y_train_res, Y_test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
X_train_res, Y_train_res = X_train, Y_train
data = X_train, X_test, Y_train, Y_test

fbeta = make_scorer(fbeta_score, beta = 2)

def params_knn(X_train, Y_train, score='f1'):
    params = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]  # для метрики Минковского
    }
    grid = GridSearchCV(KNeighborsClassifier(), params, cv=5, scoring=score)
    grid.fit(X_train, Y_train)
    print(f"Best parameters for KNN: {grid.best_params_}")
    return grid.best_estimator_

def params_svm(X_train, Y_train, score='f1'):
    params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto']
    }
    grid = GridSearchCV(SVC(), params, cv=5, scoring=score)
    grid.fit(X_train, Y_train)
    print(f"Best parameters for SVM: {grid.best_params_}")
    return grid.best_estimator_

def params_dt(X_train, Y_train, score='f1'):
    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    grid = GridSearchCV(DecisionTreeClassifier(random_state=0), params, cv=5, scoring=score)
    grid.fit(X_train, Y_train)
    print(f"Best parameters for Decision Tree: {grid.best_params_}")
    return grid.best_estimator_

def params_lr(X_train, Y_train, score='f1'):
    params = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'saga'],
        'class_weight': [None, 'balanced']
    }
    grid = GridSearchCV(LogisticRegression(max_iter=1000), params, cv=5, scoring=score)
    grid.fit(X_train, Y_train)
    print(f"Best parameters for Logistic Regression: {grid.best_params_}")
    return grid.best_estimator_

def params_rf(X_train, Y_train, score='f1'):
    params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=0), params, cv=5, scoring=score)
    grid.fit(X_train, Y_train)
    print(f"Best parameters for Random Forest: {grid.best_params_}")
    return grid.best_estimator_

print(params_rf(X_train_res, Y_train_res, fbeta))

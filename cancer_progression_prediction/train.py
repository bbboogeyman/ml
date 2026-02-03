import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import shap

import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay, confusion_matrix, f1_score, fbeta_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE

import lime
import lime.lime_tabular
from lime import lime_tabular

# импорт и подготовка датасета
df = pd.read_csv('data/new_data.csv')

rs = {'Прогрессия'}
tr = set(df.columns)-rs

X = df[list(tr)]
Y = df[list(rs)]

def randomoversample(x, y):
    ros = RandomOverSampler(random_state=10)
    return ros.fit_resample(x, y)

# разбиение и балансировка данных
# smote = SMOTE(random_state=10)
# X_res, Y_res = smote.fit_resample(X, Y)

# X_train_res, X_test, Y_train_res, Y_test = train_test_split(X_res, Y_res, test_size=0.2)
# data = X_train_res, X_test, Y_train_res, Y_test


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
X_train_res, Y_train_res = X_train, Y_train
data = X_train, X_test, Y_train, Y_test

# smote = SMOTE(random_state=10)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)
# data = X_train_res, X_test, Y_train_res, Y_test

explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train_res.values,
    feature_names=list(tr),
    class_names=['No Progression', 'Progression'],
    mode='classification',
    random_state=10
)

def explain_model(model, data_test, instance_idx=0):
    """
    Генерирует LIME объяснение для конкретного примера
    """
    exp = explainer.explain_instance(
        data_test.iloc[instance_idx].values,
        model.predict_proba,
        num_features=10
    )
    return exp

def get_lime_explanation(model, data, instance_idx=0):
    """Возвращает LIME объяснение в виде списка"""
    exp = explainer.explain_instance(
        data.iloc[instance_idx].values,
        model.predict_proba,
        num_features=10
    )
    return exp.as_list()

# модели

def KNN(data):
    data_train, data_test, res_train, res_test = data
    model_neighbor = KNeighborsClassifier(metric='manhattan', n_neighbors=3, p=1, weights='distance')
    res_train = res_train.values.ravel()
    model_neighbor = model_neighbor.fit(data_train, res_train)

    return model_neighbor

def KNN_exp(data):
    data_train, data_test, res_train, res_test = data
    model_neighbor = KNeighborsClassifier(n_neighbors=11, metric="minkowski", p=1)
    res_train = res_train.values.ravel()
    model_neighbor = model_neighbor.fit(data_train, res_train)

    exp = explain_model(model_neighbor, data_test)

    return exp.as_list()

def SVM(data):
    data_train, data_test, res_train, res_test = data
    clf = SVC(C=0.5, degree=2, kernel='linear')
    res_train = res_train.values.ravel()
    clf = clf.fit(data_train, res_train)

    return clf

def SVM_exp(data, instance_idx=0, show_table=True):
    data_train, data_test, res_train, res_test = data
    
    clf = SVC(C=0.1, degree=2, kernel='linear')
    clf.fit(data_train, res_train.values.ravel())
    
    exp = explain_model(clf, data_test, instance_idx)
    explanations = exp.as_list()
    
    if show_table:
        print("\n" + "="*55)
        print(f"SVM LIME explanation")
        print("-"*55)
        print(f"{'Признак':<30} {'Вклад':>10}")
        print("-"*55)
        for feature, weight in explanations:
            print(f"{feature:<30} {weight:>10.4f}")
        print("="*55)

def DecisionTree(data):
    data_train, data_test, res_train, res_test = data
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features='sqrt',
                       min_samples_leaf=2, random_state=0)
    dt = dt.fit(data_train, res_train)

    return dt

def DecisionTree_exp(data, instance_idx=0, show_table=True):
    data_train, data_test, res_train, res_test = data
    
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features='sqrt',
                       min_samples_leaf=2, random_state=0)
    dt.fit(data_train, res_train)
    
    exp = explain_model(dt, data_test, instance_idx)
    explanations = exp.as_list()
    
    if show_table:
        print("\n" + "="*55)
        print(f"Decision Tree LIME explanation")
        print("-"*55)
        print(f"{'Признак':<30} {'Вклад':>10}")
        print("-"*55)
        for feature, weight in explanations:
            print(f"{feature:<30} {weight:>10.4f}")
        print("="*55)


def LR(data):
    data_train, data_test, res_train, res_test = data
    lr = LogisticRegression(C=1, intercept_scaling=1.0, max_iter=100)
    res_train = res_train.values.ravel()
    lr = lr.fit(data_train, res_train)

    return lr

def LR_exp(data, instance_idx=0, show_table=True):
    data_train, data_test, res_train, res_test = data
    
    lr = LogisticRegression(C=1, intercept_scaling=1.0, max_iter=100)
    res_train = res_train.values.ravel()
    lr.fit(data_train, res_train)
    
    exp = explain_model(lr, data_test, instance_idx)
    explanations = exp.as_list()
    
    if show_table:
        print("\n" + "="*55)
        print(f"Logistic Regression LIME explanation")
        print("-"*55)
        print(f"{'Feature':<30} {'Weight':>10}")
        print("-"*55)
        for feature, weight in explanations:
            print(f"{feature:<30} {weight:>10.4f}")
        print("="*55)
    
    return explanations

def RandomForest(data):
    data_train, data_test, res_train, res_test = data
    rf = RandomForestClassifier(min_samples_leaf=4, n_estimators=200, random_state=0)
    res_train = res_train.values.ravel()
    rf = rf.fit(data_train, res_train)

    return rf

def RandomForest_exp(data, instance_idx=0, show_table=True):
    data_train, data_test, res_train, res_test = data
    
    rf = RandomForestClassifier(min_samples_leaf=4, n_estimators=200, random_state=0)
    rf.fit(data_train, res_train)
    
    exp = explain_model(rf, data_test, instance_idx)
    explanations = exp.as_list()
    
    if show_table:
        print("\n" + "="*55)
        print(f"Random Forest LIME explanation")
        print("-"*55)
        print(f"{'Feature':<30} {'Weight':>10}")
        print("-"*55)
        for feature, weight in explanations:
            print(f"{feature:<30} {weight:>10.4f}")
        print("="*55)
    
    return explanations

# отклонения
def balanced_accuracy(model, data_test, res_test):
    res_pred = model.predict(data_test)
    accur = balanced_accuracy_score(res_test, res_pred)
    return accur, res_pred

def f1_accuracy(model, data_test, res_test):
    res_pred = model.predict(data_test)
    accur = f1_score(res_test, res_pred, average="binary")
    return accur, res_pred

def f2_accuracy(model, data_test, res_test):
    res_pred = model.predict(data_test)
    accur = fbeta_score(res_test, res_pred, beta=0.5)
    return accur, res_pred

# вывод результатов
predict_data = (balanced_accuracy(KNN(data), X_test, Y_test)[1], balanced_accuracy(SVM(data), X_test, Y_test)[1], 
       balanced_accuracy(DecisionTree(data), X_test, Y_test)[1], balanced_accuracy(LR(data), X_test, Y_test)[1], 
       balanced_accuracy(RandomForest(data), X_test, Y_test)[1])
b_acc = [balanced_accuracy(KNN(data), X_test, Y_test)[0], balanced_accuracy(SVM(data), X_test, Y_test)[0], 
       balanced_accuracy(DecisionTree(data), X_test, Y_test)[0], balanced_accuracy(LR(data), X_test, Y_test)[0], 
       balanced_accuracy(RandomForest(data), X_test, Y_test)[0]]
f1_acc = [f1_accuracy(KNN(data), X_test, Y_test)[0], f1_accuracy(SVM(data), X_test, Y_test)[0], 
          f1_accuracy(DecisionTree(data), X_test, Y_test)[0], f1_accuracy(LR(data), X_test, Y_test)[0], 
          f1_accuracy(RandomForest(data), X_test, Y_test)[0], ]
fb_acc = [f2_accuracy(KNN(data), X_test, Y_test)[0], f2_accuracy(SVM(data), X_test, Y_test)[0], 
          f2_accuracy(DecisionTree(data), X_test, Y_test)[0], f2_accuracy(LR(data), X_test, Y_test)[0], 
          f2_accuracy(RandomForest(data), X_test, Y_test)[0]]
fn = [confusion_matrix(Y_test, predict_data[0])[1][0], confusion_matrix(Y_test, predict_data[1])[1][0],
      confusion_matrix(Y_test, predict_data[2])[1][0], confusion_matrix(Y_test, predict_data[3])[1][0],
      confusion_matrix(Y_test, predict_data[4])[1][0]]

models = ['KNN', 'Support vectors machine (SVM)', 'Decision Tree', 
               'Logistic Regression', 'Random Forest']
res = pd.DataFrame({
    "Models": models,
    "balanced_accuracy": b_acc,
    "f1_accuracy": f1_acc,
    "fbeta_accuracy": fb_acc,
    "FN": fn,
})


print('-'*60)
print(res)
print('-'*60)

LR_exp(data)

def adjust_shap_plot():
    """Функция для настройки отступов графика SHAP."""
    plt.gcf().set_size_inches(12, 7)
    plt.subplots_adjust(left=0.4)
    plt.tight_layout()

def KNNSHAP(data):
    plt.clf()
    model = KNN(data)
    explainer = shap.Explainer(model.predict, data[0])
    shap_values = explainer(data[0])
    shap.plots.beeswarm(shap_values, show=False)
    adjust_shap_plot()
    plt.title("SHAP Values for KNN Model", pad=20)
    plt.show()

def SVMSHAP(data):
    plt.clf()
    model = SVM(data)
    explainer = shap.Explainer(model.predict, data[0])
    shap_values = explainer(data[0])
    shap.plots.beeswarm(shap_values, show=False)
    adjust_shap_plot()
    plt.title("SHAP Values for SVM Model", pad=20)
    plt.show()

def DecisionTreeSHAP(data):
    plt.clf()
    model = DecisionTree(data)
    explainer = shap.Explainer(model.predict, data[0])
    shap_values = explainer(data[0])
    shap.plots.beeswarm(shap_values, show=False)
    adjust_shap_plot()
    plt.title("SHAP Values for Decision Tree Model", pad=20)
    plt.show()

def LRSHAP(data):
    plt.clf()
    model = LR(data)
    explainer = shap.Explainer(model.predict, data[0])
    shap_values = explainer(data[0])
    shap.plots.beeswarm(shap_values, show=False)
    adjust_shap_plot()
    plt.title("SHAP Values for Logistic Regression Model", pad=20)
    plt.show()

def RandomForestSHAP(data):
    plt.clf()
    model = RandomForest(data)
    explainer = shap.Explainer(model.predict, data[0])
    shap_values = explainer(data[0])
    shap.plots.beeswarm(shap_values, show=False)
    adjust_shap_plot()
    plt.title("SHAP Values for Random Forest Model", pad=20)
    plt.show()

#SVMSHAP(data)
#DecisionTreeSHAP(data)
LRSHAP(data)

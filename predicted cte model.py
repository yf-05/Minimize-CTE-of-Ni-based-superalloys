import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from itertools import combinations


# %% input
x_train = pd.read_excel('dataset.xlsx', usecols='A,D:X',
                        sheet_name='train', index_col=0)
y_train = pd.read_excel('dataset.xlsx', usecols='A,C',
                        sheet_name='train', index_col=0)
group = pd.read_excel('dataset.xlsx', usecols='A,B',
                      sheet_name='train', index_col=0)
y_train = np.array(y_train).ravel()
group = np.array(group).ravel()
# standard scale
scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)

# %% regression models' performance with default hyper parameters
SVR = svm.SVR(kernel='rbf')
LinearR = linear_model.Ridge(random_state=42)
LinearElastic = linear_model.ElasticNet(random_state=42, )
RandomF = RandomForestRegressor(random_state=42)
MLP = MLPRegressor(random_state=42, max_iter=10000)
GBR = GradientBoostingRegressor(random_state=42)
AdaBoost = AdaBoostRegressor(random_state=42)
DT = DecisionTreeRegressor(random_state=42)
Knn = KNeighborsRegressor()
XGB = XGBRegressor()
GPR = GaussianProcessRegressor(random_state=42)
output_predict = pd.DataFrame()
model = [SVR, LinearR, LinearElastic, RandomF, MLP, GBR, AdaBoost, DT, Knn, XGB, GPR]
model_name = ['SVR', 'LinearR', 'LinearElastic', 'RandomF', 'MLP', 'GBR', 'AdaBoost', 'DT',
              'Knn', 'XGB', 'GPR']
for i in range(len(model)):
    scores = cross_validate(model[i], x_train, y_train, groups=group, scoring='r2', cv=5)
    print("mean_r2 of " + model_name[i] + " = %0.2f" % scores['test_score'].mean())

# RF and GBR get the best r2 = 0.77 and chosen GBR for feature engineering.

# %% 1st step of feature engineering: Correlation Screening
feature_names = ['T', 'Tm', 'δTm', 'r', 'δr', 'γ', 'χ', 'δχ', 'VEC', 'δVEC',
                 'TUS', 'δTUS', 'MM', 'δMM', 'Cp', 'S', 'H', 'BMN', 'δBMN', 'Tc', 'δTc']
correlation_matrix = pd.DataFrame(x_train).corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.1,
            xticklabels=feature_names, yticklabels=feature_names)
plt.show()

threshold = 0.9

# find feature pairs with high correlation coefficient and remove feature with lower feature importance
high_corr_pairs = np.where(correlation_matrix > threshold)
high_corr_pairs = [(feature_names[i], feature_names[j]) for i, j in zip(*high_corr_pairs) if i != j and i < j]

feature_importance = pd.Series(GBR.fit(x_train, y_train).feature_importances_, index=feature_names)

features_to_remove = set()
for feature1, feature2 in high_corr_pairs:
    if feature_importance[feature1] > feature_importance[feature2]:
        features_to_remove.add(feature2)
    else:
        features_to_remove.add(feature1)

selected_features = [f for f in feature_names if f not in features_to_remove]

# dataset after screening
selected_indices = [feature_names.index(f) for f in selected_features]
x_train_selected_1step = x_train[:, selected_indices]

# Feature "T" was removed in this step

# %% 2nd step of feature engineering: Recursive Feature Elimination (RFE)
feature_names_rfe = np.array(['Tm', 'δTm', 'r', 'δr', 'γ', 'χ', 'δχ', 'VEC', 'δVEC',
                              'TUS', 'δTUS', 'MM', 'δMM', 'Cp', 'S', 'H', 'BMN', 'δBMN', 'Tc', 'δTc'])
feature_importance_rfe = GBR.fit(x_train_selected_1step, y_train).feature_importances_

# the least important feature is iteratively removed
for i in range(len(feature_importance_rfe)):
    selector = SelectFromModel(estimator=GBR, max_features=i + 1, threshold=-np.inf)
    selector.fit(x_train_selected_1step, y_train)
    new_x_train_feature = selector.transform(x_train_selected_1step)
    GBR.fit(new_x_train_feature, y_train)
    print('Number of features = %d' % (i + 1))
    feature_indices = selector.get_support(indices=True)
    print(feature_indices)
    selected_features_rfe = []
    for j in feature_indices:
        selected_features_rfe.append(feature_names_rfe[j])

    print('Selected feature names:' + str(selected_features_rfe))
    print("r2 = %0.4f, MSE = %0.4f" % (GBR.score(new_x_train_feature, y_train),
                                       mean_squared_error(y_train, GBR.predict(new_x_train_feature))))

# choose the former 11 features in which r2 and MSE reach the peak values
# Selected feature names:['Tm', 'δTm', 'δr', 'γ', 'χ', 'δχ', 'δTUS', 'MM', 'δMM', 'Cp', 'S']

# %% 3rd step of feature engineering: Exhaustive Screening
# dataset after rfe
feature_names_es = np.array(['Tm', 'δTm', 'δr', 'γ', 'χ', 'δχ', 'MM', 'δMM', 'Cp', 'S', 'BMN'])
indices = [0, 1, 3, 4, 5, 6, 11, 12, 13, 14, 16]
x_train_selected_2step = x_train_selected_1step[:, indices]

# select final features from various combinations within the set of 11 numbers
num_cols = x_train_selected_2step.shape[1]
all_combinations = []
for i in range(1, num_cols + 1):
    all_combinations.extend(combinations(range(num_cols), i))
Feature_combinations = []
Feature_number = []
R2_train = []
MSE_train = []
for comb in all_combinations:
    GBR.fit(x_train_selected_2step[:, comb], y_train)
    Feature_number.append(len(comb))
    Feature_combinations.append([feature_names_es[i] for i in comb])
    R2_train.append(GBR.score(x_train_selected_2step[:, comb], y_train))
    MSE_train.append(mean_squared_error(y_train, GBR.predict(x_train_selected_2step[:, comb])))
    print(f'Features combination: {[feature_names_es[i] for i in comb]}')
    print("r2 = %0.3f, MSE = %0.3f" % (GBR.score(x_train_selected_2step[:, comb], y_train),
                                       mean_squared_error(y_train, GBR.predict(x_train_selected_2step[:, comb]))))
ES_res = pd.DataFrame({"Feature_number": Feature_number, "Feature_combination": Feature_combinations,
                       "R2_train": R2_train, "MSE_train": MSE_train})

# %% hyperparameters optimization of models
feature_final = ['Tm', 'δr', 'γ', 'δχ', 'MM', 'δMM', 'S', 'BMN']
final_indices = [0, 2, 3, 5, 6, 7, 9, 10]
x_train_final = x_train_selected_2step[:, final_indices]
model = [SVR, LinearR, LinearElastic, RandomF, MLP, GBR, AdaBoost, DT, Knn, XGB, GPR]
model_name = ['SVR', 'LinearR', 'LinearElastic', 'RandomF', 'MLP', 'GBR', 'AdaBoost', 'DT', 'Knn', 'XGB', 'GPR']
parameters = [
    # SVR
    {
        'C': [1e-3, 0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 'scale'],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    },
    # LinearRidge
    {
        'alpha': [1e-2, 1e-1, 1, 10, 100]
    },
    # LinearElastic
    {
        'alpha': [1e-2, 1e-1, 1, 10, 100],
        'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    },
    # RandomForest
    {
        'n_estimators': [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, ],
        'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'max_features': [1, 2, 3, 4, 5, 6, 7, 8]
    },
    # MLP
    {
        'alpha': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        'hidden_layer_sizes': [(50,), (100,),
                               (50, 100), (50, 50), (100, 100), (100, 50),
                               (50, 50, 50), (100, 100, 100), (100, 50, 50), (50, 100, 50),
                               (50, 50, 100), (50, 100, 100), (100, 50, 100), (100, 100, 50)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
    },
    # GBR
    {
        'n_estimators': [10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
        'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'max_features': [1, 2, 3, 4, 5, 6, 7, 8]
    },
    # Adaboost
    {
        'n_estimators': [10, 50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2, 0.5, 0.9, 1]
    },
    # DT
    {
        'max_depth': [1, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 5, 10, 15, 20]
    },
    # KNN
    {
        'n_neighbors': [1, 5, 10, 20, 30]
    },
    # XGB
    {
           'n_estimators': [100, 200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 6, 7, 9],
        'min_child_weight': [1, 3, 5, 7],
    },
    # GPR
    {
        'kernel': [RBF(), Matern()],
        'alpha': [1e-10, 1e-5, 1e-2, 1]
    }
]
for i in range(len(model)):
    clf = GridSearchCV(model[i], param_grid=parameters[i], cv=5)
    clf.fit(x_train_final, y_train, groups=group)
    print("Best parameters and score set found on development set of %s:" % model_name[i])
    print(clf.best_params_)

# Best parameters and score set found on development set of SVR:
# {'C': 100, 'epsilon': 0.001, 'gamma': 0.01}
# Best parameters and score set found on development set of LinearR:
# {'alpha': 100}
# Best parameters and score set found on development set of LinearElastic:
# {'alpha': 0.1, 'l1_ratio': 0.1}
# Best parameters and score set found on development set of RandomF:
# {'max_depth': 10, 'max_features': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 30}
# Best parameters and score set found on development set of MLP:
# {'activation': 'logistic', 'alpha': 1, 'hidden_layer_sizes': (100, 50, 50)}
# Best parameters and score set found on development set of GBR:
# {'max_depth': 4, 'max_features': 3, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 100}
# Best parameters and score set found on development set of AdaBoost:
# {'learning_rate': 0.9, 'n_estimators': 200}
# Best parameters and score set found on development set of DT:
# {'max_depth': 10, 'min_samples_leaf': 20, 'min_samples_split': 2}
# Best parameters and score set found on development set of Knn:
# {'n_neighbors': 5}
# Best parameters and score set found on development set of XGB:
# {'learning_rate': 0.05, 'max_depth': 5, 'min_child_weight': 3, 'n_estimators': 200}
# Best parameters and score set found on development set of GPR:
# {'alpha': 1, 'kernel': Matern()}

# %% final models' performance
SVR_final = svm.SVR(C=100, epsilon=0.001, gamma=0.01)
LinearR_final = linear_model.Ridge(random_state=42, alpha=100)
LinearElastic_final = linear_model.ElasticNet(random_state=42, alpha=0.1, l1_ratio=0.1)
RandomF_final = RandomForestRegressor(random_state=42, max_depth=10, max_features=4, min_samples_leaf=1,
                                      min_samples_split=2, n_estimators=30)
MLP_final = MLPRegressor(random_state=42, activation='logistic', alpha=1, hidden_layer_sizes=(100, 50, 50),
                         max_iter=1000)
GBR_final = GradientBoostingRegressor(random_state=42, max_depth=4, max_features=3, min_samples_leaf=1,
                                      min_samples_split=8, n_estimators=100)
AdaBoost_final = AdaBoostRegressor(random_state=42, learning_rate=0.9, n_estimators=200)
DT_final = DecisionTreeRegressor(random_state=42, max_depth=10, min_samples_leaf=20, min_samples_split=2)
Knn_final = KNeighborsRegressor(n_neighbors=5)
XGB_final = XGBRegressor(learning_rate=0.05, max_depth=5, min_child_weight=3, n_estimators=200)
GPR_final = GaussianProcessRegressor(random_state=42, alpha=1, kernel=Matern())
x_test = pd.read_excel('dataset.xlsx', usecols='A,D:X',
                       sheet_name='test', index_col=0)
y_test = pd.read_excel('dataset.xlsx', usecols='A,C',
                       sheet_name='test', index_col=0)

y_test = np.array(y_test).ravel()
x_test = scaler.transform(x_test)
x_test_1step = np.delete(x_test, 0, axis=1)
indices = [0, 1, 3, 4, 5, 6, 11, 12, 13, 14, 16]
final_indices = [0, 2, 3, 5, 6, 7, 9, 10]
x_test_2step = x_test_1step[:, indices]
x_test_final = x_test_2step[:, final_indices]

model_final = [SVR_final, LinearR_final, LinearElastic_final, RandomF_final, MLP_final,
               GBR_final, AdaBoost_final, DT_final, Knn_final, XGB_final, GPR_final]
model_name = ['SVR', 'LinearR', 'LinearElastic', 'RandomF', 'MLP', 'GBR', 'AdaBoost', 'DT', 'Knn', 'XGB', 'GPR']

for i in range(len(model_final)):
    model_final[i].fit(x_train_final, y_train)
    print(model_name[i])
    print('R2_train = %0.3f, MSE_train = %0.3f'
          % (r2_score(y_train, model_final[i].predict(x_train_final)),
             mean_squared_error(y_train, model_final[i].predict(x_train_final))))
    print('R2_test = %0.3f, MSE_test = %0.3f'
          % (
              r2_score(y_test, model_final[i].predict(x_test_final)),
              mean_squared_error(y_test, model_final[i].predict(x_test_final))))

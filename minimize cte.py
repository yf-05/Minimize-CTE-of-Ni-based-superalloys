import math
import geatpy as ea
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model

# %% input and general model fitting
x_train_general = pd.read_excel('dataset.xlsx', usecols='A,D:I',
                                sheet_name='train_general', index_col=0)
y_train_general = pd.read_excel('dataset.xlsx', usecols='A,C',
                                sheet_name='train_general', index_col=0)
x_train_general = np.array(x_train_general)
y_train_general = np.array(y_train_general).ravel()
scaler_general = preprocessing.StandardScaler()
scaler_general.fit(x_train_general)
x_train_general = scaler_general.transform(x_train_general)
LinearElastic_general = linear_model.ElasticNet(random_state=42, alpha=0.015, l1_ratio=0.05)
LinearElastic_general.fit(x_train_general, y_train_general)
feature_general = np.array(pd.read_excel('dataset.xlsx',
                                         sheet_name='feature-general', index_col=0))
x_original = np.array(pd.read_excel('dataset.xlsx', usecols='A:U',
                                    sheet_name='Sheet1', index_col=0))


# %% geatpy optimize CTE#

# function to generate features for fitness function LinearElastic
def generate_feature(component_at):
    feature = [900]
    L10 = 0
    for i in component_at:
        L10 += math.pow(i, 10)
    feature.append(math.pow(L10, 0.1))
    for i in range(4):
        mean = sum(component_at * feature_general[i])
        AD = sum(abs(feature_general[i] - mean) * component_at)
        feature.append(AD)
    feature = np.array(feature).reshape(1, -1)
    feature = scaler_general.transform(feature)
    return feature


ub = np.array([0.04, 0.4, 1, 0.3, 0.07, 0.3, 0.4, 0.04, 0.06, 0.01, 0.06, 0.2, 0, 0.03, 0, 0, 0.15, 0, 0, 0])


# upper boundaries of [C, Cr, Ni, Co, W, Mo, Fe, Nb, B, V, Ti, Al, Zr, Ta, Hf, Re, Si, Cu, Y, Mn]


@ea.Problem.single
def func(component_at):
    total = sum(component_at)
    component_at = np.array(component_at)
    component_at = component_at / total
    x_in = generate_feature(component_at)
    y = LinearElastic_general.predict(x_in)
    at_lb = np.array([max(component_at) - component_at[2], 0.3 - component_at[2], ])
    at_ub = component_at - ub
    CV = np.append(
        at_lb,
        at_ub
    )
    return y[0], CV


problem = ea.Problem(name='Minimize CTE',
                     M=1,  # only one object
                     maxormins=[1],  # 1 means minimization
                     Dim=20,  # 20 vars
                     varTypes=[0 for i in range(20)],  # 0 means real and 1 means integer
                     lb=[0 for i in range(20)],
                     ub=[1 for i in range(20)],
                     evalVars=func
                     )

pop = ea.Population(Encoding='RI', NIND=100)  # 100 individuals in every population

SEGA = ea.soea_SEGA_templet(
    problem=problem,
    population=pop,
    MAXGEN=100,
    logTras=1,
)

res = ea.optimize(
    SEGA, seed=1, verbose=True, drawing=1, drawLog=True, outputMsg=True, saveFlag=False,
    prophet=x_original  # original population: collected data of Ni-based superalloys
)

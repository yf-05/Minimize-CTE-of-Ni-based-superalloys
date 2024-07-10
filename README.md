# Minimize-CTE-of-Ni-based-superalloys
Key codes and data of 'Discovery of a Ni-based Superalloy with Low Thermal Expansion via Machine Learning'

## Models for Predicting CTE
A three-step feature engineering process after features construction:   
(1) Correlation Screening, where less important features are removed if two features exhibit a correlation coefficient >0.9;   
(2) Recursive Elimination, where the least important feature is iteratively removed, starting from the maximum number of features, and proceeding to the minimum. We retained the optimal number, determined to be 11;   
(3) Exhaustive Screening, where the final feature collection, comprising 8 features, was selected from various combinations within the set of 11 numbers. This process aimed to refine the feature set, recognizing that a complex feature set often decrease model performance.   
  
Hyperparameters optimization via 5-fold gridCV:  
11 regression models: Support Vector Regression(SVR), LinearRidge(LR), LinearElastic(LE), RandomForest(RF), Multilayer Perceptron(MLP), Gradient Boosting Regression(GBR), AdaBoost(AD), Decision Tree(DT), K-nearest Neighbor(KNN), XGBoost(XGB), Gaussian Process Regression(GPR) are optimized and compare their performance in test-set. Finally, the SVR is chosen as the final model.  
  
## Minimize CTE  
A LE model with general features is chosen as the fitness function of GA via the same flow above, because general features can be generated faster without features from CALPHAD. After first screening via the LE model and GA, the optimized compositions are further chosen via the SVR model which is more accurate than the LE model. 



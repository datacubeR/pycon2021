
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.encoding import OneHotEncoder

def simple_preprocess(imp_method_values, trial=None):
    preprocess = Pipeline(steps = [
        ('imp_num', MeanMedianImputer(imputation_method=trial.suggest_categorical("imp_num", imp_method_values))),
        ('sc', StandardScaler())
    ])
    return preprocess

def complex_preprocess(imputation_num, imputation_cat, trial=None):
    preprocess = Pipeline(steps = [
        ('imp_num', MeanMedianImputer(imputation_method=trial.suggest_categorical("imp_num", imputation_num))),
        ('imp_cat', CategoricalImputer(imputation_method=trial.suggest_categorical("imp_cat", imputation_cat))),
        ('ohe', OneHotEncoder()),
        ('sc', StandardScaler())
    ])
    return preprocess
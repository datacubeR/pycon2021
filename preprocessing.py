from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.encoding import OneHotEncoder

def simple_preprocess(imputation_method):
    preprocess = Pipeline(steps = [
        ('imp_num', MeanMedianImputer(imputation_method=imputation_method)),
        ('sc', StandardScaler())
    ])
    return preprocess

def complex_preprocess(imputation_num, imputation_cat):
    preprocess = Pipeline(steps = [
        ('imp_num', MeanMedianImputer(imputation_method=imputation_num)),
        ('imp_cat', CategoricalImputer(imputation_method=imputation_cat)),
        ('ohe', OneHotEncoder()),
        ('sc', StandardScaler())
    ])
    return preprocess

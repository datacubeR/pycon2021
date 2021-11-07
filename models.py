from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def random_forest(n_estimators, max_depth, min_samples_split, trial = None):
    return RandomForestClassifier(n_estimators = trial.suggest_int('n_estimators', **n_estimators),
                                max_depth = trial.suggest_int('max_depth', **max_depth),
                                min_samples_split = trial.suggest_discrete_uniform('min_samples_split', **min_samples_split),
                                n_jobs = -1,
                                random_state = 123
                                )
    
def log_reg(C, fit_intercept, trial = None):
    return LogisticRegression(C = trial.suggest_loguniform('C', **C),
                            fit_intercept=trial.suggest_categorical('fit_intercept', fit_intercept),
                            n_jobs = -1,
                            random_state = 123
                            )

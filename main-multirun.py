import optuna
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from models import random_forest, log_reg
from preprocessing_multirun import simple_preprocess, complex_preprocess
from optuna.samplers import TPESampler
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path

@hydra.main(config_name='config', config_path='conf')
def train_model(cfg: DictConfig):
    train_df_small = pd.read_csv(to_absolute_path(cfg.preprocess.data_train))
    test_df_small = pd.read_csv(to_absolute_path(cfg.preprocess.data_test))
    X_train, y_train = train_df_small.drop(columns = 'Survived'), train_df_small.Survived
    X_test, y_test = test_df_small.drop(columns = 'Survived'), test_df_small.Survived

    def optimize_model(trial):
        preprocess = hydra.utils.call(cfg.preprocess.type, trial = trial)
        
        model_pipe = Pipeline(steps = [
            ('prep', preprocess),
            ('model', hydra.utils.call(cfg.models.type, trial = trial))
        ])

        model_pipe.fit(X_train,y_train)
        y_pred = model_pipe.predict(X_test)
        return accuracy_score(y_test, y_pred)

    sampler = TPESampler(seed=123)
    study = optuna.create_study(sampler = sampler, direction="maximize")
    study.optimize(optimize_model, n_trials=cfg.n_trials)

    print(f'El mejor accuracy conseguido fue: {study.best_value}')
    print(f'usando los siguientes parámetros: \n \t \t{study.best_params}')

if __name__ == '__main__':
    train_model()
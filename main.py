from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
import pandas as pd
from preprocessing import simple_preprocess, complex_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

@hydra.main(config_path ='conf', config_name = 'config')
def train_model(cfg: DictConfig):
    train_df_small = pd.read_csv(to_absolute_path(cfg.preprocess.data_train))
    test_df_small = pd.read_csv(to_absolute_path(cfg.preprocess.data_test))
    X_train, y_train = train_df_small.drop(columns = 'Survived'), train_df_small.Survived
    X_test, y_test = test_df_small.drop(columns = 'Survived'), test_df_small.Survived
    
    preprocess = hydra.utils.call(cfg.preprocess.type)
    
    model_pipe = Pipeline(steps = [
        ('prep', preprocess),
        ('model', hydra.utils.instantiate(cfg.models.type))
    ])
    
    model_pipe.fit(X_train,y_train)
    y_pred = model_pipe.predict(X_test)

    print(f"El Accuracy obtenido por {cfg.models.name} es:", accuracy_score(y_test, y_pred))

    
if __name__ == '__main__':
    train_model()
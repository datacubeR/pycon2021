import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engine.imputation import MeanMedianImputer
from sklearn.metrics import accuracy_score

train_df_small = pd.read_csv('data/data-train-small.csv')
test_df_small = pd.read_csv('data/data-test-small.csv')
X_train, y_train = train_df_small.drop(columns = 'Survived'), train_df_small.Survived
X_test, y_test = test_df_small.drop(columns = 'Survived'), test_df_small.Survived


preprocess = Pipeline(steps = [
    ('imp_num', MeanMedianImputer(imputation_method='mean')),
    ('sc', StandardScaler())
])

model_pipe = Pipeline(steps = [
    ('prep', preprocess),
    ('model', RandomForestClassifier(
        n_estimators = 100, 
        max_depth = 5,
        min_samples_split = 10,
        random_state=123))
])

model_pipe.fit(X_train,y_train)
y_pred = model_pipe.predict(X_test)

print("El accuracy obtenido es:", accuracy_score(y_test, y_pred))



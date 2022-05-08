import click
import pandas as pd
import joblib as jb
from catboost import Pool, CatBoostClassifier
from typing import List
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, roc_auc_score
import mlflow
import os

os.environ['MLFLOW_TRACKING_USERNAME'] = "RyazaNow"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "abfac14e692ddb7ac8d2ed69487440f035bd0aa7"

mlflow.set_tracking_uri('https://dagshub.com/RyazaNow/Deploy-ML-Model.mlflow')
mlflow.set_experiment('catboost')

@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path())
def train(input_paths: List[str], output_path: str):

    train_df = pd.read_csv(input_paths[0])
    test_df = pd.read_csv(input_paths[1])

    X = list(train_df.drop(['Оператор'], axis=1).columns)
    cat_features_names = list(train_df[X].select_dtypes('object').columns)


    train_pool_operator = Pool(train_df[X],
                           label= train_df['Оператор'],
                           cat_features= cat_features_names )

    valid_pool_operator = Pool(test_df[X],
                           label= test_df['Оператор'],
                           cat_features= cat_features_names )

    params = { 'verbose':False,
               'eval_metric':'F1',
               'loss_function': 'Logloss',
               'random_seed': 2000,
               'learning_rate': 0.01,
               'auto_class_weights': 'SqrtBalanced',
               'leaf_estimation_method': 'Newton',
               'bootstrap_type':'Bernoulli'}

    model = CatBoostClassifier(**params)

    catboost = model.fit(train_pool_operator,eval_set=valid_pool_operator)
    jb.dump(catboost, output_path)

    # mlFlow logging params
    x_holdout = test_df.drop('Оператор', axis=1)
    y_holdout = test_df['Оператор']

    pred = model.predict(x_holdout)

    accuracy_score(y_holdout, pred)
    score = dict(
        accuracy=accuracy_score(y_holdout, pred),
        f1_scrore=f1_score(y_holdout, pred),
        roc_auc=roc_auc_score(y_holdout, pred)
    )

    mlflow.log_params(params)
    mlflow.log_metrics(score)



    #mlflow.catboost.save_model(cb_model=model, path='models/model_cat')
    mlflow.catboost.log_model(cb_model=model, artifact_path='models/model_cat')

if __name__ == "__main__":
    train()

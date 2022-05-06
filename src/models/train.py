import click
import pandas as pd
import joblib as jb
from catboost import Pool, CatBoostClassifier
from typing import List
from sklearn.metrics import mean_absolute_error, mean_squared_error


@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path())
def train(input_paths: List[str], output_path: str):

    train_df = pd.read_csv(input_paths[0])
    test_df = pd.read_csv(input_paths[1])

    cat_features_names = list(train_df.select_dtypes('object').columns)
    X = list(train_df.drop(['Оператор'], axis=1).columns)
    cat_features_names = list(train_df[X].select_dtypes('object').columns)


    train_pool_operator = Pool(train_df[X],
                           label= train_df['Оператор'],
                           cat_features= cat_features_names )

    valid_pool_operator = Pool(test_df[X],
                           label= test_df['Оператор'],
                           cat_features= cat_features_names )

    params = { 'verbose':200,
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

if __name__ == "__main__":
    train()

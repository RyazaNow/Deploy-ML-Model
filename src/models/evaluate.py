import click
import pandas as pd
import joblib as jb
from typing import List
from sklearn.metrics import accuracy_score, confusion_matrix,  roc_auc_score, f1_score


@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path())
def evaluate(input_paths: List[str], output_path: str):
    test_df = pd.read_csv(input_paths[0])
    model = jb.load(input_paths[1])


    x_holdout = test_df.drop('Оператор', axis=1)
    y_holdout = test_df['Оператор']

    pred = model.predict(x_holdout)

    accuracy_score(y_holdout, pred)
    index = ['Accuracy', 'F1_Score', 'AUC_ROC']
    score = pd.DataFrame({'Оператор':[accuracy_score(y_holdout, pred),
                                      f1_score(y_holdout, pred),
                                      roc_auc_score(y_holdout, pred)]}, index=index)

    score.to_csv(output_path, index=False)



if __name__ == "__main__":
    evaluate()

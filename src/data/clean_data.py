import pandas as pd
import click


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(exists=True))
def clean_data(input_path: str, output_path: str):
    '''
    :param input_path:
    :param output_path:
    '''

    df = pd.read_csv(input_path)

    df = df.drop(['hhid', 'id'], axis=1)
    numeric_columns = df.drop('Почтовый индекс', axis=1).select_dtypes(include=['int'])


    for i in numeric_columns:
        Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    up_limit = Q3 + 4 * IQR
    lim_max = df[i].max() * 0.7
    if df[i].median() == 0:
        df[i] = df[i].apply(lambda x: x if x < lim_max else lim_max)
    else:
        df[i] = df[i].apply(lambda x: x if x < up_limit else df[i].median())

    df = df.dropna()
    df = df.drop_duplicates()
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    clean_data()
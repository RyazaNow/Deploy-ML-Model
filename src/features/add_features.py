import pandas as pd
import click

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def add_features(input_path: str, output_path: str):
    '''
    :param input_path:
    :param output_path:
    '''
    df = pd.read_csv(input_path)

    df['Почтовый индекс'] = df['Почтовый индекс'].astype('str')
    df['Почтовый индекс'] = df['Почтовый индекс'].apply(lambda x:x[:2])
    df['Оператор'] = df['Оператор'].apply(lambda x:1 if x == 'Подключен только к Триколор ТВ' else 0)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    add_features()
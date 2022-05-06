import click
import pandas as pd
from typing import List


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_file_paths", type=click.Path(), nargs=2)
def prepare_datasets(input_filepath, output_file_paths: List[str]):

    df = pd.read_csv(input_filepath)
    train = df.sample(frac=0.75, random_state=200)
    test = df.drop(train.index)

    train.to_csv(output_file_paths[0], index=False)
    test.to_csv(output_file_paths[1], index=False)


if __name__ == "__main__":
    prepare_datasets()

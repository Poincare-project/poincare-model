import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
from gensim.models.poincare import (LinkPredictionEvaluation, PoincareModel,
                                    PoincareRelations,
                                    ReconstructionEvaluation)
from tqdm import tqdm

DATA_DIRECTORY = os.path.join(os.getcwd(), "data")
OUTPUT_DIRECTORY = os.path.join(os.getcwd(), "outputs")


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def save_bench_mark(bench_mark_df):
    bench_mark_df.to_csv(os.path.join(OUTPUT_DIRECTORY, "bench_mark.csv"), index=False)
    # plot table of results
    formatted_df = bench_mark_df.round(3)
    # size and negative should integer columns
    formatted_df["size"] = formatted_df["size"].astype(int)
    formatted_df["negative"] = formatted_df["negative"].astype(int)

    column_mapping = {
        "size": "Dimension",
        "negative": "Negative samples",
        "mean_rank": "Mean Rank",
        "map": "MAP",
    }
    formatted_df = formatted_df.rename(columns=column_mapping)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.table(cellText=formatted_df.values, colLabels=formatted_df.columns, loc="center")
    ax.axis("off")
    ax.set_title("Poincare Bench Mark")
    plt.show()
    fig.savefig(os.path.join(OUTPUT_DIRECTORY, "bench_mark.png"))


def generate_bench_mark(train_relation_file_path):
    relations = PoincareRelations(train_relation_file_path)

    size_set = [5, 10, 20, 50, 100, 200]
    negative_set = [5, 10, 20]
    epochs_set = [50]

    parameters_grid = list(
        product_dict(epochs=epochs_set, size=size_set, negative=negative_set)
    )
    model_evaluations = []

    for parameters in tqdm(parameters_grid):
        current_size = parameters["size"]
        negative = parameters["negative"]
        epochs = parameters["epochs"]

        model = PoincareModel(relations, size=current_size, negative=negative)
        model.train(epochs=epochs, print_every=1, batch_size=10)

        reconstruction_evaluation = ReconstructionEvaluation(
            file_path=train_relation_file_path, embedding=model.kv
        )
        mean_rank, map_value = reconstruction_evaluation.evaluate().values()
        eval_result = dict(
            size=current_size, negative=negative, mean_rank=mean_rank, map=map_value
        )
        model_evaluations.append(eval_result)

    bench_mark_df = pd.DataFrame(model_evaluations)
    save_bench_mark(bench_mark_df)


if __name__ == "__main__":
    relations_file_path = os.path.join(DATA_DIRECTORY, "relations.tsv")
    generate_bench_mark(relations_file_path)

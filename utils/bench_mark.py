import os
import itertools
from tqdm import tqdm
from gensim.models.poincare import PoincareModel, PoincareRelations, LinkPredictionEvaluation,LexicalEntailmentEvaluation
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIRECTORY = os.path.join(os.getcwd(), 'data')
OUTPUT_DIRECTORY = os.path.join(os.getcwd(), 'outputs')

def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def save_bench_mark(bench_mark_df):
    bench_mark_df.to_csv(os.path.join(OUTPUT_DIRECTORY, 'bench_mark.csv'), index=False)
    # plot table of results
    formatted_df = bench_mark_df.round(3)
    # size and negative should integer columns
    formatted_df[['size', 'negative']] = formatted_df[['size', 'negative']].astype(int)
    column_mapping = {'size': 'Dimension', 'negative': 'Negative samples', 'mean_rank': 'Mean Rank', 'map': 'MAP', 'spearman': 'Spearman coefficient'}
    formatted_df = formatted_df.rename(columns=column_mapping)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    
    ax.table(cellText=formatted_df.values, colLabels=formatted_df.columns, loc='center')
    ax.axis('off')
    ax.set_title('Poincare Bench Mark')
    plt.show()
    fig.savefig(os.path.join(OUTPUT_DIRECTORY, 'bench_mark.png'))
    
def generate_bench_mark(train_relation_file_path,test_relation_file_path):
    relations = PoincareRelations(train_relation_file_path)
    
    size_set= [5,10,20,50, 100, 200]
    negative_set = [5, 10, 20]
    epochs_set = [50]
    
    parameters_grid = list(product_dict(epochs=epochs_set, size=size_set, negative=negative_set))
    model_evaluations = []
    
    for parameters in tqdm(parameters_grid):
        current_size = parameters['size']
        negative = parameters['negative']
        epochs = parameters['epochs']
        
        model = PoincareModel(relations, size=current_size, negative=negative)
        model.train(epochs=epochs, print_every=1,batch_size=10)
        
        lp_evaluation = LinkPredictionEvaluation(train_relation_file_path,test_relation_file_path, model.kv)
        lexical_evaluation = LexicalEntailmentEvaluation(train_relation_file_path)
        spearman = lexical_evaluation.evaluate_spearman(model.kv)
        mean_rank, map_value = lp_evaluation.evaluate().values()
        eval_result = dict(size=current_size, negative=negative, mean_rank=mean_rank, map=map_value,spearman=spearman)
        model_evaluations.append(eval_result)

    bench_mark_df = pd.DataFrame(model_evaluations)
    save_bench_mark(bench_mark_df)

if __name__ == "__main__":
    
    
    train_relations = os.path.join(DATA_DIRECTORY, 'train_relation.tsv')
    test_relations = os.path.join(DATA_DIRECTORY, 'test_relation.tsv')
    
    generate_bench_mark(train_relations,test_relations)
    
    
    
    

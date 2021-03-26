import networkx as nx
import pandas as pd
import numpy as np
import node2vec
import argparse

cora_address = 'D:/yuyicong/workspace/cora/cora/'
cora_content = 'cora.content'
cora_cite = 'cora.cites'
def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")
    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window_size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')
    
    parser.add_argument('--embed-size',type=int,default=128,
                        help='Number of embedding size. Default is 128.')
    
    parser.add_argument('--min-count',type=int,default=0,
                        help='Number of min count of word. Default is 0.')
    
    parser.add_argument('--sg',type=int,default = 1,
                        help='Skip-gram/Cbow,0 is cbow and 1 is sg. Default is 1')
    
    parser.add_argument('--hs',type=int,default = 1,
                        help='use Hierarchical Softmax or not, 1 yes, 0 no. Default is 1')
    parser.set_defaults(directed=False)
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    cora_edge = pd.read_table(cora_address+cora_cite,sep='\t',names = ['src','dst'])
    G = nx.Graph()
    node_dict = {i:v for v,i in enumerate(set(np.append(cora_edge.src.values,cora_edge.dst.values)))}
    cora_edge['src'] = cora_edge['src'].apply(lambda x: str(node_dict[x]))
    cora_edge['dst'] = cora_edge['dst'].apply(lambda x: str(node_dict[x]))
    cora_edge_list = cora_edge.values.tolist()
    def map_func(x):
        return (x[0],x[1],{'weight':1})
    cora_edge_list = list(map(map_func,cora_edge_list))

    G.add_edges_from(cora_edge_list)
    n2v = node2vec.Node2Vec(G,args.p,args.q)
    n2v.train(num_walks = args.num_walks ,walk_length = args.walk_length, embed_size = args.embed_size, 
              window_size = args.window_size, workers = args.workers, iter_num = args.iter, min_count = args.min_count,
             sg = args.sg, hs = args.hs)
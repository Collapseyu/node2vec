import networkx as nx
import pandas as pd
import numpy as np
import random
from gensim.models import Word2Vec

# 别名采样 

def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

class Node2Vec:
    def __init__(self,G,p=1,q=1):
        self.G = G
        self.p = p
        self.q = q
    def get_alias_edge(self, src, dst):
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)
    def preprocess_transition_probs(self):
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            # 
            unnormalized_probs = [G[node][nbr].get('weight', 1.0)
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)
        alias_edges = {}
        triads = {}
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        return
    def node2vec_walk(self,walk_length,start_node):
        G = self.G
        walk_list = [start_node]
        while(len(walk_list)<walk_length):
            curNode = walk_list[-1]
            curNeighbor = list(G.neighbors(curNode))
            if len(curNeighbor) > 0:
                if len(walk_list) == 1:
                    walk_list.append(curNeighbor[alias_draw(self.alias_nodes[curNode][0], self.alias_nodes[curNode][1])])
                else:
                    prev = walk_list[-2]
                    next_Node = curNeighbor[alias_draw(self.alias_edges[(prev, curNode)][0], 
                        self.alias_edges[(prev, curNode)][1])]
                    walk_list.append(next_Node)
            else:
                break
#         print(walk_list)
        return walk_list
    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        self.walks = walks
        return walks
    def train(self,num_walks = 10 ,walk_length = 10, embed_size = 128, 
              window_size = 5, workers = 3, iter_num = 5, min_count = 0,
             sg = 1, hs = 0):
        self.preprocess_transition_probs()
        _ = self.simulate_walks(num_walks,walk_length)
        param = {}
        param["sentences"] = self.walks
        param["min_count"] = min_count
        param["size"] = embed_size
        param["sg"] = sg
        param["hs"] = hs  
        param["workers"] = workers
        param["window"] = window_size
        param["iter"] = iter_num
        
        print('start word2vec training ...')
        model = Word2Vec(**param)
        print('done')
        self.model = model
        return model
    def get_embedding(self):
        if self.model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.G.nodes():
            self._embeddings[word] = self.model.wv[word]

        return self._embeddings
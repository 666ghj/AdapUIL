import networkx as nx 
import numpy as np 
import random
from gensim.models import Word2Vec
import torch

# torch.manual_seed()
random.seed()
np.random.seed()
class DeepWalk:
    def __init__(self, G, emb_size=256, length_walk=40, num_walks=10, window_size=10, num_iters=1):
        self.G = G
        self.emb_size = emb_size
        self.length_walk = length_walk
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_iters = num_iters
        random.seed()

    def random_walk(self):
        # random walk with every node as start point once
        random.seed()
        walks = []
        for node in self.G.nodes():
            walk = [str(node)]
            v = node
            for _ in range(self.length_walk):
                nbs = list(self.G.neighbors(v))
                if len(nbs) == 0:
                    break
                v = random.choice(nbs)
                walk.append(str(v))
            walks.append(walk)

        return walks


    def sentenses(self):
        random.seed()
        sts = []
        for _ in range(self.num_walks):
            sts.extend(self.random_walk())

        return sts


    def train(self, workers=4, is_loadmodel=False, is_loaddata=False,node_type="source"):
        random.seed()
        if is_loadmodel:
            print('Load model from file')
            if node_type ==  "source":
                w2v = Word2Vec.load('../models/source.model')
            else:
                w2v = Word2Vec.load('../models/target.model')
            return w2v

        if is_loaddata:
            print('Load data from file')
            if node_type == "source":
                with open('../data/dpsource.txt', 'r') as f:
                    sts = f.read()
                    sentenses = eval(sts)
            else:
                with open('../data/dptarget.txt', 'r') as f:
                    sts = f.read()
                    sentenses = eval(sts)
        else:
            print('Random walk to get training data...')
            print(node_type)
            sentenses = self.sentenses()

            print('Number of sentenses to train: ', len(sentenses))
            if node_type ==  "source":
                with open('dataspace/douban/source.txt', 'w') as f:
                    f.write(str(sentenses))
            else:
                with open('dataspace/douban/target.txt', 'w') as f:
                    f.write(str(sentenses))

        print('Start training...')
        random.seed()
        w2v = Word2Vec(sentences=sentenses, vector_size=self.emb_size, window=self.window_size, epochs=self.num_iters, sg=1, hs=1, min_count=0, workers=workers)
        if node_type == "source":
            w2v.save('models/douban/source.model')
        else:
            w2v.save('models/douban/target.model')
        print('Training Done.')
        
        return w2v



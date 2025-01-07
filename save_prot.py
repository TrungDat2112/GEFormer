import os
import pickle
import pandas as pd
from torch_geometric import data as DATA
from utils import *
from collate import *
from dataset import *
from data_process import prot_to_graph
from metrics import *

compound_iso_smiles = []
pdbs = []
pdbs_seqs = []
all_labels = []

df = pd.read_csv('davis/split/trainer.csv')
compound_iso_smiles += list(df['compound_iso_smiles'])
pdbs += list(df['target_name'])
pdbs_seqs += list(df['target_sequence'])
all_labels += list(df['affinity'])
pdbs_tseqs = set(zip(pdbs, pdbs_seqs, compound_iso_smiles, all_labels))

dta_graph = {}
print('Pre-processing protein')
saved_prot_graph = {}

for target, seq in set(zip(pdbs, pdbs_seqs)):
    if os.path.isfile('davis/map/' + target + '.npy'):
        contactmap = np.load('davis/map/' + target + '.npy')
    else:
        raise FileNotFoundError
    c_size, features, edge_index, edge_weight = prot_to_graph(seq, contactmap, target)
    g = DATA.Data(
        x = torch.tensor(np.array(features), dtype=torch.float32),
        edge_index=torch.LongTensor(edge_index).transpose(1, 0),
        edge_attr=torch.FloatTensor(edge_weight),
        prot_len=c_size
    )
    saved_prot_graph[target] = g
with open('saved_prot_graph.pickle', 'wb') as handle:
    pickle.dump(saved_prot_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

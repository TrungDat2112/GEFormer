import pickle
import pandas as pd
from torch_geometric import data as DATA
from utils import *
from collate import *
from dataset import *
from data_process import smile_to_graph, drug_embedding
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

output_file = "saved_drug_graph.pickle"
with open(output_file, 'ab') as handle:
    for smiles in compound_iso_smiles:
        c_size2, features2, edge_index2 = smile_to_graph(smiles)
        d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input = drug_embedding(smiles)
        g2 = DATA.Data(
            x=torch.tensor(np.array(features2)),
            edge_index=torch.LongTensor(edge_index2).transpose(1, 0),
            node=d_node,
            attn_bias=d_attn_bias,
            spatial_pos=d_spatial_pos,
            in_degree=d_in_degree,
            out_degree=d_out_degree,
            edge_input=d_edge_input
        )
        pickle.dump({smiles: g2}, handle, protocol=pickle.HIGHEST_PROTOCOL)


print('load success!')
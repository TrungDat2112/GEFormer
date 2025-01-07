import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import heapq


def pad_attn_bias_unsqueeze(tensor, max_d_node):
    current_size = tensor.size(0)
    if current_size > max_d_node:
        tensor = tensor[:max_d_node, :max_d_node]
    pad_size = max(0, max_d_node - current_size)
    padded_tensor = F.pad(tensor, (0, pad_size, 0, pad_size), value=float('-inf'))
    padded_tensor = padded_tensor.unsqueeze(0)
    return padded_tensor




def pad_spatial_pos_unsqueeze(tensor, max_d_node):
 
    current_size = tensor.size(0)
    pad_size = max_d_node - current_size
    padded_tensor = F.pad(tensor, (0, pad_size, 0, pad_size), value=0)
    padded_tensor = padded_tensor.unsqueeze(0)
    
    return padded_tensor

def pad_1d_unsqueeze(tensor, max_d_node):
    current_size = tensor.size(0)
    pad_size = max_d_node - current_size
    padded_tensor = F.pad(tensor, (0, pad_size), value=0)
    padded_tensor = padded_tensor.unsqueeze(0)

    return padded_tensor

def pad_4d_unsqueeze(tensor, max_dim1, max_dim2, max_drug_dist):
    dim1, dim2, current_dist, _ = tensor.size()
    pad_dim1 = max_dim1 - dim1
    pad_dim2 = max_dim2 - dim2
    pad_dist = max_drug_dist - current_dist
    padded_tensor = F.pad(tensor, (0, 0, 0, pad_dist, 0, pad_dim2, 0, pad_dim1), value=0)
    padded_tensor = padded_tensor.unsqueeze(0)

    return padded_tensor

def pad_2d_unsqueeze(tensor, max_dim):
 
    current_dim = tensor.size(0)
    pad_size = max_dim - current_dim
    padded_tensor = F.pad(tensor, (0, 0, 0, pad_size), value=0)
    padded_tensor = padded_tensor.unsqueeze(0)

    return padded_tensor


def bond_to_feature_vector(bond):
    bond_type = bond.GetBondTypeAsDouble()
    bond_stereo = bond.GetStereo()
    is_conjugated = bond.GetIsConjugated()
    features = np.array([
        bond_type,
        bond_stereo,
        is_conjugated
    ], dtype=np.float32)

    return features

def atom_features(atom):
    atom_type = atom.GetAtomicNum()
    chirality = atom.GetChiralTag()
    num_bonds = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    num_hydrogens = atom.GetTotalNumHs()
    num_radical_electrons = atom.GetNumRadicalElectrons()
    hybridization = atom.GetHybridization()
    is_aromatic = atom.GetIsAromatic()
    is_in_ring = atom.IsInRing()
    features = np.array([
        atom_type,
        chirality,
        num_bonds,
        formal_charge,
        num_hydrogens,
        num_radical_electrons,
        hybridization,
        is_aromatic,
        is_in_ring
    ], dtype=np.float32)

    return features


def atom_to_feature_vector(atom):

    degree = atom.GetDegree()
    num_hydrogens = atom.GetTotalNumHs()
    atomic_mass = atom.GetMass()
    feature_vector = [degree, num_hydrogens, atomic_mass]
    
    return feature_vector


def dijkstra(adj, start_node):
    num_nodes = adj.shape[0]
    dist = np.full(num_nodes, np.inf)
    dist[start_node] = 0
    path = [-1] * num_nodes
    visited = [False] * num_nodes
    priority_queue = [(0, start_node)]

    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)

        if visited[current_node]:
            continue

        visited[current_node] = True

        for neighbor in range(num_nodes):
            if adj[current_node, neighbor] and not visited[neighbor]:
                new_dist = current_dist + 1
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    path[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_dist, neighbor))

    return dist, path

def all_pairs_shortest_paths_dijkstra(adj):
    num_nodes = adj.shape[0]
    shortest_path = np.full((num_nodes, num_nodes), np.inf)
    paths = [[[] for _ in range(num_nodes)] for _ in range(num_nodes)]  # Khởi tạo danh sách lồng nhau

    for start_node in range(num_nodes):
        dist, path = dijkstra(adj, start_node)
        shortest_path[start_node] = dist

        for end_node in range(num_nodes):
            if dist[end_node] < np.inf:
                current = end_node
                path_list = []
                while current != -1:
                    path_list.insert(0, current)
                    current = path[current]
                paths[start_node][end_node] = path_list  # Luôn là danh sách, ngay cả khi rỗng

    shortest_path[shortest_path == np.inf] = 0
    return shortest_path, paths




def gen_edge_input(max_dist, paths, attn_edge_type):
    num_nodes = len(paths)
    edge_input = np.zeros((num_nodes, num_nodes, max_dist, attn_edge_type.shape[-1]), dtype=np.int32)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and len(paths[i][j]) > 1:  # Đảm bảo có đường đi
                distance = len(paths[i][j]) - 1  # Độ dài đường đi
                if distance < max_dist:
                    edge_input[i, j, distance - 1, :] = attn_edge_type[i, j]

    return edge_input



def aa_sas_feature(prot_target):
    sas_features = []
    acc_file_path = f'davis/profile/{prot_target}_PROP/{prot_target}.acc'
    with open(acc_file_path, 'r') as f:
        lines = f.readlines()[3:]  
        for line in lines:
            values = line.strip().split()
            sas_probs = list(map(float, values[3:6])) 
            sas_features.append(sas_probs)

    return np.array(sas_features)

def aa_ss_feature(prot_target):
    ss_features = []

    ss3_file_path = f'davis/profile/{prot_target}_PROP/{prot_target}.ss8'
    
    with open(ss3_file_path, 'r') as f:
        lines = f.readlines()[2:]  
        for line in lines:
            values = line.strip().split()
            ss_probs = list(map(float, values[3:11])) 
            ss_features.append(ss_probs)

    return np.array(ss_features)

class MolEmbedding(nn.Module):
    def __init__(self, input_dim=3, embedding_dim=64):
        super(MolEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)  

    def forward(self, x):
        x = x.float()
        return self.linear(x) 


def mol_to_single_emb(x, embedding_dim=64):

    embedding_layer = MolEmbedding(input_dim=3, embedding_dim=embedding_dim)

    x_embedded = embedding_layer(x) 
    
    return x_embedded


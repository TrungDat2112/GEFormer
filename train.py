import copy
import pickle
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from collate import *
from dataset import *
from metrics import *
from model import GEFormerDTA


torch.manual_seed(2)
np.random.seed(3)

num_feat_xp = 0
num_feat_xd = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

NUM_EPOCHS = 100
TRAIN_BATCH_SIZE = 16
cuda = 0
LR = 0.001
LOG_INTERVAL = 3

import torch
import gc




def train(model, device, train_loader, optimizer, epoch):
    print(f'Training on {len(train_loader.dataset)} samples...', flush=True)
    model.train()
    total_train_loss = 0.0

    all_predictions = []
    all_labels = []

    for batch_idx, data in enumerate(train_loader):
        # Clear cache
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

        # Forward pass
        drug = data[0].to(device)
        prot = data[1].to(device)
        optimizer.zero_grad()
        output = model(drug, prot)

        # Calculate loss
        affinity = drug.y.view(-1, 1).float()
        loss = loss_fn(output, affinity.to(device))
        total_train_loss += loss.item()

        # Collect predictions
        all_predictions.extend(output.cpu().detach().numpy().flatten())
        all_labels.extend(affinity.cpu().detach().numpy().flatten())

        # Backward pass
        loss.backward()
        optimizer.step()

        # Log output
        if batch_idx % LOG_INTERVAL == 0:
            print(f'Train epoch: {epoch} [{batch_idx * len(drug.y)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}', flush=True)
    return total_train_loss / len(train_loader)



compound_iso_smiles = []
pdbs = []
pdbs_seqs = []
all_labels = []

df = pd.read_csv('davis/split/train.csv')
compound_iso_smiles += list(df['compound_iso_smiles'])
pdbs += list(df['target_name'])
pdbs_seqs += list(df['target_sequence'])
all_labels += list(df['affinity'])
pdbs_tseqs = set(zip(pdbs, pdbs_seqs, compound_iso_smiles, all_labels))

dta_graph = {}
saved_prot_graph = {}
print("Load pre-processed file for protein graph")
with open('saved_prot_graph.pickle', 'rb') as handle:
    saved_prot_graph = pickle.load(handle)


drug_file = "saved_drug_graph.pickle"
saved_drug_graph = {}
with open(drug_file, 'rb') as handle:
    while True:
        try:
            data = pickle.load(handle)
            saved_drug_graph.update(data)  
        except EOFError:
            break


for i in tqdm(pdbs_tseqs):
    g = copy.copy(saved_prot_graph[i[0]])
    g2 = copy.copy(saved_drug_graph[i[2]])
    g.y = torch.FloatTensor([i[3]])
    g2.y = torch.FloatTensor([i[3]])
    dta_graph[(i[0], i[2])] = [g, g2]
    num_feat_xp = g.x.size()[1]
    num_feat_xd = g2.x.size()[1]
pd.DataFrame(dta_graph).to_csv('./dta_graph.csv', index=False, index_label=False)

df = pd.read_csv('davis/split/train.csv')
df = df[:200]
train_drugs, train_prots, train_prots_seq, train_Y = list(df['compound_iso_smiles']), list(df['target_name']), list(
    df['target_sequence']), list(df['affinity'])
train_drugs, train_prots, train_prots_seq, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(
    train_prots_seq), np.asarray(train_Y)


train_data = GraphPairDataset(smile_list=train_drugs, dta_graph=dta_graph, prot_list=train_prots)

train_loader = DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate,
                          num_workers=0, pin_memory=False)


model = GEFormerDTA(num_features_xd=num_feat_xd, num_features_xt=num_feat_xp,device=device).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#for epoch in range (1,NUM_EPOCHS + 1):
train(model,device=device,train_loader=train_loader,optimizer=optimizer,epoch=NUM_EPOCHS)


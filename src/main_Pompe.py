from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from scGNN import GENELink
from torch.optim.lr_scheduler import StepLR
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation
import pandas as pd
import numpy as np
import random
import os
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
parser.add_argument('--num_head', type=list, default=[3, 3], help='Number of attention heads.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for leaky_relu.')
parser.add_argument('--hidden_dim', type=list, default=[128, 64, 32], help='Hidden dimensions')
parser.add_argument('--output_dim', type=int, default=16, help='Latent dimension')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--loop', type=bool, default=False, help='Add self-loop in adjacency')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type', type=str, default='dot', help='Score metric')
parser.add_argument('--flag', type=bool, default=False, help='Whether to conduct causal inference')
parser.add_argument('--reduction', type=str, default='concate', help='Multihead reduction')
parser.add_argument(
    '--prior_edges',
    type=str,
    default='/content/drive/MyDrive/MultiCausGRN_Pompe/pompe_gse159062/pompe_model_input/Prior_train_edges.csv',
    help='CSV with columns tf_idx,target_idx defining prior positive edges'
)
args = parser.parse_args()

# ---------------- Reproducibility ----------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- Dataset paths ----------------
data_type = 'POMPE_GSE159062'
num = 6116

exp_file = '/content/drive/MyDrive/MultiCausGRN_Pompe/pompe_gse159062/pompe_model_input/ExpressionData_clean.csv'
tf_file = '/content/drive/MyDrive/MultiCausGRN_Pompe/pompe_gse159062/pompe_model_input/TF.csv'
target_file = '/content/drive/MyDrive/MultiCausGRN_Pompe/pompe_gse159062/pompe_model_input/Target.csv'

train_file = '/content/drive/MyDrive/MultiCausGRN_Pompe/pompe_gse159062/pompe_model_input/Train_set2.csv'
test_file = '/content/drive/MyDrive/MultiCausGRN_Pompe/pompe_gse159062/pompe_model_input/Test_set2.csv'
val_file = '/content/drive/MyDrive/MultiCausGRN_Pompe/pompe_gse159062/pompe_model_input/Validation_set2.csv'

result_dir = f'Result/Pompe/{data_type}_{num}'
model_dir = 'model'
os.makedirs(result_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

tf_embed_path = os.path.join(result_dir, 'Channel1.csv')
target_embed_path = os.path.join(result_dir, 'Channel2.csv')

# ---------------- Helpers ----------------
def embed2file(tf_embed, tg_embed, gene_names, tf_path, target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    tf_df = pd.DataFrame(tf_embed, index=gene_names)
    tg_df = pd.DataFrame(tg_embed, index=gene_names)

    tf_df.to_csv(tf_path)
    tg_df.to_csv(target_path)

# ---------------- Load expression ----------------
data_input = pd.read_csv(exp_file, index_col=0)
loader = load_data(data_input)
feature = loader.exp_data()

tf = pd.read_csv(tf_file)["index"].astype("int64").values
target = pd.read_csv(target_file)["index"].astype("int64").values  # kept for compatibility

feature = torch.from_numpy(feature)
tf = torch.from_numpy(tf)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_feature = feature.to(device)
tf = tf.to(device)

# ---------------- Load splits ----------------
train_data = pd.read_csv(train_file).values
validation_data = pd.read_csv(val_file).values
test_data = pd.read_csv(test_file).values

# safety cast
for arr in [train_data, validation_data, test_data]:
    arr[:, 0] = arr[:, 0].astype(np.int64)
    arr[:, 1] = arr[:, 1].astype(np.int64)
    arr[:, 2] = arr[:, 2].astype(np.float32)

print("train row:", train_data[0], train_data[:,0].dtype, train_data[:,1].dtype)
print("train pos rate:", train_data[:,-1].mean())
print("val pos rate:", validation_data[:,-1].mean())
print("test pos rate:", test_data[:,-1].mean())

# ---------------- Dataset loaders ----------------
train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)

# adjacency from TRAIN PRIOR ONLY
prior_df = pd.read_csv(args.prior_edges)
prior_df = prior_df[["tf_idx", "target_idx"]].drop_duplicates().copy()
prior_df["tf_idx"] = prior_df["tf_idx"].astype(np.int64)
prior_df["target_idx"] = prior_df["target_idx"].astype(np.int64)

prior_pairs = np.column_stack([
    prior_df["tf_idx"].values,
    prior_df["target_idx"].values,
    np.ones(len(prior_df), dtype=np.int64)
]).astype(np.int64)

print("Prior edges loaded:", prior_pairs.shape[0], "from", args.prior_edges)
print("Prior idx max:", prior_pairs[:, :2].max(), "n_genes:", feature.shape[0])

adj_load = scRNADataset(prior_pairs, feature.shape[0], flag=False)
adj = adj_load.Adj_Generate(tf, direction=True, loop=args.loop)
adj = adj2saprse_tensor(adj)

# ---------------- Convert to tensors ----------------
train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
val_data = torch.from_numpy(validation_data)

# ---------------- Model ----------------
model = GENELink(
    input_dim=feature.size()[1],
    hidden1_dim=args.hidden_dim[0],
    hidden2_dim=args.hidden_dim[1],
    hidden3_dim=args.hidden_dim[2],
    output_dim=args.output_dim,
    num_head1=args.num_head[0],
    num_head2=args.num_head[1],
    alpha=args.alpha,
    device=device,
    type=args.Type,
    reduction=args.reduction
)

adj = adj.to(device)
model = model.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
validation_data = val_data.to(device)

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

best_val_aupr = -1
best_state = None
best_epoch = -1

# ---------------- Training ----------------
for epoch in range(args.epochs):
    running_loss = 0.0

    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        model.train()
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)

        pred = model(data_feature, adj, train_x)

        if args.flag:
            pred = torch.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)

        loss_bce = F.binary_cross_entropy(pred, train_y)
        loss_bce.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss_bce.item()

    # validation
    model.eval()
    with torch.no_grad():
        score = model(data_feature, adj, validation_data)
        if args.flag:
            score = torch.softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)

        AUC, AUPR, AUPR_norm = Evaluation(
            y_true=validation_data[:, -1],
            y_pred=score,
            flag=args.flag
        )

    print(
        f"Epoch:{epoch+1} "
        f"train_loss:{running_loss:.4f} "
        f"val_AUC:{AUC:.4f} "
        f"val_AUPR:{AUPR:.4f}"
    )

    if AUPR > best_val_aupr:
        best_val_aupr = AUPR
        best_epoch = epoch + 1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"[best so far] epoch={best_epoch} val_AUPR={best_val_aupr:.4f}")

# ---------------- Save/load best ----------------
best_model_path = os.path.join(model_dir, f'{data_type}_{num}_best.pkl')
# ---------------- Save best model ----------------

checkpoint = {
    "model_state_dict": best_state,
    "best_epoch": best_epoch,
    "best_val_aupr": best_val_aupr,
    "args": vars(args)
}

best_model_path = os.path.join(
    model_dir,
    f"{data_type}_{num}_seed{args.seed}_best.pkl"
)

torch.save(checkpoint, best_model_path)

print(f"\nBest model saved to:")
print(best_model_path)

# ---------------- Load best model ----------------

checkpoint = torch.load(best_model_path)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()

print(
    f"Loaded best epoch={checkpoint['best_epoch']} "
    f"val_AUPR={checkpoint['best_val_aupr']:.4f}"
)
model = model.to(device)
model.eval()
print(f"Loaded best epoch={best_epoch} with val_AUPR={best_val_aupr:.4f}")

# embeddings
tf_embed, target_embed = model.get_embedding()
gene_names = data_input.index.astype(str).tolist()
embed2file(tf_embed, target_embed, gene_names, tf_embed_path, target_embed_path)

# ---------------- Test ----------------
with torch.no_grad():
    score = model(data_feature, adj, test_data)
    if args.flag:
        score = torch.softmax(score, dim=1)
    else:
        score = torch.sigmoid(score)

AUC, AUPR, AUPR_norm = Evaluation(
    y_true=test_data[:, -1],
    y_pred=score,
    flag=args.flag
)

print(
    f"FINAL_RESULT seed={args.seed} "
    f"val_AUPR={best_val_aupr:.6f} "
    f"test_AUC={AUC:.6f} "
    f"test_AUPRC={AUPR:.6f}"
)

# save results
results_csv = os.path.join(result_dir, "results.csv")
file_exists = os.path.isfile(results_csv)

with open(results_csv, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["seed", "val_AUPR", "test_AUC", "test_AUPRC"])
    writer.writerow([args.seed, best_val_aupr, AUC, AUPR])
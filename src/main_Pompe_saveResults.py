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
import json
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=90, help='Number of epochs.')
parser.add_argument('--num_head', nargs=2, type=int, default=[3, 3], help='Number of attention heads.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for leaky_relu.')
parser.add_argument('--hidden_dim', nargs=3, type=int, default=[128, 64, 32], help='Hidden dimensions')
parser.add_argument('--output_dim', type=int, default=16, help='Latent dimension')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--loop', action='store_true', help='Add self-loop in adjacency')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type', type=str, default='dot', help='Score metric')
parser.add_argument('--flag', action='store_true', help='Whether to conduct causal inference')
parser.add_argument('--reduction', type=str, default='concate', help='Multihead reduction')
parser.add_argument('--prior_edges', type=str, default='pompe_model_input/Prior_train_edges.csv',
                    help='CSV with columns tf_idx,target_idx defining prior positive edges')
parser.add_argument('--exp_file', type=str, default='pompe_model_input/ExpressionData_clean.csv')
parser.add_argument('--tf_file', type=str, default='pompe_model_input/TF.csv')
parser.add_argument('--target_file', type=str, default='pompe_model_input/Target.csv')
parser.add_argument('--train_file', type=str, default='pompe_model_input/Train_set2.csv')
parser.add_argument('--val_file', type=str, default='pompe_model_input/Validation_set2.csv')
parser.add_argument('--test_file', type=str, default='pompe_model_input/Test_set2.csv')
parser.add_argument('--gene_map_file', type=str, default='pompe_model_input/gene_index_map.csv')
parser.add_argument('--data_type', type=str, default='POMPE_GSE159062')
parser.add_argument('--num_genes', type=int, default=6116)
parser.add_argument('--result_root', type=str, default='Result/Pompe')
parser.add_argument('--model_root', type=str, default='model')
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

# ---------------- Folders ----------------
run_name = f"{args.data_type}_genes{args.num_genes}_seed{args.seed}"
result_dir = os.path.join(args.result_root, run_name)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(args.model_root, exist_ok=True)

best_model_path = os.path.join(args.model_root, f"{run_name}_best.pkl")
history_csv = os.path.join(result_dir, "training_history.csv")
final_metrics_csv = os.path.join(result_dir, "final_metrics.csv")
all_runs_csv = os.path.join(args.result_root, "all_runs_summary.csv")
config_json = os.path.join(result_dir, "run_config.json")
tf_embed_path = os.path.join(result_dir, "tf_embedding.csv")
target_embed_path = os.path.join(result_dir, "target_embedding.csv")

# ---------------- Save config ----------------
with open(config_json, "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "args": vars(args)
    }, f, indent=2)

# ---------------- Helpers ----------------
def embed2file(tf_embed, tg_embed, gene_names, tf_path, target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    tf_df = pd.DataFrame(tf_embed, index=gene_names)
    tg_df = pd.DataFrame(tg_embed, index=gene_names)

    tf_df.to_csv(tf_path)
    tg_df.to_csv(target_path)

def save_prediction_csv(split_name, score_tensor, data_tensor, idx_to_gene, out_dir):
    scores = score_tensor.detach().cpu().numpy().reshape(-1)
    tf_idx = data_tensor[:, 0].detach().cpu().numpy().astype(int)
    target_idx = data_tensor[:, 1].detach().cpu().numpy().astype(int)
    labels = data_tensor[:, 2].detach().cpu().numpy()

    df = pd.DataFrame({
        "tf_idx": tf_idx,
        "target_idx": target_idx,
        "label": labels,
        "score": scores
    })
    df.to_csv(os.path.join(out_dir, f"{split_name}_predictions.csv"), index=False)

    df_named = df.copy()
    df_named["TF"] = df_named["tf_idx"].map(idx_to_gene)
    df_named["Target"] = df_named["target_idx"].map(idx_to_gene)
    df_named.to_csv(os.path.join(out_dir, f"{split_name}_predictions_named.csv"), index=False)

    # top confident positives by score
    top_df = df_named.sort_values("score", ascending=False).head(1000)
    top_df.to_csv(os.path.join(out_dir, f"{split_name}_top1000_edges.csv"), index=False)

    # hub TF ranking from scores
    hub_df = (
        df_named.groupby("TF", as_index=False)["score"]
        .agg(["sum", "mean", "count"])
        .reset_index()
        .rename(columns={"sum": "score_sum", "mean": "score_mean", "count": "n_edges"})
        .sort_values("score_sum", ascending=False)
    )
    hub_df.to_csv(os.path.join(out_dir, f"{split_name}_hub_tfs.csv"), index=False)

    return df_named

# ---------------- Load data ----------------
data_input = pd.read_csv(args.exp_file, index_col=0)
loader = load_data(data_input)
feature = loader.exp_data()

tf = pd.read_csv(args.tf_file)["index"].astype("int64").values
target = pd.read_csv(args.target_file)["index"].astype("int64").values

gene_map = pd.read_csv(args.gene_map_file)
idx_to_gene = dict(zip(gene_map["index"].astype(int), gene_map["gene_symbol"].astype(str)))

feature = torch.from_numpy(feature)
tf = torch.from_numpy(tf)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_feature = feature.to(device)
tf = tf.to(device)

# ---------------- Load splits ----------------
train_data = pd.read_csv(args.train_file).values
validation_data = pd.read_csv(args.val_file).values
test_data = pd.read_csv(args.test_file).values

for arr in [train_data, validation_data, test_data]:
    arr[:, 0] = arr[:, 0].astype(np.int64)
    arr[:, 1] = arr[:, 1].astype(np.int64)
    arr[:, 2] = arr[:, 2].astype(np.float32)

print("train row:", train_data[0], train_data[:, 0].dtype, train_data[:, 1].dtype)
print("train pos rate:", train_data[:, -1].mean())
print("val pos rate:", validation_data[:, -1].mean())
print("test pos rate:", test_data[:, -1].mean())

train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)

# ---------------- Build adjacency from TRAIN PRIOR ONLY ----------------
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

# ---------------- Convert split arrays to tensors ----------------
train_data = torch.from_numpy(train_data)
validation_data = torch.from_numpy(validation_data)
test_data = torch.from_numpy(test_data)

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
validation_data = validation_data.to(device)
test_data = test_data.to(device)

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

best_val_aupr = -1
best_val_auc = -1
best_val_aupr_norm = -1
best_state = None
best_epoch = -1

history_rows = []

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
        val_score = model(data_feature, adj, validation_data)
        if args.flag:
            val_score = torch.softmax(val_score, dim=1)
        else:
            val_score = torch.sigmoid(val_score)

        val_auc, val_aupr, val_aupr_norm = Evaluation(
            y_true=validation_data[:, -1],
            y_pred=val_score,
            flag=args.flag
        )

    current_lr = scheduler.get_last_lr()[0]
    history_rows.append({
        "epoch": epoch + 1,
        "train_loss": running_loss,
        "val_auc": float(val_auc),
        "val_aupr": float(val_aupr),
        "val_aupr_norm": float(val_aupr_norm),
        "lr": float(current_lr)
    })

    print(
        f"Epoch:{epoch+1} "
        f"train_loss:{running_loss:.4f} "
        f"val_AUC:{val_auc:.4f} "
        f"val_AUPR:{val_aupr:.4f}"
    )

    pd.DataFrame(history_rows).to_csv(history_csv, index=False)

    if val_aupr > best_val_aupr:
        best_val_aupr = float(val_aupr)
        best_val_auc = float(val_auc)
        best_val_aupr_norm = float(val_aupr_norm)
        best_epoch = epoch + 1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, best_model_path)

        # save best validation predictions too
        save_prediction_csv("validation_best", val_score, validation_data, idx_to_gene, result_dir)

    print(f"[best so far] epoch={best_epoch} val_AUPR={best_val_aupr:.4f}")

# ---------------- Load best ----------------
model.load_state_dict(torch.load(best_model_path))
model = model.to(device)
model.eval()
print(f"Loaded best epoch={best_epoch} with val_AUPR={best_val_aupr:.4f}")

# ---------------- Save embeddings ----------------
tf_embed, target_embed = model.get_embedding()
gene_names = data_input.index.astype(str).tolist()
embed2file(tf_embed, target_embed, gene_names, tf_embed_path, target_embed_path)

# ---------------- Final evaluation ----------------
with torch.no_grad():
    val_score = model(data_feature, adj, validation_data)
    test_score = model(data_feature, adj, test_data)

    if args.flag:
        val_score = torch.softmax(val_score, dim=1)
        test_score = torch.softmax(test_score, dim=1)
    else:
        val_score = torch.sigmoid(val_score)
        test_score = torch.sigmoid(test_score)

val_auc, val_aupr, val_aupr_norm = Evaluation(
    y_true=validation_data[:, -1],
    y_pred=val_score,
    flag=args.flag
)

test_auc, test_aupr, test_aupr_norm = Evaluation(
    y_true=test_data[:, -1],
    y_pred=test_score,
    flag=args.flag
)

print(
    f"FINAL_RESULT seed={args.seed} "
    f"best_epoch={best_epoch} "
    f"val_AUPR={val_aupr:.6f} "
    f"test_AUC={test_auc:.6f} "
    f"test_AUPRC={test_aupr:.6f}"
)

# ---------------- Save final predictions ----------------
save_prediction_csv("validation_final", val_score, validation_data, idx_to_gene, result_dir)
save_prediction_csv("test_final", test_score, test_data, idx_to_gene, result_dir)

# ---------------- Save final metrics ----------------
final_metrics = pd.DataFrame([{
    "run_name": run_name,
    "seed": args.seed,
    "best_epoch": best_epoch,
    "n_genes": args.num_genes,
    "n_tfs": len(tf.cpu().numpy()),
    "n_train_edges": int(train_data.shape[0]),
    "n_val_edges": int(validation_data.shape[0]),
    "n_test_edges": int(test_data.shape[0]),
    "n_prior_train_edges": int(len(prior_df)),
    "best_val_auc": best_val_auc,
    "best_val_aupr": best_val_aupr,
    "best_val_aupr_norm": best_val_aupr_norm,
    "final_val_auc": float(val_auc),
    "final_val_aupr": float(val_aupr),
    "final_val_aupr_norm": float(val_aupr_norm),
    "test_auc": float(test_auc),
    "test_aupr": float(test_aupr),
    "test_aupr_norm": float(test_aupr_norm),
    "lr": args.lr,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "output_dim": args.output_dim,
    "type": args.Type,
    "reduction": args.reduction
}])

final_metrics.to_csv(final_metrics_csv, index=False)

# append to global summary
file_exists = os.path.isfile(all_runs_csv)
with open(all_runs_csv, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(final_metrics.columns.tolist())
    writer.writerow(final_metrics.iloc[0].tolist())
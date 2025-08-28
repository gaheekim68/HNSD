"""
Training script for (Sheaf) Hypergraph models.
Expected local modules:
    layers.py, models.py, preprocessing.py, convert_datasets_to_pygDataset.py
"""
from __future__ import annotations

import argparse
import copy
import os
import os.path as osp
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from tqdm import tqdm  # noqa: F401  # (kept for potential progress bars)

from convert_datasets_to_pygDataset import dataset_Hypergraph
from layers import *  # noqa: F403  # local project imports
from models import *  # noqa: F403
from preprocessing import *  # noqa: F403

# -----------------------------------------------------------------------------
# Reproducibility & env
# -----------------------------------------------------------------------------
os.environ.setdefault("WANDB_AGENT_MAX_INITIAL_FAILURES", "200")
np.random.seed(100)
torch.manual_seed(100)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval_acc(y_true: torch.Tensor, y_pred_logprobs: torch.Tensor, name: str) -> float:
    """Compute accuracy from labels and log-probabilities."""
    y_true_np = y_true.detach().cpu().numpy()
    y_pred = y_pred_logprobs.argmax(dim=-1, keepdim=False).detach().cpu().numpy()
    is_labeled = y_true_np == y_true_np
    correct = y_true_np[is_labeled] == y_pred[is_labeled]
    return float(np.sum(correct) / len(correct)) if len(correct) else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data,
    split_idx: Dict[str, torch.Tensor],
    eval_func,
    result: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, out_logprobs)."""
    if result is not None:
        out_log = result
    else:
        model.eval()
        out = model(data)
        out_log = F.log_softmax(out, dim=1)

    train_acc = eval_func(data.y[split_idx["train"]], out_log[split_idx["train"]], name="train")
    valid_acc = eval_func(data.y[split_idx["valid"]], out_log[split_idx["valid"]], name="valid")
    test_acc = eval_func(data.y[split_idx["test"]], out_log[split_idx["test"]], name="test")

    train_loss = F.nll_loss(out_log[split_idx["train"]], data.y[split_idx["train"]])
    valid_loss = F.nll_loss(out_log[split_idx["valid"]], data.y[split_idx["valid"]])
    test_loss = F.nll_loss(out_log[split_idx["test"]], data.y[split_idx["test"]])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out_log


class Logger:
    """Keep per-epoch metrics and print/plot summaries (OGB-style).

    Handles uneven run lengths by truncating to the shortest run.
    """

    def __init__(self, runs: int, info: Optional[argparse.Namespace] = None):
        self.info = info
        self.results: List[List[Tuple[float, float, float]]] = [[] for _ in range(runs)]

    def add_result(self, run: int, result: Tuple[float, float, float]):
        assert len(result) == 3
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def _truncate(self) -> List[List[Tuple[float, float, float]]]:
        min_len = min(len(r) for r in self.results)
        return [r[:min_len] for r in self.results]

    def print_statistics(self, run: Optional[int] = None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result[:, 0].max():.2f}")
            print(f"Highest Valid: {result[:, 1].max():.2f}")
            print(f"  Final Train: {result[argmax, 0]:.2f}")
            print(f"   Final Test: {result[argmax, 2]:.2f}")
            return None, None
        # all runs
        results_trimmed = self._truncate()
        result = 100 * torch.tensor(results_trimmed)
        best_results = []
        for r in result:
            train1 = r[:, 0].max().item()
            valid = r[:, 1].max().item()
            train2 = r[r[:, 1].argmax(), 0].item()
            test = r[r[:, 1].argmax(), 2].item()
            best_results.append((train1, valid, train2, test))
        best_result = torch.tensor(best_results)
        print("All runs:")
        r = best_result[:, 0]
        print(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
        r = best_result[:, 1]
        print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
        r = best_result[:, 2]
        print(f"  Final Train: {r.mean():.2f} ± {r.std():.2f}")
        r = best_result[:, 3]
        print(f"   Final Test: {r.mean():.2f} ± {r.std():.2f}")
        return best_result[:, 1], best_result[:, 3]

    def plot_result(self, run: Optional[int] = None):
        plt.style.use("seaborn")
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f"Run {run + 1:02d}:")
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(["Train", "Valid", "Test"])
            return
        result = 100 * torch.tensor(self.results[0])
        x = torch.arange(result.shape[0])
        plt.figure()
        plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
        plt.legend(["Train", "Valid", "Test"])


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------

def parse_method(args: argparse.Namespace, data) -> nn.Module:
    """Instantiate the model based on args.method."""
    if args.method == "AllSetTransformer":
        return SetGNN(args, data.norm) if args.LearnMask else SetGNN(args)  # noqa: F405

    if args.method == "AllDeepSets":
        args.PMA = False
        args.aggregate = "add"
        return SetGNN(args, data.norm) if args.LearnMask else SetGNN(args)  # noqa: F405

    if args.method == "CEGCN":
        return CEGCN(  # noqa: F405
            in_dim=args.num_features,
            hid_dim=args.MLP_hidden,
            out_dim=args.num_classes,
            num_layers=args.All_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
        )

    if args.method == "CEGAT":
        return CEGAT(  # noqa: F405
            in_dim=args.num_features,
            hid_dim=args.MLP_hidden,
            out_dim=args.num_classes,
            num_layers=args.All_num_layers,
            heads=args.heads,
            output_heads=args.output_heads,
            dropout=args.dropout,
            Normalization=args.normalization,
        )

    if args.method == "HyperGCN":
        He_dict = get_HyperGCN_He_dict(data)  # noqa: F405
        return HyperGCN(  # noqa: F405
            V=data.x.shape[0],
            E=He_dict,
            X=data.x,
            num_features=args.num_features,
            num_layers=args.All_num_layers,
            num_classses=args.num_classes,
            args=args,
        )

    if args.method in ["SheafHyperGCNDiag", "SheafHyperGCNOrtho", "SheafHyperGCNGeneral", "SheafHyperGCNLowRank"]:
        sheaf_map = {
            "SheafHyperGCNDiag": "DiagSheafs",
            "SheafHyperGCNOrtho": "OrthoSheafs",
            "SheafHyperGCNGeneral": "GeneralSheafs",
            "SheafHyperGCNLowRank": "LowRankSheafs",
        }
        return SheafHyperGCN(  # noqa: F405
            V=data.x.shape[0],
            num_features=args.num_features,
            num_layers=args.All_num_layers,
            num_classses=args.num_classes,
            args=args,
            sheaf_type=sheaf_map[args.method],
        )

    if args.method == "HGNN":
        args.use_attention = False
        return HCHA(args)  # noqa: F405

    if args.method == "HNHN":
        return HNHN(args)  # noqa: F405

    if args.method == "HCHA":
        return HCHA(args)  # noqa: F405

    if args.method == "MLP":
        return MLP_model(args)  # noqa: F405

    if args.method in ["SheafHyperGNNDiag", "SheafHyperGNNOrtho", "SheafHyperGNNGeneral", "SheafHyperGNNLowRank"]:
        return SheafHyperGNN(args, args.method)  # noqa: F405

    raise ValueError(f"Unknown method: {args.method}")


# -----------------------------------------------------------------------------
# t-SNE helpers
# -----------------------------------------------------------------------------

def plot_tsne_features(data, save_path: Optional[str] = None):
    x = data.x.cpu().numpy()
    labels = data.y.cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    x_2d = tsne.fit_transform(x)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(x_2d[:, 0], x_2d[:, 1], c=labels, cmap="tab10", s=10, alpha=0.8)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE on Raw Features (before split)")
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved t-SNE plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_tsne_embeddings(
    model: nn.Module,
    data,
    title: str = "t-SNE on HNSD Embeddings",
    save_path: Optional[str] = None,
):
    model.eval()
    with torch.no_grad():
        out = model(data)
        embeddings = out.cpu().numpy()
        labels = data.y.cpu().numpy()
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        z_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab10", s=10, alpha=0.8)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Saved t-SNE plot to {save_path}")
        else:
            plt.show()
        plt.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # general
    p.add_argument("--dname", default="cora")
    p.add_argument("--method", default="SheafHyperGNNDiag")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--cuda", type=int, default=0, choices=[-1, 0])
    p.add_argument("--display_step", type=int, default=-1)
    p.add_argument("--seed", type=int, default=100)
    # early stopping option
    p.add_argument("--earlystop", type=str2bool, default=True,
    help="Use early stopping if True; train full epochs if False")

    # data/splits
    p.add_argument("--train_prop", type=float, default=0.5)
    p.add_argument("--valid_prop", type=float, default=0.25)
    p.add_argument("--feature_noise", default="1")

    # opt
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)

    # architecture
    p.add_argument("--All_num_layers", type=int, default=2)
    p.add_argument("--MLP_num_layers", type=int, default=2)
    p.add_argument("--MLP_hidden", type=int, default=128)
    p.add_argument("--Classifier_num_layers", type=int, default=2)
    p.add_argument("--Classifier_hidden", type=int, default=64)
    p.add_argument("--aggregate", default="mean", choices=["sum", "mean"])  # AllDeepSets may override
    p.add_argument("--normalization", default="ln")
    p.add_argument("--activation", default="relu", choices=["Id", "relu", "prelu"])

    # toggles
    p.add_argument("--new_edge", type=str2bool, default=False)
    p.add_argument("--plot", type=str2bool, default=False)
    p.add_argument("--ablation_fixed_sheaf", type=str2bool, default=False)
    p.add_argument("--no_diffusion", action="store_true", help="Disable diffusion layers for ablation")
    p.add_argument("--add_self_loop", action="store_true")
    p.add_argument("--exclude_self", action="store_true")
    p.add_argument("--GPR", action="store_true")
    p.add_argument("--LearnMask", action="store_true")
    p.add_argument("--PMA", action="store_true")

    # placeholders (populated after data load)
    p.add_argument("--num_features", type=int, default=0)
    p.add_argument("--num_classes", type=int, default=0)

    # HyperGCN
    p.add_argument("--HyperGCN_mediators", action="store_true")
    p.add_argument("--HyperGCN_fast", type=str2bool, default=True)

    # Attention / GAT / SetGNN
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--output_heads", type=int, default=1)

    # HNHN
    p.add_argument("--HNHN_alpha", type=float, default=-1.5)
    p.add_argument("--HNHN_beta", type=float, default=-0.5)
    p.add_argument("--HNHN_nonlinear_inbetween", type=str2bool, default=True)

    # HCHA
    p.add_argument("--HCHA_symdegnorm", action="store_true")
    p.add_argument("--use_attention", type=str2bool, default=True)

    # UniGNN (kept for compatibility)
    p.add_argument("--UniGNN_use-norm", action="store_true")
    p.add_argument("--UniGNN_degV", type=int, default=0)
    p.add_argument("--UniGNN_degE", type=int, default=0)

    # Sheaf options
    p.add_argument("--init_hedge", type=str, default="avg", choices=["rand", "avg"]) 
    p.add_argument("--sheaf_normtype", type=str, default="sym_degree_norm",
                   choices=["degree_norm", "block_norm", "sym_degree_norm", "sym_block_norm"]) 
    p.add_argument("--sheaf_act", type=str, default="tanh", choices=["sigmoid", "tanh", "none"]) 
    p.add_argument("--sheaf_dropout", type=str2bool, default=False) 
    p.add_argument("--sheaf_left_proj", type=str2bool, default=True) 
    p.add_argument("--dynamic_sheaf", type=str2bool, default=True) 
    p.add_argument("--sheaf_special_head", type=str2bool, default=False) 
    p.add_argument("--sheaf_pred_block", type=str, default="cp_decomp") 
    p.add_argument("--sheaf_transformer_head", type=int, default=1) 
    p.add_argument("--residual_HCHA", type=str2bool, default=False) 
    p.add_argument("--rank", type=int, default=0, help="rank for dxd blocks in LowRankSheafs") 
    p.add_argument("--AllSet_input_norm", type=str2bool, default=True)
    p.add_argument("--normtype", default="all_one", choices=["all_one", "deg_half_sym"])

    # logging
    p.add_argument("--wandb", type=str2bool, default=False)
    p.add_argument("--tag", type=str, default="testing")

    # paths for plots/outputs
    p.add_argument("--tsne_dir", type=str, default="tsne")
    p.add_argument("--res_root", type=str, default="hyperparameter_tuning")
    p.add_argument("--csv_prefix", type=str, default="0_")

    return p


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # seed (repeatable across dataloading libs if needed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.wandb:
        import wandb
        wandb.init(sync_tensorboard=False, project="hyper_sheaf_baseline", reinit=False, config=args, tags=[args.tag])
        print("Monitoring using wandb")

    # ---------------------------- Load data ---------------------------------
    existing_dataset = [
        "coauthor_cora",
        "coauthor_dblp",
        "cora",
        "citeseer",
        "senate-committees",
        "senate-committees-100"
    ]

    synthetic_list = [
        "senate-committees",
        "senate-committees-100"
    ]

    if args.method in ["SheafHyperGCNLowRank", "LowRankSheafsDiffusion", "SheafEquivSetGNN_LowRank"]:
        assert args.rank <= args.heads // 2

    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = "../data/AllSet_all_raw_data/"
            dataset = dataset_Hypergraph(name=dname, feature_noise=f_noise, p2raw=p2raw)
        else:
            if dname in ["cora", "citeseer", "pubmed"]:
                p2raw = "../data/AllSet_all_raw_data/cocitation/"
            elif dname in ["coauthor_cora", "coauthor_dblp"]:
                p2raw = "../data/AllSet_all_raw_data/coauthorship/"
            else:
                p2raw = "../data/AllSet_all_raw_data/"
            dataset = dataset_Hypergraph(name=dname, root="../data/pyg_data/hypergraph_dataset_updated/", p2raw=p2raw)
        data = dataset.data
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        if args.dname in [
            "house-committees",
            "house-committees-100",
            "senate-committees",
            "senate-committees-100",
            "congress-bills",
            "congress-bills-100",
        ]:
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
        if not hasattr(data, "n_x"):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, "num_hyperedges"):
            data.num_hyperedges = torch.tensor([data.edge_index[0].max() - data.n_x[0] + 1])
        assert data.y.min().item() == 0
    else:
        raise ValueError(f"Unknown dataset name: {args.dname}")

    # ---------------------------- Preprocess --------------------------------
    if args.method in ["AllSetTransformer", "AllDeepSets"]:
        data = ExtractV2E(data)  # noqa: F405
        if args.new_edge:
            data = new_edge_index(data)  # noqa: F405
        if args.add_self_loop:
            data = Add_Self_Loops(data)  # noqa: F405
        if args.exclude_self:
            data = expand_edge_index(data)  # noqa: F405
        data = norm_contruction(data, option=args.normtype)  # noqa: F405

    elif args.method in ["CEGCN", "CEGAT"]:
        data = ExtractV2E(data)  # noqa: F405
        if args.new_edge:
            data = new_edge_index(data)  # noqa: F405
        data = ConstructV2V(data)  # noqa: F405
        data = norm_contruction(data, TYPE="V2V")  # noqa: F405

    elif args.method in ["HyperGCN"]:
        data = ExtractV2E(data)  # noqa: F405
        if args.new_edge:
            data = new_edge_index(data)  # noqa: F405

    elif args.method in [
        "SheafHyperGCNDiag",
        "SheafHyperGCNOrtho",
        "SheafHyperGCNGeneral",
        "SheafHyperGCNLowRank",
    ]:
        data = ExtractV2E(data)  # noqa: F405
        if args.new_edge:
            data = new_edge_index(data)  # noqa: F405
        data.edge_index[1] -= data.edge_index[1].min()

    elif args.method in ["HNHN"]:
        data = ExtractV2E(data)  # noqa: F405
        if args.new_edge:
            data = new_edge_index(data)  # noqa: F405
        if args.add_self_loop:
            data = Add_Self_Loops(data)  # noqa: F405
        H = ConstructH_HNHN(data)  # noqa: F405
        data = generate_norm_HNHN(H, data, args)  # noqa: F405
        data.edge_index[1] -= data.edge_index[1].min()

    elif args.method in [
        "HCHA",
        "HGNN",
        "DiagSheafs",
        "OrthoSheafs",
        "GeneralSheafs",
        "LowRankSheafs",
        "SheafHyperGNNDiag",
        "SheafHyperGNNOrtho",
        "SheafHyperGNNGeneral",
        "SheafHyperGNNLowRank",
    ]:
        data = ExtractV2E(data)  # noqa: F405
        if args.new_edge:
            data = new_edge_index(data)  # noqa: F405
        if args.add_self_loop:
            data = Add_Self_Loops(data)  # noqa: F405
        data.edge_index[1] -= data.edge_index[1].min()

    else:
        pass

    # train/val/test splits
    split_idx_lst: List[Dict[str, torch.Tensor]] = []
    for _ in range(args.runs):
        split_idx = rand_train_test_idx(data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)  # noqa: F405
        split_idx_lst.append(split_idx)

    if args.plot:
        os.makedirs(args.tsne_dir, exist_ok=True)
        plot_tsne_features(data, save_path=osp.join(args.tsne_dir, f"{args.dname}_raw.png"))

    # ------------------------------ Model -----------------------------------
    model = parse_method(args, data)

    # device
    device = torch.device(f"cuda:{args.cuda}") if (args.cuda >= 0 and torch.cuda.is_available()) else torch.device("cpu")
    print("Device:", device)
    model, data = model.to(device), data.to(device)

    if args.wandb:
        import wandb
        wandb.watch(model)

    # ------------------------------ Train -----------------------------------
    logger = Logger(args.runs, args)
    criterion = nn.NLLLoss()
    eval_func = eval_acc

    runtime_list: List[float] = []

    # Make sure we have a fallback best state (in case no improvement occurs)
    best_model_state = copy.deepcopy(model.state_dict())

    for run in range(args.runs):
        run_start = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx["train"].to(device)


        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)


        best_val_acc = -float("inf")
        best_test_at_val = -float("inf")
        early_stop_counter = 0
        patience = 10


        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            out_log = F.log_softmax(out, dim=1)
            loss = criterion(out_log[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()


            result = evaluate(model, data, split_idx, eval_func)
            _, val_acc, test_acc, _, val_loss, _, _ = result


            scheduler.step(val_loss)


            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_test_at_val = test_acc
                early_stop_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                early_stop_counter += 1


            logger.add_result(run, result[:3])


            if args.earlystop and early_stop_counter >= patience:
                break


        runtime_list.append(time.time() - run_start)
        logger.plot_result(run)

    # restore best model across all runs for optional t-SNE
    model.load_state_dict(best_model_state)

    if args.plot:
        os.makedirs(args.tsne_dir, exist_ok=True)
        plot_tsne_embeddings(
            model,
            data,
            save_path=osp.join(args.tsne_dir, f"{args.dname}_{args.method}.png"),
            title=f"t-SNE on {args.method} Embeddings",
        )

    # ------------------------------ Save results ----------------------------
    avg_time, std_time = float(np.mean(runtime_list)), float(np.std(runtime_list))

    best_val, best_test = logger.print_statistics()

    os.makedirs(args.res_root, exist_ok=True)
    filename = osp.join(args.res_root, f"{args.csv_prefix}{args.dname}_.csv")
    print(f"Saving results to {filename}")
    with open(filename, "a+", encoding="utf-8") as f:
        cur_line = f"METHOD_{args.method}_lr_{args.lr}_wd_{args.wd}_dropout_{args.dropout}_"
        cur_line += f",{best_val.mean():.3f} ± {best_val.std():.3f}"
        cur_line += f",{best_test.mean():.3f} ± {best_test.std():.3f}"
        cur_line += f",{avg_time//60:.0f} min {(avg_time % 60):.2f} ± {std_time:.2f}s\n"
        f.write(cur_line)

    all_args_file = osp.join(args.res_root, f"hyperpara_all_args_{args.dname}_.csv")
    with open(all_args_file, "a+", encoding="utf-8") as f:
        f.write(str(args))
        f.write("\n")

    total_time = sum(runtime_list)
    print(
        f"TIME FOR ONE EXPERIMENT WITH {args.runs} RUNS: \\ Minutes: {total_time//60:.0f}, seconds {total_time%60:.0f}"
    )
    print("All done! Exit python code")


if __name__ == "__main__":
    main()

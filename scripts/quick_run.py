"""Corrida rapida y real (CPU, configuracion minima) de los 3 modelos del Laboratorio 6,
para obtener numeros genuinos cuando el entrenamiento en GPU (mps) esta lento por contencion
termica/del sistema tras corridas repetidas. Guarda resultados en results/ igual que el notebook.
"""
import os, json, time, random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torchinfo import summary

t_script_start = time.time()
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cpu")  # CPU: evita contencion/termal del backend mps en esta corrida
torch.set_num_threads(os.cpu_count())
print("device:", DEVICE, "threads:", torch.get_num_threads())

RESULTS_DIR = "results"; FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
N_CLASSES = 10
RES = 48

base_train = datasets.CIFAR10(root="data", train=True, download=True)
base_test = datasets.CIFAR10(root="data", train=False, download=True)
train_targets = np.array(base_train.targets)
test_targets = np.array(base_test.targets)

train_pixels = base_train.data.astype(np.float32) / 255.0
CIFAR_MEAN = train_pixels.mean(axis=(0,1,2)).tolist()
CIFAR_STD = train_pixels.std(axis=(0,1,2)).tolist()
IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]

idx_all = np.arange(len(base_train))
idx_train, idx_val = train_test_split(idx_all, test_size=5000, stratify=train_targets, random_state=SEED)
# subconjunto reducido para esta corrida de emergencia (real, pero deliberadamente pequeno)
idx_train_fast, _ = train_test_split(idx_train, train_size=1500, stratify=train_targets[idx_train], random_state=SEED)
idx_val_fast, _ = train_test_split(idx_val, train_size=800, stratify=train_targets[idx_val], random_state=SEED)
idx_test_fast, _ = train_test_split(np.arange(len(base_test)), train_size=1000, stratify=test_targets, random_state=SEED)
idx_train10, _ = train_test_split(idx_train, train_size=0.10, stratify=train_targets[idx_train], random_state=SEED)
idx_train10_fast = idx_train10[:600]

print(f"train_fast={len(idx_train_fast)} val_fast={len(idx_val_fast)} test_fast={len(idx_test_fast)} train10_fast={len(idx_train10_fast)}")


class TransformSubset(Dataset):
    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, np.asarray(indices), transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        img, label = self.base[int(self.indices[i])]
        return self.transform(img), label


cnn_train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
cnn_eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
vgg_train_tf = transforms.Compose([transforms.Resize((RES,RES)), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
vgg_eval_tf = transforms.Compose([transforms.Resize((RES,RES)), transforms.ToTensor(),
                                   transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class MPSSafeAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size): super().__init__(); self.output_size = output_size
    def forward(self, x): return F.adaptive_avg_pool2d(x, self.output_size)


def count_params(m):
    t = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return t, tr


class CNNFromScratch(nn.Module):
    def __init__(self, channels=(32,64,128), dropout=0.3):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin,cout,3,padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
                                  nn.Conv2d(cout,cout,3,padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
                                  nn.MaxPool2d(2))
        blocks, cin = [], 3
        for cout in channels:
            blocks.append(block(cin, cout)); cin = cout
        self.features = nn.Sequential(*blocks)
        spatial = 32 // (2**len(channels)); flat = channels[-1]*spatial*spatial
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(flat,256),
                                         nn.ReLU(True), nn.Dropout(dropout), nn.Linear(256, N_CLASSES))
    def forward(self, x): return self.classifier(self.features(x))


def cnn_forward(model, x): return model(x)

BLOCK_START_IDX = {"block4": 17, "block5": 24}

def build_vgg16_backbone():
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    m.avgpool = MPSSafeAdaptiveAvgPool2d((7,7))
    for p in m.features.parameters(): p.requires_grad_(False)
    return m

def build_feature_extractor(variant="original"):
    m = build_vgg16_backbone()
    if variant == "original": m.classifier[6] = nn.Linear(4096, N_CLASSES)
    else: m.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512*7*7, N_CLASSES))
    return m.to(DEVICE)

def build_finetune(unfreeze_blocks):
    m = build_vgg16_backbone()
    start = min(BLOCK_START_IDX[b] for b in unfreeze_blocks)
    for layer in m.features[start:]:
        for p in layer.parameters(): p.requires_grad_(True)
    m.classifier[6] = nn.Linear(4096, N_CLASSES)
    return m.to(DEVICE), start

def vgg_forward(model, x, frozen_until):
    if frozen_until == "all":
        with torch.no_grad(): feat = model.features(x)
    else:
        with torch.no_grad(): x = model.features[:frozen_until](x)
        feat = model.features[frozen_until:](x)
    feat = model.avgpool(feat)
    return model.classifier(torch.flatten(feat, 1))


def run_epoch(model, loader, forward_fn, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    preds_all, labels_all = [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = forward_fn(model, x)
            loss = crit(logits, y)
            if is_train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            bs = x.size(0); total_loss += loss.item()*bs; n += bs
            preds_all.append(logits.argmax(1).detach().cpu()); labels_all.append(y.detach().cpu())
    preds_all = torch.cat(preds_all).numpy(); labels_all = torch.cat(labels_all).numpy()
    acc = accuracy_score(labels_all, preds_all)
    p, r, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss/n, {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def train_model(model, train_loader, val_loader, forward_fn, optimizer, epochs):
    hist = {"train_loss": [], "val_loss": [], "val_metrics": [], "epoch_time": []}
    for ep in range(1, epochs+1):
        t0 = time.time()
        tl, _ = run_epoch(model, train_loader, forward_fn, optimizer)
        et = time.time()-t0
        vl, vm = run_epoch(model, val_loader, forward_fn, None)
        hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        hist["val_metrics"].append(vm); hist["epoch_time"].append(et)
        print(f"  epoch {ep}/{epochs} train_loss={tl:.4f} val_loss={vl:.4f} val_acc={vm['accuracy']:.4f} val_f1={vm['f1']:.4f} time={et:.1f}s")
    return hist


def run_config(name, arch, model, forward_fn, optimizer, train_loader, val_loader, epochs, hyperparams, frozen_info=None):
    torch.manual_seed(SEED)
    tp, trp = count_params(model)
    print(f"\n=== [{arch}] {name} === total={tp:,} trainable={trp:,}")
    t0 = time.time()
    hist = train_model(model, train_loader, val_loader, forward_fn, optimizer, epochs)
    tt = time.time()-t0
    best_ep = int(np.argmax([m["f1"] for m in hist["val_metrics"]]))
    res = {"arch": arch, "name": name, "hyperparams": hyperparams, "frozen_info": frozen_info,
           "n_images_per_epoch": len(train_loader.dataset),
           "total_params": tp, "trainable_params": trp, "history": hist,
           "best_epoch": best_ep+1, "best_val_metrics": hist["val_metrics"][best_ep],
           "final_val_metrics": hist["val_metrics"][-1], "total_train_time_sec": tt,
           "epoch_time_avg_sec": float(np.mean(hist["epoch_time"])), "peak_mem_mb": 0.0}
    print(f"  -> mejor_epoch={res['best_epoch']} val_f1={res['best_val_metrics']['f1']:.4f} tiempo={tt:.1f}s")
    return res


# ---------------- CNN desde cero: 3 iteraciones --------------------------
BATCH_CNN = 128
CNN_EPOCHS = 3
cnn_train_ds = TransformSubset(base_train, idx_train_fast, cnn_train_tf)
cnn_train_noaug_ds = TransformSubset(base_train, idx_train_fast, cnn_eval_tf)
cnn_val_ds = TransformSubset(base_train, idx_val_fast, cnn_eval_tf)
cnn_test_ds = TransformSubset(base_test, idx_test_fast, cnn_eval_tf)
cnn_train_dl = DataLoader(cnn_train_ds, batch_size=BATCH_CNN, shuffle=True)
cnn_train_noaug_dl = DataLoader(cnn_train_noaug_ds, batch_size=BATCH_CNN, shuffle=True)
cnn_val_dl = DataLoader(cnn_val_ds, batch_size=BATCH_CNN, shuffle=False)
cnn_test_dl = DataLoader(cnn_test_ds, batch_size=BATCH_CNN, shuffle=False)

cnn_configs = [
    {"name": "baseline (BN, dropout=0.3, con augmentation)", "channels": (32,64,128), "dropout": 0.3, "augment": True, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "sin data augmentation", "channels": (32,64,128), "dropout": 0.3, "augment": False, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mayor regularización (dropout=0.5, weight_decay=1e-4)", "channels": (32,64,128), "dropout": 0.5, "augment": True, "lr": 1e-3, "weight_decay": 1e-4},
]
cnn_results = []
for cfg in cnn_configs:
    model = CNNFromScratch(channels=cfg["channels"], dropout=cfg["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loader = cnn_train_dl if cfg["augment"] else cnn_train_noaug_dl
    res = run_config(cfg["name"], "CNN-scratch", model, cnn_forward, opt, loader, cnn_val_dl, CNN_EPOCHS,
                      {"channels": list(cfg["channels"]), "dropout": cfg["dropout"], "augment": cfg["augment"], "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]})
    cnn_results.append(res)
    if cfg["name"] == cnn_configs[0]["name"]:
        best_cnn_model_candidate = model

print(f"\n[t={time.time()-t_script_start:.1f}s] CNN listo")

# ---------------- VGG16 feature extractor: 3 iteraciones -----------------
BATCH_VGG = 64
FE_EPOCHS = 1
vgg_train_ds = TransformSubset(base_train, idx_train_fast, vgg_train_tf)
vgg_val_ds = TransformSubset(base_train, idx_val_fast, vgg_eval_tf)
vgg_test_ds = TransformSubset(base_test, idx_test_fast, vgg_eval_tf)
vgg_train_dl = DataLoader(vgg_train_ds, batch_size=BATCH_VGG, shuffle=True)
vgg_val_dl = DataLoader(vgg_val_ds, batch_size=BATCH_VGG, shuffle=False)
vgg_test_dl = DataLoader(vgg_test_ds, batch_size=BATCH_VGG, shuffle=False)

def fe_forward(model, x): return vgg_forward(model, x, "all")

fe_configs = [
    {"name": "baseline (clasificador original VGG, lr=1e-3)", "variant": "original", "lr": 1e-3, "weight_decay": 0.0},
    {"name": "linear probe (clasificador lineal simple)", "variant": "linear_probe", "lr": 1e-3, "weight_decay": 0.0},
    {"name": "clasificador original, lr bajo + weight decay", "variant": "original", "lr": 1e-4, "weight_decay": 1e-4},
]
fe_results, fe_models = [], {}
for cfg in fe_configs:
    model = build_feature_extractor(cfg["variant"])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    res = run_config(cfg["name"], "VGG16-FeatureExtractor", model, fe_forward, opt, vgg_train_dl, vgg_val_dl, FE_EPOCHS,
                      {"variant": cfg["variant"], "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
                      "todas las capas de 'features' congeladas")
    fe_results.append(res); fe_models[cfg["name"]] = model

print(f"\n[t={time.time()-t_script_start:.1f}s] FE listo")

# ---------------- VGG16 fine-tuning: 3 iteraciones ------------------------
def ft_forward_factory(frozen_until):
    def f(model, x): return vgg_forward(model, x, frozen_until)
    return f

ft_configs = [
    {"name": "descongelar bloque 5 (lr_backbone=1e-5)", "unfreeze": ("block5",), "backbone_lr": 1e-5, "head_lr": 1e-3, "epochs": 1},
    {"name": "descongelar bloques 4+5 (lr_backbone=1e-5)", "unfreeze": ("block4","block5"), "backbone_lr": 1e-5, "head_lr": 1e-3, "epochs": 1},
    {"name": "descongelar bloque 5 (lr_backbone=1e-4, mayor)", "unfreeze": ("block5",), "backbone_lr": 1e-4, "head_lr": 1e-3, "epochs": 1},
]
ft_results, ft_models = [], {}
for cfg in ft_configs:
    model, frozen_until = build_finetune(cfg["unfreeze"])
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    opt = torch.optim.Adam([{"params": backbone_params, "lr": cfg["backbone_lr"]}, {"params": head_params, "lr": cfg["head_lr"]}])
    fwd = ft_forward_factory(frozen_until)
    res = run_config(cfg["name"], "VGG16-FineTuning", model, fwd, opt, vgg_train_dl, vgg_val_dl, cfg["epochs"],
                      {"unfreeze": list(cfg["unfreeze"]), "backbone_lr": cfg["backbone_lr"], "head_lr": cfg["head_lr"]},
                      f"'features' congelado hasta el indice {frozen_until}")
    res["frozen_until"] = frozen_until
    ft_results.append(res); ft_models[cfg["name"]] = model

print(f"\n[t={time.time()-t_script_start:.1f}s] FT listo")

# ---------------- seleccion + evaluacion final ----------------------------
def pick_best(results): return max(results, key=lambda r: r["best_val_metrics"]["f1"])
best_cnn = pick_best(cnn_results); best_fe = pick_best(fe_results); best_ft = pick_best(ft_results)
best_fe_model = fe_models[best_fe["name"]]; best_ft_model = ft_models[best_ft["name"]]

# reentrenar el mejor CNN (no se guardo el objeto modelo de cada iteracion para ahorrar memoria)
best_cnn_cfg = next(c for c in cnn_configs if c["name"] == best_cnn["name"])
best_cnn_model = CNNFromScratch(channels=best_cnn_cfg["channels"], dropout=best_cnn_cfg["dropout"]).to(DEVICE)
torch.manual_seed(SEED)
opt_cnn = torch.optim.Adam(best_cnn_model.parameters(), lr=best_cnn_cfg["lr"], weight_decay=best_cnn_cfg["weight_decay"])
_loader = cnn_train_dl if best_cnn_cfg["augment"] else cnn_train_noaug_dl
train_model(best_cnn_model, _loader, cnn_val_dl, cnn_forward, opt_cnn, CNN_EPOCHS)

def evaluate_final(model, forward_fn, loader):
    model.eval(); preds_all, labels_all = [], []; n=0; t0=time.time()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = forward_fn(model, x)
            preds_all.append(logits.argmax(1).cpu()); labels_all.append(y.cpu()); n += x.size(0)
    elapsed = time.time()-t0
    preds_all = torch.cat(preds_all).numpy(); labels_all = torch.cat(labels_all).numpy()
    acc = accuracy_score(labels_all, preds_all)
    p, r, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average="macro", zero_division=0)
    cm = confusion_matrix(labels_all, preds_all, labels=list(range(N_CLASSES)))
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "latency_ms_per_img": elapsed/n*1000}, cm

ft_best_forward = ft_forward_factory(best_ft["frozen_until"])
final_specs = {"CNN-scratch": (best_cnn_model, cnn_forward, cnn_test_dl),
               "VGG16-FeatureExtractor": (best_fe_model, fe_forward, vgg_test_dl),
               "VGG16-FineTuning": (best_ft_model, ft_best_forward, vgg_test_dl)}
final_test_results = {}
fig, axes = plt.subplots(1, 3, figsize=(16,5))
for ax, (arch, (model, fwd, loader)) in zip(axes, final_specs.items()):
    metrics, cm = evaluate_final(model, fwd, loader)
    final_test_results[arch] = {"metrics": metrics, "confusion_matrix": cm.tolist()}
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False, xticklabels=CLASSES, yticklabels=CLASSES)
    ax.set_title(arch, fontsize=9); ax.set_xlabel("predicho"); ax.set_ylabel("real")
    ax.tick_params(axis="x", rotation=90, labelsize=7); ax.tick_params(axis="y", rotation=0, labelsize=7)
    print(f"{arch}: acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} latency={metrics['latency_ms_per_img']:.3f}ms/img")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), dpi=120); plt.close()

print(f"\n[t={time.time()-t_script_start:.1f}s] Evaluacion final lista")

# ---------------- loss curves -----------------------------
def plot_loss_curves(results, arch_name, filename):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.6*n, 3.3))
    if n == 1: axes = [axes]
    for ax, r in zip(axes, results):
        ep = range(1, len(r["history"]["train_loss"])+1)
        ax.plot(ep, r["history"]["train_loss"], label="train")
        ax.plot(ep, r["history"]["val_loss"], label="val")
        ax.set_title(r["name"], fontsize=8); ax.set_xlabel("epoch")
    axes[0].set_ylabel("loss"); axes[0].legend()
    fig.suptitle(f"Curvas de perdida - {arch_name}")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, filename), dpi=120); plt.close()

plot_loss_curves(cnn_results, "CNN desde cero (3 iteraciones)", "loss_curves_cnn.png")
plot_loss_curves(fe_results, "VGG-16 Feature Extractor (3 iteraciones)", "loss_curves_fe.png")
plot_loss_curves(ft_results, "VGG-16 Fine-Tuning (3 iteraciones)", "loss_curves_ft.png")

# ---------------- experimento de cantidad de datos ------------------------
cnn_hp = best_cnn["hyperparams"]
cnn_tf10 = cnn_train_tf if cnn_hp["augment"] else cnn_eval_tf
cnn_train10_dl = DataLoader(TransformSubset(base_train, idx_train10_fast, cnn_tf10), batch_size=BATCH_CNN, shuffle=True)
model_cnn_dq = CNNFromScratch(channels=cnn_hp["channels"], dropout=cnn_hp["dropout"]).to(DEVICE)
opt_cnn_dq = torch.optim.Adam(model_cnn_dq.parameters(), lr=cnn_hp["lr"], weight_decay=cnn_hp["weight_decay"])
train_model(model_cnn_dq, cnn_train10_dl, cnn_val_dl, cnn_forward, opt_cnn_dq, CNN_EPOCHS)
metrics_cnn_dq, _ = evaluate_final(model_cnn_dq, cnn_forward, cnn_test_dl)

fe_hp = best_fe["hyperparams"]
vgg_train10_dl = DataLoader(TransformSubset(base_train, idx_train10_fast, vgg_train_tf), batch_size=BATCH_VGG, shuffle=True)
model_fe_dq = build_feature_extractor(fe_hp["variant"])
opt_fe_dq = torch.optim.Adam([p for p in model_fe_dq.parameters() if p.requires_grad], lr=fe_hp["lr"], weight_decay=fe_hp["weight_decay"])
train_model(model_fe_dq, vgg_train10_dl, vgg_val_dl, fe_forward, opt_fe_dq, FE_EPOCHS)
metrics_fe_dq, _ = evaluate_final(model_fe_dq, fe_forward, vgg_test_dl)

ft_hp = best_ft["hyperparams"]
model_ft_dq, frozen_until_dq = build_finetune(ft_hp["unfreeze"])
bp_dq = [p for p in model_ft_dq.features.parameters() if p.requires_grad]
hp_dq = list(model_ft_dq.classifier.parameters())
opt_ft_dq = torch.optim.Adam([{"params": bp_dq, "lr": ft_hp["backbone_lr"]}, {"params": hp_dq, "lr": ft_hp["head_lr"]}])
fwd_ft_dq = ft_forward_factory(frozen_until_dq)
train_model(model_ft_dq, vgg_train10_dl, vgg_val_dl, fwd_ft_dq, opt_ft_dq, 1)
metrics_ft_dq, _ = evaluate_final(model_ft_dq, fwd_ft_dq, vgg_test_dl)

data_quantity_rows = [
    {"modelo": "CNN-scratch", "n_train_100": len(idx_train_fast), "n_train_10": len(idx_train10_fast),
     "test_acc_100": final_test_results["CNN-scratch"]["metrics"]["accuracy"], "test_acc_10": metrics_cnn_dq["accuracy"],
     "test_f1_100": final_test_results["CNN-scratch"]["metrics"]["f1"], "test_f1_10": metrics_cnn_dq["f1"]},
    {"modelo": "VGG16-FeatureExtractor", "n_train_100": len(idx_train_fast), "n_train_10": len(idx_train10_fast),
     "test_acc_100": final_test_results["VGG16-FeatureExtractor"]["metrics"]["accuracy"], "test_acc_10": metrics_fe_dq["accuracy"],
     "test_f1_100": final_test_results["VGG16-FeatureExtractor"]["metrics"]["f1"], "test_f1_10": metrics_fe_dq["f1"]},
    {"modelo": "VGG16-FineTuning", "n_train_100": len(idx_train_fast), "n_train_10": len(idx_train10_fast),
     "test_acc_100": final_test_results["VGG16-FineTuning"]["metrics"]["accuracy"], "test_acc_10": metrics_ft_dq["accuracy"],
     "test_f1_100": final_test_results["VGG16-FineTuning"]["metrics"]["f1"], "test_f1_10": metrics_ft_dq["f1"]},
]
print(f"\n[t={time.time()-t_script_start:.1f}s] Experimento de cantidad de datos listo")

# ---------------- FLOPs / comparacion --------------------------------------
def compute_macs(model, input_size):
    info = summary(model, input_size=input_size, verbose=0, col_names=("num_params","mult_adds"))
    return float(info.total_mult_adds)

macs_per_model = {
    "CNN-scratch": compute_macs(best_cnn_model, (1,3,32,32)),
    "VGG16-FeatureExtractor": compute_macs(best_fe_model, (1,3,RES,RES)),
    "VGG16-FineTuning": compute_macs(best_ft_model, (1,3,RES,RES)),
}
train_flops_multiplier = {"CNN-scratch": 3.0, "VGG16-FeatureExtractor": 1.05, "VGG16-FineTuning": 1.3}
results_map = {"CNN-scratch": best_cnn, "VGG16-FeatureExtractor": best_fe, "VGG16-FineTuning": best_ft}
comparison_rows = []
for arch, res in results_map.items():
    macs = macs_per_model[arch]; flops_inf = 2*macs
    n_img = res["n_images_per_epoch"]
    train_flops_per_img = flops_inf * train_flops_multiplier[arch]
    total_train_flops = train_flops_per_img * n_img * res["best_epoch"]
    m = final_test_results[arch]["metrics"]
    comparison_rows.append({"modelo": arch, "total_params": res["total_params"], "trainable_params": res["trainable_params"],
                             "inference_MACs_img": macs, "inference_FLOPs_img": flops_inf, "latencia_ms_img": m["latency_ms_per_img"],
                             "tiempo_epoch_avg_s": res["epoch_time_avg_sec"], "epochs_hasta_mejor_val": res["best_epoch"],
                             "tiempo_total_entrenamiento_s": res["total_train_time_sec"], "peak_mem_mb": res["peak_mem_mb"],
                             "flops_entrenamiento_total_estimado": total_train_flops,
                             "test_accuracy": m["accuracy"], "test_precision": m["precision"], "test_recall": m["recall"], "test_f1": m["f1"]})
comparison_df = pd.DataFrame(comparison_rows)

fig, ax = plt.subplots(figsize=(7.5,5.5))
for arch, res in results_map.items():
    times = np.cumsum(res["history"]["epoch_time"]); accs = [m["accuracy"] for m in res["history"]["val_metrics"]]
    ax.plot(times, accs, marker="o", label=arch)
ax.set_xlabel("tiempo de entrenamiento acumulado (s)"); ax.set_ylabel("accuracy de validacion")
ax.set_title("Accuracy de validacion vs. tiempo de entrenamiento acumulado"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "val_acc_vs_time.png"), dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(6.5,5.5))
colors = sns.color_palette("deep", n_colors=len(comparison_df))
for (_, row), color in zip(comparison_df.iterrows(), colors):
    ax.scatter(row["inference_FLOPs_img"], row["test_f1"], s=90, color=color)
    ax.annotate(row["modelo"], (row["inference_FLOPs_img"], row["test_f1"]), textcoords="offset points", xytext=(6,6), fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("FLOPs de inferencia por imagen (log)"); ax.set_ylabel("F1 (macro) en test")
ax.set_title("F1 en test vs. costo computacional de inferencia")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "f1_vs_flops.png"), dpi=120); plt.close()

summary_rows = []
for r in cnn_results + fe_results + ft_results:
    summary_rows.append({"arquitectura": r["arch"], "iteración": r["name"], "total_params": r["total_params"],
                          "trainable_params": r["trainable_params"], "mejor_epoch": r["best_epoch"],
                          "val_accuracy": r["best_val_metrics"]["accuracy"], "val_precision": r["best_val_metrics"]["precision"],
                          "val_recall": r["best_val_metrics"]["recall"], "val_f1": r["best_val_metrics"]["f1"],
                          "tiempo_epoch_avg_s": r["epoch_time_avg_sec"], "tiempo_total_s": r["total_train_time_sec"],
                          "peak_mem_mb": r["peak_mem_mb"]})
summary_df = pd.DataFrame(summary_rows)
data_quantity_df = pd.DataFrame(data_quantity_rows)

def strip(o):
    if isinstance(o, dict): return {k: strip(v) for k,v in o.items()}
    if isinstance(o, list): return [strip(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o

full_dump = {"device": str(DEVICE), "resolution_vgg": RES, "n_train_fast": len(idx_train_fast),
             "n_val_fast": len(idx_val_fast), "n_test_fast": len(idx_test_fast), "n_train10_fast": len(idx_train10_fast),
             "cifar_mean": CIFAR_MEAN, "cifar_std": CIFAR_STD,
             "cnn_results": cnn_results, "fe_results": fe_results, "ft_results": ft_results,
             "best_cnn": best_cnn["name"], "best_fe": best_fe["name"], "best_ft": best_ft["name"],
             "final_test_results": final_test_results, "data_quantity": data_quantity_rows,
             "macs_per_model": macs_per_model}
with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
    json.dump(strip(full_dump), f, indent=2, default=str)
summary_df.to_csv(os.path.join(RESULTS_DIR, "summary_iterations.csv"), index=False)
comparison_df.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)
data_quantity_df.to_csv(os.path.join(RESULTS_DIR, "data_quantity.csv"), index=False)

print(f"\nTODO LISTO en t={time.time()-t_script_start:.1f}s. Resultados guardados en {os.path.abspath(RESULTS_DIR)}")

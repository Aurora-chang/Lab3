from IPython.display import display

import os, sys, json, time, copy, random, zipfile, subprocess, warnings
from pathlib import Path
import numpy as np, pandas as pd
import cv2
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import timm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
print("torch", torch.__version__, "| timm", timm.__version__, "| GPU:", torch.cuda.is_available() and torch.cuda.get_device_name(0))

COMP_SLUG   = "nycu-aimi-2026-lab3-cxr"
DATA_DIR    = "/content/data"
USE_DRIVE   = True
DRIVE_SUBDIR = "Lab3_v2"
IMG         = 224
EPOCHS      = 12
BATCH       = 32
SEED        = 42
VAL_FRACTION = 0.15
CLASSES     = ["normal", "bacteria", "virus", "COVID-19"]

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = Path("/content/drive/MyDrive") / DRIVE_SUBDIR / "outputs"
else:
    OUT = Path("/content/outputs")
OUT.mkdir(parents=True, exist_ok=True)
print("outputs ->", OUT)

from google.colab import userdata
os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()

slug = COMP_SLUG
if not slug:
    rc, out = sh("kaggle competitions list --group entered --csv")
    print(out[:1500])
    ent = pd.read_csv(__import__("io").StringIO(out))
    ent = ent[pd.to_datetime(ent["deadline"]) > pd.Timestamp.now()]
    rows = [r.split("/")[-1] for r in ent["ref"]]
    if len(rows) == 1:
        slug = rows[0]
    else:
        raise SystemExit("Could not pick the competition automatically. Set COMP_SLUG in the CONFIG cell "
                         "(the last part of the competition URL on Kaggle, e.g. kaggle.com/competitions/<slug>).")
print("competition:", slug)

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
if not any(Path(DATA_DIR).rglob("*.jp*g")) and not any(Path(DATA_DIR).rglob("*.png")):
    rc, out = sh(f"kaggle competitions download -c {slug} -p {DATA_DIR}")
    print(out)
    assert rc == 0, "Download failed: check the token secret and that you pressed Join Competition."
    for z in Path(DATA_DIR).glob("*.zip"):
        with zipfile.ZipFile(z) as f: f.extractall(DATA_DIR)
        z.unlink()
print("files:", sum(1 for _ in Path(DATA_DIR).rglob("*") if _.is_file()))

DATA = Path(DATA_DIR)
IMG_EXT = {".jpg", ".jpeg", ".png"}
all_imgs = [p for p in DATA.rglob("*") if p.suffix.lower() in IMG_EXT]
by_name = {}
for p in all_imgs: by_name.setdefault(p.name, p)
print("images found:", len(all_imgs))
print("top-level:", sorted(x.name for x in DATA.iterdir())[:20])
print("csv files:", [str(c.relative_to(DATA)) for c in DATA.rglob("*.csv")])

ALIASES = [("covid", "COVID-19"), ("normal", "normal"), ("bacteri", "bacteria"), ("virus", "virus"), ("viral", "virus")]
def alias(s):
    s = str(s).lower()
    for k, v in ALIASES:
        if k in s: return v
    return None
def split_of(p):
    parts = [x.lower() for x in p.relative_to(DATA).parts[:-1]]
    for part in reversed(parts):
        if part.startswith("train"): return "train"
        if part.startswith("val"):   return "val"
        if part.startswith("test"):  return "test"
    return None

rows = []
for p in all_imgs:
    sp = split_of(p)
    lab = next((alias(x) for x in reversed(p.relative_to(DATA).parts[:-1]) if alias(x)), None)
    if sp in ("train", "val") and lab: rows.append((sp, str(p), lab))
lab_df = pd.DataFrame(rows, columns=["split", "path", "label"])

if lab_df.empty:
    for c in DATA.rglob("*.csv"):
        if "sample" in c.name.lower() or "submission" in c.name.lower(): continue
        d = pd.read_csv(c)
        fcol = next((k for k in d.columns if "file" in k.lower() or "name" in k.lower() or "image" in k.lower()), d.columns[0])
        cls_cols = {k: alias(k) for k in d.columns if alias(k)}
        if len(set(cls_cols.values())) == 4:
            labs = d[list(cls_cols)].values.argmax(1); names = list(cls_cols.values())
            d["label"] = [names[i] for i in labs]
        else:
            lcol = next((k for k in d.columns if k.lower() in ("label", "class", "target")), None)
            if lcol is None: continue
            d["label"] = d[lcol].map(lambda v: alias(v) or v)
        d["path"] = d[fcol].map(lambda n: str(by_name.get(Path(str(n)).name, "")))
        d = d[d.path != ""]
        d["split"] = "val" if "val" in c.name.lower() else "train"
        lab_df = pd.concat([lab_df, d[["split", "path", "label"]]]); print("labels from", c.name)

assert not lab_df.empty, "Could not find labelled training images. Check the printout above and adapt this cell."
print(lab_df.groupby(["split", "label"]).size().unstack(fill_value=0))

train_df = lab_df[lab_df.split == "train"].reset_index(drop=True)
val_df = lab_df[lab_df.split == "val"].reset_index(drop=True)
if val_df.empty:
    train_df, val_df = train_test_split(train_df, test_size=VAL_FRACTION, stratify=train_df.label, random_state=SEED)
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    print("no val split in the download -> made a stratified split")

sample_paths = list(DATA.rglob("test_data_sample.csv")) + list((Path("/content/drive/MyDrive") / DRIVE_SUBDIR).glob("test_data_sample.csv"))
if sample_paths:
    test_names = pd.read_csv(sample_paths[0])["new_filename"].tolist(); print("test order from", sample_paths[0])
else:
    test_names = sorted(p.name for p in all_imgs if split_of(p) == "test"); print("WARNING: no sample csv found, using sorted test file names")
test_paths = [str(by_name[n]) for n in test_names]
print("train", len(train_df), "val", len(val_df), "test", len(test_paths))

_cache = {}
def load_set(paths, clahe):
    key = (hash(tuple(paths)), clahe)
    if key in _cache: return _cache[key]
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None
    arr = np.zeros((len(paths), IMG, IMG), np.uint8)
    for i, p in enumerate(paths):
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (IMG, IMG), interpolation=cv2.INTER_AREA)
        arr[i] = cl.apply(g) if cl else g
    _cache[key] = arr
    return arr

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF_TRAIN = T.Compose([T.RandomResizedCrop(IMG, scale=(0.75, 1.0), ratio=(0.9, 1.1)), T.RandomRotation(10),
                      T.ColorJitter(0.2, 0.2), T.ToTensor(), NORM])
TF_EVAL = T.Compose([T.ToTensor(), NORM])

class CXR(Dataset):
    def __init__(self, arr, labels, tf): self.arr, self.labels, self.tf = arr, labels, tf
    def __len__(self): return len(self.arr)
    def __getitem__(self, i):
        img = Image.fromarray(self.arr[i]).convert("RGB")
        return self.tf(img), (self.labels[i] if self.labels is not None else -1)

c2i = {c: i for i, c in enumerate(CLASSES)}
y_train = train_df.label.map(c2i).values; y_val = val_df.label.map(c2i).values
counts = np.bincount(y_train, minlength=4); print("train class counts:", dict(zip(CLASSES, counts)))

class Focal(nn.Module):
    def __init__(self, weight, gamma=2.0): super().__init__(); self.w, self.g = weight, gamma
    def forward(self, logits, y):
        lp = F.log_softmax(logits, 1); p = lp.exp()
        lpt = lp.gather(1, y[:, None])[:, 0]; pt = p.gather(1, y[:, None])[:, 0]
        w = self.w[y] if self.w is not None else 1.0
        return (-w * (1 - pt) ** self.g * lpt).mean()

def make_loss(name, power):
    w = torch.tensor((counts.sum() / (4 * counts)) ** power, dtype=torch.float32, device=DEV)
    if name == "ce":    return nn.CrossEntropyLoss()
    if name == "wce":   return nn.CrossEntropyLoss(weight=w)
    if name == "focal": return Focal(w)
    raise ValueError(name)

@torch.no_grad()
def predict(model, arr):
    model.eval(); out = []
    for x, _ in DataLoader(CXR(arr, None, TF_EVAL), batch_size=64, num_workers=2):
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            out.append(F.softmax(model(x.to(DEV)).float(), 1).cpu())
    return torch.cat(out).numpy()

RESULTS = {}
def run_exp(cfg):
    name = cfg["name"]; seed_all(SEED); t0 = time.time()
    cl = cfg.get("clahe", False)
    Xtr, Xva, Xte = load_set(list(train_df.path), cl), load_set(list(val_df.path), cl), load_set(test_paths, cl)
    ds = CXR(Xtr, y_train, TF_TRAIN)
    if cfg.get("sampler"):
        sw = (1.0 / counts)[y_train] ** 0.5
        dl = DataLoader(ds, BATCH, sampler=WeightedRandomSampler(sw, len(ds), replacement=True), num_workers=2, drop_last=True)
    else:
        dl = DataLoader(ds, BATCH, shuffle=True, num_workers=2, drop_last=True)
    model = timm.create_model(cfg["model"], pretrained=True, num_classes=4).to(DEV)
    crit = make_loss(cfg.get("loss", "ce"), cfg.get("wpow", 0.5))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 3e-4), total_steps=EPOCHS * len(dl), pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda")
    best, best_state, hist = -1, None, []
    for ep in range(EPOCHS):
        model.train(); tl = 0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item()
        vf1 = f1_score(y_val, predict(model, Xva).argmax(1), average="macro")
        hist.append((tl / len(dl), vf1))
        if vf1 > best: best, best_state = vf1, copy.deepcopy(model.state_dict())
        print(f"[{name}] ep {ep+1:02d}/{EPOCHS} loss {tl/len(dl):.4f} val macro-F1 {vf1:.4f}")
    model.load_state_dict(best_state)
    vp, tp = predict(model, Xva), predict(model, Xte)
    RESULTS[name] = dict(cfg=cfg, val_f1=best, val_probs=vp, test_probs=tp, hist=hist, minutes=(time.time() - t0) / 60,
                         per_class=f1_score(y_val, vp.argmax(1), average=None))
    np.savez_compressed(OUT / f"probs_{name}.npz", val=vp, test=tp)
    del model, opt; torch.cuda.empty_cache()
    print(f"[{name}] best val macro-F1 {best:.4f}  ({RESULTS[name]['minutes']:.1f} min)\n")

EXPERIMENTS = [
    dict(name="r50_ce",              model="resnet50",        loss="ce"),
    dict(name="r50_wce",             model="resnet50",        loss="wce"),
    dict(name="r50_wce_clahe",       model="resnet50",        loss="wce",   clahe=True),
    dict(name="r50_sampler_clahe",   model="resnet50",        loss="ce",    clahe=True, sampler=True),
    dict(name="effb0_wce_clahe",     model="efficientnet_b0", loss="wce",   clahe=True, lr=1e-3),
    dict(name="convnext_wce_clahe",  model="convnext_tiny",   loss="wce",   clahe=True, lr=1e-4),
    dict(name="convnext_focal_clahe",model="convnext_tiny",   loss="focal", clahe=True, lr=1e-4),
]
for cfg in EXPERIMENTS:
    if cfg["name"] in RESULTS: continue
    run_exp(cfg)

rows = []
for n, r in RESULTS.items():
    rows.append(dict(name=n, val_macroF1=r["val_f1"], **{f"F1_{c}": f for c, f in zip(CLASSES, r["per_class"])},
                     model=r["cfg"]["model"], loss=r["cfg"].get("loss", "ce"), clahe=r["cfg"].get("clahe", False),
                     sampler=r["cfg"].get("sampler", False), lr=r["cfg"].get("lr", 3e-4), minutes=round(r["minutes"], 1)))
tbl = pd.DataFrame(rows).sort_values("val_macroF1", ascending=False).reset_index(drop=True)
tbl.to_csv(OUT / "experiment_results.csv", index=False)
display(tbl.round(4))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].barh(tbl.name[::-1], tbl.val_macroF1[::-1]); ax[0].set_xlabel("val macro-F1"); ax[0].set_title("Experiments")
for n, r in RESULTS.items(): ax[1].plot([h[1] for h in r["hist"]], label=n)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val macro-F1"); ax[1].legend(fontsize=6); plt.tight_layout()
plt.savefig(OUT / "experiments.png", dpi=150); plt.show()

TOP_K = 3
order = list(tbl.name)
cands = {f"single:{order[0]}": ([order[0]])}
for k in range(2, min(TOP_K, len(order)) + 1): cands[f"ensemble top{k}"] = order[:k]
scores = {}
for label, names in cands.items():
    vp = np.mean([RESULTS[n]["val_probs"] for n in names], 0)
    scores[label] = f1_score(y_val, vp.argmax(1), average="macro"); print(f"{label:28s} val macro-F1 {scores[label]:.4f}")
best_label = max(scores, key=scores.get); names = cands[best_label]
print("\nFINAL:", best_label, names)

vp = np.mean([RESULTS[n]["val_probs"] for n in names], 0); tp = np.mean([RESULTS[n]["test_probs"] for n in names], 0)
print(classification_report(y_val, vp.argmax(1), target_names=CLASSES, digits=4))
cm = confusion_matrix(y_val, vp.argmax(1))
plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="Blues"); plt.xticks(range(4), CLASSES, rotation=30); plt.yticks(range(4), CLASSES)
for i in range(4):
    for j in range(4): plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Validation confusion matrix"); plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=150); plt.show()

pred = tp.argmax(1)
sub = pd.DataFrame({"new_filename": test_names})
for i, c in enumerate(CLASSES): sub[c] = (pred == i).astype(int)
sub.to_csv(OUT / "submission.csv", index=False)
print(sub.head()); print("predicted class share:", dict(zip(CLASSES, (np.bincount(pred, minlength=4) / len(pred)).round(3))))
print("saved:", OUT / "submission.csv")

import shutil
json.dump(dict(EPOCHS=EPOCHS, BATCH=BATCH, IMG=IMG, SEED=SEED, final=best_label, final_models=names,
               experiments=EXPERIMENTS, val_scores=scores), open(OUT / "run_config.json", "w"), indent=2)
zip_path = shutil.make_archive("/content/lab3_results", "zip", OUT)
print("zip:", zip_path, f"({os.path.getsize(zip_path)/1e6:.1f} MB)"); print(sorted(os.listdir(OUT)))
from google.colab import files
files.download(zip_path)


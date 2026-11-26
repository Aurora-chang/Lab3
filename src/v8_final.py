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
import torchxrayvision as xrv
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
print("torch", torch.__version__, "| timm", timm.__version__, "| GPU:", torch.cuda.is_available() and torch.cuda.get_device_name(0))

COMP_SLUG   = "nycu-aimi-2026-lab3-cxr"
DATA_DIR    = "/content/data"
USE_DRIVE   = True
DRIVE_SUBDIR = "Lab3_v2"
IMG         = 288
EPOCHS      = 20
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

TEST_ORDER = """1736.jpeg 3397.jpeg 1446.jpeg 4619.jpeg 5639.jpeg 956.jpeg 3376.jpeg 4151.jpeg 676.jpeg 4821.jpeg 632.jpeg 1263.jpeg 2044.jpeg 1819.jpeg 3065.jpeg 5690.jpeg 2675.jpeg 2466.jpeg 3369.jpeg 2959.jpeg 3977.jpeg 2653.jpeg 4033.jpeg 1739.jpeg 2871.jpeg 2142.jpeg 747.jpeg 4466.jpeg 2765.jpeg 3550.jpeg 3354.jpeg 1761.jpeg 4634.jpeg 3447.jpeg 3093.jpeg 1345.jpeg 4070.jpeg 2158.jpeg 2420.jpeg 2344.jpeg 3121.jpeg 3021.jpeg 2718.jpeg 5751.jpeg 3971.jpeg 175.jpeg 5009.jpeg 5623.jpeg 5210.jpeg 4436.jpeg 2893.jpeg 716.jpeg 5536.jpeg 2464.jpeg 5151.jpeg 1863.jpeg 3481.jpeg 194.jpeg 1662.jpeg 3952.jpeg 2166.jpeg 3762.jpeg 5694.jpeg 1048.jpeg 204.jpeg 1486.jpeg 5172.jpeg 3972.jpeg 512.jpeg 441.jpeg 3848.jpeg 1676.jpeg 1459.jpeg 3039.jpeg 2405.jpeg 3522.jpeg 3943.jpeg 598.jpeg 2791.jpeg 1368.jpeg 5860.jpeg 1574.jpeg 989.jpeg 1684.jpeg 5143.jpeg 1350.jpeg 2695.jpeg 397.jpeg 3782.jpeg 5803.jpeg 2995.jpeg 4664.jpeg 3046.jpeg 655.jpeg 3150.jpeg 1559.jpeg 172.jpeg 3520.jpeg 304.jpeg 4098.jpeg 4304.jpeg 3996.jpeg 1774.jpeg 2105.jpeg 4832.jpeg 3476.jpeg 3744.jpeg 3290.jpeg 4988.jpeg 5804.jpeg 1695.jpeg 3635.jpeg 4474.jpeg 1756.jpeg 3101.jpeg 3786.jpeg 4271.jpeg 2191.jpeg 4370.jpeg 445.jpeg 3331.jpeg 864.jpeg 3455.jpeg 2570.jpeg 5478.jpeg 4953.jpeg 3984.jpeg 3408.jpeg 278.jpeg 10.jpeg 4375.jpeg 5814.jpeg 2190.jpeg 4114.jpeg 2039.jpeg 1282.jpeg 4237.jpeg 2890.jpeg 2757.jpeg 4801.jpeg 4274.jpeg 4238.jpeg 3260.jpeg 2153.jpeg 3003.jpeg 308.jpeg 3603.jpeg 5079.jpeg 1077.jpeg 213.jpeg 5718.jpeg 5201.jpeg 5516.jpeg 3025.jpeg 1201.jpeg 2656.jpeg 1151.jpeg 1398.jpeg 3871.jpeg 1217.jpeg 3037.jpeg 2878.jpeg 4289.jpeg 631.jpeg 7.jpg 3500.jpeg 1855.jpeg 3909.jpeg 2517.jpeg 2338.jpeg 4327.jpeg 2052.jpeg 4585.jpeg 776.jpeg 3338.jpeg 4514.jpeg 4596.jpeg 2391.jpeg 371.jpeg 5073.jpeg 139.jpeg 3821.jpeg 4937.jpeg 955.jpeg 196.jpeg 4150.jpeg 3753.jpeg 5732.jpeg 4799.jpeg 1466.jpeg 18.jpeg 4496.jpeg 4154.jpeg 2116.jpeg 2722.jpeg 5887.jpeg 4679.jpeg 3055.jpeg 4575.jpeg 5716.jpeg 2136.jpeg 996.jpeg 2794.jpeg 449.jpeg 2162.jpeg 111.jpeg 2335.jpeg 4837.jpeg 980.jpeg 793.jpeg 4128.jpeg 5412.jpeg 2585.jpeg 4690.jpeg 4622.jpeg 3874.jpeg 537.jpeg 5483.jpeg 958.jpeg 5023.jpeg 5875.jpeg 4645.jpeg 4863.jpeg 1598.jpeg 5262.jpeg 20.jpeg 2980.jpeg 5286.jpeg 2454.jpeg 4944.jpeg 1440.jpeg 4362.jpeg 1983.jpeg 2308.jpeg 575.jpeg 5381.jpeg 22.jpeg 1966.jpeg 5788.jpeg 1303.jpeg 3978.jpeg 3589.jpeg 5182.jpeg 3618.jpeg 900.jpeg 5573.jpeg 3339.jpeg 2144.jpeg 2332.jpeg 826.jpeg 116.jpeg 874.jpeg 3733.jpeg 5862.jpeg 5329.jpeg 3889.jpeg 2894.jpeg 420.jpeg 1370.jpeg 5429.jpeg 4322.jpeg 2716.jpeg 5226.jpeg 4911.jpeg 1259.jpeg 519.jpeg 2286.jpeg 4367.jpeg 2031.jpeg 4560.jpeg 3958.jpeg 2262.jpeg 287.jpeg 268.jpeg 482.jpeg 4629.jpeg 691.jpeg 2234.jpeg 791.jpeg 5333.jpeg 5539.jpeg 3647.jpeg 856.jpeg 5846.jpeg 1007.jpeg 3175.jpeg 1395.jpeg 2419.jpeg 2918.jpeg 565.jpeg 4382.jpeg 2199.jpeg 5281.jpeg 5880.jpeg 617.jpeg 4545.jpeg 324.jpeg 4675.jpeg 4693.jpeg 4583.jpeg 4598.jpeg 4751.jpeg 5709.jpeg 3328.jpeg 4951.jpeg 731.jpeg 1031.jpeg 3052.jpeg 349.jpeg 76.jpeg 1012.jpeg 4765.jpeg 246.jpeg 5319.jpeg 4578.jpeg 5836.jpeg 423.jpeg 3363.jpeg 1047.jpeg 1210.jpeg 5523.jpeg 4632.jpeg 227.jpeg 2525.jpeg 51.jpg 428.jpeg 728.jpeg 4133.jpeg 618.jpeg 5446.jpeg 262.jpeg 2907.jpeg 5199.jpeg 3516.jpeg 5813.jpeg 4486.jpeg 2663.jpeg 5020.jpeg 3740.jpeg 2637.jpeg 2556.jpeg 944.jpeg 2414.jpeg 5883.jpeg 2554.jpeg 5517.jpeg 952.jpeg 3907.jpeg 5743.jpeg 3028.jpeg 3593.jpeg 1004.jpeg 4762.jpeg 5726.jpeg 173.jpeg 2619.jpeg 3274.jpeg 486.jpeg 2533.jpeg 3342.jpeg 2300.jpeg 1730.jpeg 4328.jpeg 4397.jpeg 2934.jpeg 4004.jpeg 3209.jpeg 4153.jpeg 3489.jpeg 5253.jpeg 3079.jpeg 3426.jpeg 4961.jpeg 1763.jpeg 1519.jpeg 5824.jpeg 2557.jpeg 3307.jpeg 2990.jpeg 1638.jpeg 4191.jpeg 1648.jpeg 1019.jpeg 5269.jpeg 4428.jpeg 5352.jpeg 3619.jpeg 1400.jpeg 1184.jpeg 3717.jpeg 374.jpeg 3317.jpeg 5506.jpeg 5345.jpeg 3805.jpeg 2988.jpeg 1337.jpeg 1793.jpeg 4411.jpeg 5773.jpeg 3323.jpeg 4359.jpeg 995.jpeg 3464.jpeg 3523.jpeg 2513.jpeg 2530.jpeg 1522.jpeg 3966.jpeg 1206.jpeg 5543.jpeg 4255.jpeg 689.jpeg 3188.jpeg 2962.jpeg 5535.jpeg 5175.jpeg 1932.jpeg 912.jpeg 4500.jpeg 1431.jpeg 2645.jpeg 1194.jpeg 913.jpeg 1971.jpeg 2407.jpeg 1424.jpeg 393.jpeg 5276.jpg 2596.jpeg 3327.jpeg 5040.jpeg 1032.jpeg 5378.jpeg 775.jpeg 2228.jpeg 3257.jpeg 3146.jpeg 4933.jpeg 1056.jpeg 1656.jpeg 4077.jpeg 851.jpeg 171.jpeg 630.jpeg 959.jpeg 591.jpeg 2913.jpeg 1008.jpeg 5757.jpeg 1173.jpeg 3044.jpeg 2211.jpeg 5615.jpeg 1510.jpeg 462.jpeg 223.jpeg 5712.jpeg 2898.jpeg 4451.jpeg 1318.jpeg 5094.jpeg 5271.jpeg 1740.jpeg 725.jpeg 5505.jpeg 2285.jpeg 1995.jpeg 5115.jpeg 1566.jpeg 352.jpeg 4627.jpeg 4862.jpeg 362.jpeg 3334.jpeg 4830.jpeg 2486.jpeg 5207.jpeg 4003.jpeg 5638.jpeg 1993.jpeg 3151.jpeg 4229.jpeg 3245.jpeg 1822.jpeg 5110.jpeg 945.jpeg 1857.jpeg 452.jpeg 2795.jpeg 3032.jpeg 443.jpeg 558.jpeg 5729.jpeg 2828.jpeg 3929.jpeg 3844.jpeg 5896.jpeg 5379.jpeg 1703.jpeg 5552.jpeg 2365.jpeg 4392.jpeg 2738.jpeg 5624.jpeg 2366.jpeg 5197.jpeg 1498.jpeg 5068.jpeg 14.jpeg 5209.jpeg 1788.jpeg 5504.jpeg 5719.jpeg 1682.jpeg 3024.jpeg 594.jpeg 2035.jpeg 101.jpeg 1484.jpeg 1292.jpeg 1575.jpeg 586.jpeg 4178.jpeg 3139.jpeg 2208.jpeg 345.jpeg 4606.jpeg 57.jpeg 134.jpeg 1092.jpeg 1672.jpeg 3187.jpeg 3937.jpeg 2953.jpeg 4570.jpeg 5015.jpeg 4935.jpeg 2681.jpeg 2382.jpeg 861.jpeg 4291.jpeg 742.jpeg 4461.jpeg 4318.jpeg 1990.jpeg 3816.jpeg 982.jpeg 1143.jpeg 3253.jpeg 2993.jpeg 5370.jpeg 5748.jpeg 5901.jpeg 2746.jpeg 302.jpeg 5812.jpeg 1715.jpeg 3849.jpeg 3957.jpeg 5597.jpeg 848.jpeg 3892.jpeg 4368.jpeg 5075.jpeg 387.jpeg 5176.jpeg 357.jpeg 3207.jpeg 1154.jpeg 2970.jpeg 1174.jpeg 4651.jpeg 4169.jpeg 5631.jpeg 5102.jpeg 5576.jpeg 5283.jpeg 2997.jpeg 469.jpeg 640.jpeg 3506.jpeg 5733.jpeg 2610.jpeg 2113.jpeg 3388.jpeg 1661.jpeg 3227.jpeg 4002.jpeg 2463.jpeg 1639.jpeg 5734.jpeg 161.jpeg 4251.jpeg 150.jpeg 4581.jpeg 5044.jpeg 4649.jpeg 4788.jpeg 440.jpeg 4389.jpeg 2529.jpeg 252.jpeg 891.jpeg 3928.jpeg 3879.jpeg 3305.jpeg 3713.jpeg 1256.jpeg 5398.jpeg 5774.jpeg 3558.jpeg 277.jpeg 2594.jpeg 3478.jpeg 707.jpeg 4617.jpeg 1130.jpeg 5617.jpeg 506.jpeg 4321.jpeg 1315.jpeg 4741.jpeg 1829.jpeg 3475.jpeg 585.jpeg 3752.jpeg 878.jpeg 5577.jpeg 2201.jpeg 556.jpeg 599.jpeg 3264.jpeg 2605.jpeg 5127.jpeg 1375.jpeg 3069.jpeg 3191.jpeg 1769.jpeg 310.jpeg 1757.jpeg 28.jpeg 82.jpeg 392.jpeg 3470.jpeg 4480.jpeg 5588.jpeg 3043.jpeg 3198.jpeg 5414.jpeg 1121.jpeg 3574.jpeg 380.jpeg 860.jpeg 4204.jpeg 2943.jpeg 473.jpeg 4395.jpeg 218.jpeg 701.jpeg 2731.jpeg 2752.jpeg 5067.jpeg 4669.jpeg 3993.jpeg 2322.jpeg 5491.jpeg 60.jpeg 4447.jpeg 2532.jpeg 1944.jpeg 4584.jpeg 3571.jpeg 5755.jpeg 3595.jpeg 925.jpeg 4567.jpeg 752.jpeg 4511.jpeg 5290.jpeg 3373.jpeg 2475.jpeg 3923.jpeg 5892.jpeg 2796.jpeg 1681.jpeg 4940.jpeg 3534.jpeg 1470.jpeg 4761.jpeg 2940.jpeg 3800.jpeg 5838.jpeg 784.jpeg 2313.jpeg 516.jpeg 950.jpeg 3938.jpeg 3547.jpeg 4843.jpeg 4124.jpeg 1088.jpeg 307.jpeg 4221.jpeg 4914.jpeg 3009.jpeg 4384.jpeg 2392.jpeg 261.jpeg 4464.jpeg 2679.jpeg 2976.jpeg 4156.jpeg 5587.jpeg 2267.jpeg 2252.jpeg 5126.jpeg 1316.jpeg 2824.jpeg 3082.jpeg 3053.jpeg 1467.jpeg 5105.jpeg 3599.jpeg 3351.jpeg 5476.jpeg 2121.jpeg 396.jpeg 3779.jpeg 692.jpeg 5010.jpeg 5240.jpeg 2504.jpeg 2626.jpeg 916.jpeg 5140.jpeg 4662.jpeg 2068.jpeg 4924.jpeg 5785.jpeg 2717.jpeg 628.jpeg 3394.jpeg 4985.jpeg 4653.jpeg 3341.jpeg 1737.jpeg 2448.jpeg 3440.jpeg 2706.jpeg 3237.jpeg 5450.jpeg 4042.jpeg 3488.jpeg 5123.jpeg 3818.jpeg 5417.jpeg 2349.jpeg 4247.jpeg 5092.jpeg 2410.jpeg 2612.jpeg 2156.jpeg 3321.jpeg 2336.jpeg 3122.jpeg 3670.jpeg 1633.jpeg 2935.jpeg 2501.jpeg 5705.jpeg 3164.jpeg 5500.jpeg 1916.jpeg 915.jpeg 1445.jpeg 547.jpeg 1807.jpeg 5698.jpeg 2547.jpeg 745.jpeg 5879.jpeg 4071.jpeg 1298.jpeg 1693.jpeg 4442.jpeg 1129.jpeg 1889.jpeg 1288.jpeg 2676.jpeg 4181.jpeg 2345.jpeg 3345.jpeg 5416.jpeg 4874.jpeg 1624.jpeg 356.jpeg 2385.jpeg 2781.jpeg 113.jpeg 2353.jpeg 3969.jpeg 3005.jpeg 974.jpeg 2010.jpeg 5099.jpeg 1909.jpeg 2520.jpeg 2125.jpeg 2772.jpeg 5816.jpeg 4689.jpeg 4413.jpeg 118.jpeg 2713.jpeg 2966.jpeg 5833.jpeg 125.jpeg 5490.jpeg 5779.jpeg 1428.jpeg 2006.jpeg 1591.jpeg 4284.jpeg 128.jpeg 2733.jpeg 4615.jpeg 5255.jpeg 5301.jpeg 1357.jpeg 5230.jpeg 1280.jpeg 390.jpeg 789.jpeg 2710.jpeg 3990.jpeg 2131.jpeg 3695.jpeg 4186.jpeg 1015.jpeg 3728.jpeg 971.jpeg 195.jpeg 1813.jpeg 837.jpeg 808.jpeg 4276.jpeg 4356.jpeg 3325.jpeg 3555.jpeg 1687.jpeg 4120.jpeg 3074.jpeg 3326.jpeg 5224.jpeg 2509.jpeg 1491.jpeg 673.jpeg 5111.jpeg 3521.jpeg 3389.jpeg 2261.jpeg 38.jpeg 5243.jpeg 5144.jpeg 1462.jpeg 542.jpeg 4702.jpeg 3023.jpeg 408.jpeg 4355.jpeg 5173.jpeg 242.jpeg 515.jpeg 4365.jpeg 4507.jpeg 3468.jpeg 4628.jpeg 885.jpeg 2470.jpeg 751.jpeg 1904.jpeg 708.jpeg 3642.jpeg 2861.jpeg 1205.jpeg 2205.jpeg 97.jpeg 1059.jpeg 3332.jpeg 3126.jpeg 697.jpeg 4056.jpeg 4345.jpeg 5070.jpeg 999.jpeg 687.jpeg 2985.jpeg 5300.jpeg 4023.jpeg 3935.jpeg 119.jpeg 3997.jpeg 3462.jpeg 4533.jpeg 3137.jpeg 5265.jpeg 4818.jpeg 2902.jpeg 2314.jpeg 3352.jpeg 1339.jpeg 2434.jpeg 4990.jpeg 3675.jpeg 3608.jpeg 1014.jpeg 4852.jpeg 1425.jpeg 895.jpeg 1953.jpeg 5347.jpeg 3954.jpeg 5439.jpeg 4366.jpeg 2577.jpeg 2941.jpeg 2225.jpeg 2702.jpeg 3411.jpeg 5648.jpeg 3802.jpeg 1289.jpeg 3703.jpeg 1989.jpeg 3148.jpeg 3097.jpeg 2740.jpeg 2523.jpeg 4786.jpeg 2816.jpeg 1172.jpeg 3612.jpeg 901.jpeg 4936.jpeg 2178.jpeg 5327.jpeg 2400.jpeg 5654.jpeg 5609.jpeg 5832.jpeg 4861.jpeg 5586.jpeg 198.jpeg 153.jpeg 2864.jpeg 42.png 3193.jpeg 5519.jpeg 4954.jpeg 5157.jpeg 405.jpeg 2042.jpeg 458.jpeg 3527.jpeg 3111.jpeg 2204.jpeg 1880.jpeg 4401.jpeg 131.jpeg 614.jpeg 114.jpeg 1490.jpeg 4816.jpeg 1414.jpeg 1596.jpeg 4501.jpeg 453.jpeg 665.jpeg 1675.jpeg 4338.jpeg 4425.jpeg 5471.jpeg 4478.jpeg 4155.jpeg 4117.jpeg 2237.jpeg 3047.jpeg 571.jpeg 251.jpeg 2062.jpeg 1997.jpeg 4484.jpeg 2811.jpeg 228.jpeg 2040.jpeg 4360.jpeg 3416.jpeg 2689.jpeg 3512.jpeg 4403.jpeg 3250.jpeg 3959.jpeg 1405.jpeg 3181.jpeg 2021.jpeg 4008.jpeg 5823.jpeg 2853.jpeg 5843.jpeg 1759.jpeg 1313.jpeg 2212.jpeg 4906.jpeg 3020.jpeg 3048.jpeg 2057.jpeg 1049.jpeg 2453.jpeg 1604.jpeg 3998.jpeg 3441.jpeg 2789.jpeg 5889.jpeg 5403.jpeg 2045.jpeg 5878.jpeg 3577.jpeg 1937.jpeg 3428.jpeg 5464.jpeg 4885.jpeg 5074.jpeg 3234.jpeg 3131.jpeg 960.jpeg 815.jpeg 4967.jpeg 5656.jpeg 773.jpeg 3922.jpeg 3594.jpeg 1709.jpeg 562.jpeg 1346.jpeg 2087.jpeg 2931.jpeg 3419.jpeg 2633.jpeg 2960.jpeg 250.jpeg 2384.jpeg 3891.jpeg 1816.jpeg 1069.jpeg 5682.jpeg 3240.jpeg 2586.jpeg 882.jpeg 3648.jpeg 2514.jpeg 2085.jpeg 1573.jpeg 2840.jpeg 1003.jpeg 5568.jpeg 3199.jpeg 3186.jpeg 589.jpeg 3732.jpeg 4523.jpeg 2203.jpeg 1208.jpeg 4680.jpeg 2916.jpeg 3790.jpeg 1864.jpeg 4784.jpeg 5419.jpeg 2441.jpeg 296.jpeg 4048.jpeg 1140.jpeg 977.jpeg 1236.jpeg 3650.jpeg 5474.jpeg 232.jpeg 1924.jpeg 3722.jpeg 5775.jpeg 597.jpeg 5239.jpeg 1272.jpeg 4036.jpeg 4277.jpeg 84.jpeg 3382.jpeg 2386.jpeg 5217.jpeg 962.jpeg 1203.jpeg 1615.jpeg 264.jpeg 4272.jpeg 5459.jpeg 838.jpeg 1421.jpeg 735.jpeg 5436.jpeg 5320.jpeg 1433.jpeg 1363.jpeg 4957.jpeg 4907.jpeg 1877.jpeg 2880.jpeg 3767.jpeg 4902.jpeg 1051.jpeg 5090.jpeg 5228.jpeg 1447.jpeg 3778.jpeg 1848.jpeg 1869.jpeg 4314.jpeg 5160.jpeg 4112.jpeg 740.jpeg 5072.jpeg 2899.jpeg 3798.jpeg 2181.jpeg 1770.jpeg 4743.jpeg 715.jpeg 2480.jpeg 1772.jpeg 3379.jpeg 650.jpeg 5174.jpeg 1458.jpeg 3493.jpeg 263.jpeg 5560.jpeg 3132.jpeg 1225.jpeg 3089.jpeg 47.png 3437.jpeg 5425.jpeg 1678.jpeg 572.jpeg 889.jpeg 3368.jpeg 1753.jpeg 4122.jpeg 987.jpeg 5231.jpeg 4688.jpeg 4374.jpeg 3737.jpeg 5551.jpeg 1095.jpeg 4749.jpeg 3807.jpeg 74.jpeg 2279.jpeg 388.jpeg 766.jpeg 492.jpeg 4813.jpeg 4931.jpeg 1562.jpeg 2389.jpeg 422.jpeg 4992.jpeg 5028.jpeg 4148.jpeg 1436.jpeg 4767.jpeg"""
test_names = TEST_ORDER.split()
missing = [n for n in test_names if n not in by_name]
assert not missing, f"{len(missing)} test images from the sample order are not in the download, e.g. {missing[:3]}"
test_paths = [str(by_name[n]) for n in test_names]
print("train", len(train_df), "val", len(val_df), "test", len(test_paths))

_cache = {}
def load_set(paths, clahe, img):
    key = (hash(tuple(paths)), clahe, img)
    if key in _cache: return _cache[key]
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None
    arr = np.zeros((len(paths), img, img), np.uint8)
    for i, p in enumerate(paths):
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (img, img), interpolation=cv2.INTER_AREA)
        arr[i] = cl.apply(g) if cl else g
    _cache[key] = arr
    return arr

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
def tf_train(img):
    return T.Compose([T.RandomResizedCrop(img, scale=(0.75, 1.0), ratio=(0.9, 1.1)), T.RandomRotation(10),
                      T.ColorJitter(0.2, 0.2), T.ToTensor(), NORM])
TF_EVAL = T.Compose([T.ToTensor(), NORM])

class CXR(Dataset):
    def __init__(self, arr, labels, tf): self.arr, self.labels, self.tf = arr, labels, tf
    def __len__(self): return len(self.arr)
    def __getitem__(self, i):
        img = Image.fromarray(self.arr[i]).convert("RGB")
        return self.tf(img), (self.labels[i] if self.labels is not None else -1)

c2i = {c: i for i, c in enumerate(CLASSES)}

class XrvNet(nn.Module):
    is_xrv = True
    def __init__(self):
        super().__init__()
        self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.drop = nn.Dropout(0.2); self.head = nn.Linear(1024, 4)
    def forward(self, x): return self.head(self.drop(self.backbone.features2(x)))

def xrv_scale(x): return (2 * x - 1) * 1024
def tf_train_x(img):
    return T.Compose([T.RandomResizedCrop(img, scale=(0.75, 1.0), ratio=(0.9, 1.1)), T.RandomRotation(10),
                      T.ColorJitter(0.2, 0.2), T.ToTensor(), xrv_scale])
TF_EVAL_X = T.Compose([T.ToTensor(), xrv_scale])

class CXRL(Dataset):
    def __init__(self, arr, labels, tf): self.arr, self.labels, self.tf = arr, labels, tf
    def __len__(self): return len(self.arr)
    def __getitem__(self, i): return self.tf(Image.fromarray(self.arr[i])), (self.labels[i] if self.labels is not None else -1)

def is_xrv_cfg(cfg): return cfg["model"] == "xrv_densenet121"
def build_model(cfg):
    return XrvNet() if is_xrv_cfg(cfg) else timm.create_model(cfg["model"], pretrained=True, num_classes=cfg.get("n_classes", 4))
def eval_ds(model, arr):
    return CXRL(arr, None, TF_EVAL_X) if getattr(model, "is_xrv", False) else CXR(arr, None, TF_EVAL)

class Focal(nn.Module):
    def __init__(self, weight, gamma=2.0): super().__init__(); self.w, self.g = weight, gamma
    def forward(self, logits, y):
        lp = F.log_softmax(logits, 1); p = lp.exp()
        lpt = lp.gather(1, y[:, None])[:, 0]; pt = p.gather(1, y[:, None])[:, 0]
        w = self.w[y] if self.w is not None else 1.0
        return (-w * (1 - pt) ** self.g * lpt).mean()

def make_loss(name, power, counts):
    w = torch.tensor((counts.sum() / (len(counts) * counts)) ** power, dtype=torch.float32, device=DEV)
    if name == "ce":    return nn.CrossEntropyLoss()
    if name == "wce":   return nn.CrossEntropyLoss(weight=w)
    if name == "focal": return Focal(w)
    raise ValueError(name)

@torch.no_grad()
def predict(model, arr):
    model.eval(); out = []
    for x, _ in DataLoader(eval_ds(model, arr), batch_size=64, num_workers=2):
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            out.append(F.softmax(model(x.to(DEV)).float(), 1).cpu())
    return torch.cat(out).numpy()

def train_model(cfg, Xtr, ytr, Xva=None, yva=None, select=True):
    name = cfg["name"]; seed_all(SEED)
    epochs = cfg.get("epochs", EPOCHS); counts = np.bincount(ytr, minlength=cfg.get("n_classes", 4))
    ds = CXRL(Xtr, ytr, tf_train_x(Xtr.shape[1])) if is_xrv_cfg(cfg) else CXR(Xtr, ytr, tf_train(Xtr.shape[1]))
    if cfg.get("sampler"):
        sw = (1.0 / counts)[ytr] ** 0.5
        dl = DataLoader(ds, cfg.get("batch", BATCH), sampler=WeightedRandomSampler(sw, len(ds), replacement=True), num_workers=2, drop_last=True)
    else:
        dl = DataLoader(ds, cfg.get("batch", BATCH), shuffle=True, num_workers=2, drop_last=True)
    model = build_model(cfg).to(DEV)
    crit = make_loss(cfg.get("loss", "ce"), cfg.get("wpow", 0.5), counts)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 3e-4), total_steps=epochs * len(dl), pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda")
    best, best_state, hist = -1, None, []
    for ep in range(epochs):
        model.train(); tl = 0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item()
        if Xva is not None:
            vf1 = f1_score(yva, predict(model, Xva).argmax(1), average="macro"); hist.append((tl / len(dl), vf1))
            if select and vf1 > best: best, best_state = vf1, copy.deepcopy(model.state_dict())
            print(f"[{name}] ep {ep+1:02d}/{epochs} loss {tl/len(dl):.4f} val macro-F1 {vf1:.4f}")
        else:
            hist.append((tl / len(dl), None)); print(f"[{name}] ep {ep+1:02d}/{epochs} loss {tl/len(dl):.4f}")
    if best_state is not None: model.load_state_dict(best_state)
    return model, best, hist


@torch.no_grad()
def predict_both(model, arr):
    model.eval(); plain, tta = [], []
    for x, _ in DataLoader(eval_ds(model, arr), batch_size=64, num_workers=2):
        x = x.to(DEV); h = x.shape[-1]; c = int(h * 0.9); o = (h - c) // 2
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            p0 = F.softmax(model(x).float(), 1)
            xc = F.interpolate(x[:, :, o:o + c, o:o + c], size=(h, h), mode="bilinear", align_corners=False)
            p1 = F.softmax(model(xc).float(), 1)
        plain.append(p0.cpu()); tta.append(((p0 + p1) / 2).cpu())
    return torch.cat(plain).numpy(), torch.cat(tta).numpy()

pool_df = pd.concat([train_df, val_df]).reset_index(drop=True)
y_pool = pool_df.label.map(c2i).values
print("pool:", len(pool_df), dict(zip(CLASSES, np.bincount(y_pool, minlength=4))))

ALL_PATHS = list(pool_df.path) + test_paths
box_file = OUT / "lung_boxes.npz"
key = np.array([Path(p).name for p in ALL_PATHS])
boxes = None
if box_file.exists():
    d = np.load(box_file, allow_pickle=True)
    if len(d["key"]) == len(key) and (d["key"] == key).all(): boxes = d["boxes"]; print("lung boxes loaded from Drive")
if boxes is None:
    seg = xrv.baseline_models.chestx_det.PSPNet().to(DEV).eval()
    boxes = np.zeros((len(ALL_PATHS), 4), np.float32)
    n_fallback = 0
    for s in range(0, len(ALL_PATHS), 16):
        chunk = ALL_PATHS[s:s + 16]
        x = np.stack([cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (512, 512), interpolation=cv2.INTER_AREA) for p in chunk]).astype(np.float32)
        x = torch.from_numpy((2 * (x / 255.0) - 1) * 1024)[:, None].to(DEV)
        with torch.no_grad(), torch.autocast(DEV, enabled=DEV == "cuda"):
            out = seg(x)
        m = (torch.sigmoid(out[:, 4:6].float()) > 0.5).any(1).cpu().numpy()
        for i in range(len(chunk)):
            ys, xs = np.where(m[i])
            if m[i].mean() < 0.05 or len(xs) == 0:
                boxes[s + i] = (0, 0, 1, 1); n_fallback += 1; continue
            x0, x1, y0, y1 = np.percentile(xs, 1) / 512, np.percentile(xs, 99) / 512, np.percentile(ys, 1) / 512, np.percentile(ys, 99) / 512
            w, h = x1 - x0, y1 - y0; mg = 0.08
            boxes[s + i] = (max(0, x0 - mg * w), max(0, y0 - mg * h), min(1, x1 + mg * w), min(1, y1 + mg * h))
    np.savez_compressed(box_file, key=key, boxes=boxes)
    del seg; torch.cuda.empty_cache()
    print(f"segmented {len(ALL_PATHS)} images, {n_fallback} fell back to the full image")
print("average box covers", round(float(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).mean()), 3), "of the image area")

LUNG_IMG = 384
def load_lung(paths, bxs, img=LUNG_IMG):
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)); arr = np.zeros((len(paths), img, img), np.uint8)
    for i, (p, b) in enumerate(zip(paths, bxs)):
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE); H, W = g.shape
        g = g[int(b[1] * H):max(int(b[3] * H), int(b[1] * H) + 8), int(b[0] * W):max(int(b[2] * W), int(b[0] * W) + 8)]
        arr[i] = cl.apply(cv2.resize(g, (img, img), interpolation=cv2.INTER_AREA))
    return arr
X_all = load_lung(ALL_PATHS, boxes)
X_lung_pool, X_lung_test = X_all[:len(pool_df)], X_all[len(pool_df):]; del X_all
print("lung-crop arrays:", X_lung_pool.shape, X_lung_test.shape)

rs = np.random.RandomState(0); pick = rs.choice(len(pool_df), 8, replace=False)
fig, ax = plt.subplots(2, 8, figsize=(20, 5.5))
for j, i in enumerate(pick):
    g = cv2.imread(pool_df.path[i], cv2.IMREAD_GRAYSCALE); H, W = g.shape; b = boxes[i]
    ax[0, j].imshow(g, cmap="gray"); ax[0, j].add_patch(plt.Rectangle((b[0] * W, b[1] * H), (b[2] - b[0]) * W, (b[3] - b[1]) * H, fill=False, color="lime", lw=2))
    ax[0, j].set_title(pool_df.label[i], fontsize=9); ax[1, j].imshow(X_lung_pool[i], cmap="gray")
    ax[0, j].axis("off"); ax[1, j].axis("off")
plt.tight_layout(); plt.savefig(OUT / "lung_crops_preview.png", dpi=110); plt.show()

from sklearn.model_selection import StratifiedKFold
N_FOLDS = 5
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(pool_df, y_pool))
fold_id = np.zeros(len(pool_df), int)
for k, (_, va) in enumerate(folds): fold_id[va] = k

TRAIN_BUDGET_MIN = 70
T0 = time.time(); fold_times = []
CV = {}
def run_cv(cfg, Xp, Xte, train_missing):
    n = cfg["name"]
    r = dict(oof_tta=np.zeros((len(Xp), 4)), test_tta=np.zeros((len(Xte), 4)), done=[])
    for k, (tr, va) in enumerate(folds):
        f = OUT / f"cv_{n}_fold{k}.npz"
        if not f.exists():
            if not train_missing: print(f"[{n}] fold {k}: file missing on Drive, skipped"); continue
            avg = np.mean(fold_times) if fold_times else 0
            if (time.time() - T0) / 60 + avg / 60 > TRAIN_BUDGET_MIN:
                print(f"[{n}] fold {k}: time budget reached, not started"); continue
            t = time.time()
            model, _, _ = train_model(dict(cfg, name=f"{n}_f{k}"), Xp[tr], y_pool[tr], Xp[va], y_pool[va], select=False)
            vp, vt = predict_both(model, Xp[va]); tp, tt = predict_both(model, Xte)
            np.savez_compressed(f, vp=vp, vt=vt, tp=tp, tt=tt)
            del model; torch.cuda.empty_cache(); fold_times.append(time.time() - t)
            print(f"[{n}] fold {k} took {fold_times[-1]/60:.1f} min")
        d = np.load(f)
        r["oof_tta"][va] = d["vt"]; r["test_tta"] += d["tt"]; r["done"].append(k)
        print(f"[{n}] fold {k} macro-F1 (tta): {f1_score(y_pool[va], d['vt'].argmax(1), average='macro'):.4f}")
    if r["done"]:
        r["test_tta"] /= len(r["done"]); CV[n] = r
    return r

OLD = ["cvA_cnx_wce_224", "cvB_cnx_focal_288", "cvC_xrv_wce_224", "cvD_lung_cnx_wce_384"]
for n in OLD:
    r = run_cv(dict(name=n), X_lung_pool, X_lung_test, train_missing=False)
    if len(r["done"]) < N_FOLDS: print(f"WARNING: {n} has only folds {r['done']}"); CV.pop(n, None)

def shrink(a, s): return np.stack([cv2.resize(x, (s, s), interpolation=cv2.INTER_AREA) for x in a])
E_CFG = dict(name="cvE_lung_cnx_focal_256", model="convnext_tiny", loss="focal", clahe=True, lr=1e-4, img=256, epochs=16, batch=32)
X_e_pool, X_e_test = shrink(X_lung_pool, 256), shrink(X_lung_test, 256)
T0 = time.time()
rE = run_cv(E_CFG, X_e_pool, X_e_test, train_missing=True)
if len(rE["done"]) < 3: print("E has fewer than 3 finished folds -> left out"); CV.pop(E_CFG["name"], None)
print("models in the ensemble:", list(CV))

import itertools
def f1m(y, p): return f1_score(y, p.argmax(1), average="macro")
common = sorted(set.intersection(*[set(v["done"]) for v in CV.values()]))
mask = np.isin(fold_id, common); print("evaluated on folds", common, f"({mask.sum()} images)")
names_cv = list(CV); rows, cand = [], {}
for r_ in range(1, len(names_cv) + 1):
    for combo in itertools.combinations(names_cv, r_):
        lab = "+".join(c.split("_")[0] for c in combo)
        cand[lab] = (np.mean([CV[c]["oof_tta"] for c in combo], 0), np.mean([CV[c]["test_tta"] for c in combo], 0))
for label, (p, _) in cand.items():
    rows.append(dict(candidate=label, OOF_macroF1=f1m(y_pool[mask], p[mask]),
                     **{f"F1_{c}": f for c, f in zip(CLASSES, f1_score(y_pool[mask], p[mask].argmax(1), average=None))}))
tbl = pd.DataFrame(rows).sort_values("OOF_macroF1", ascending=False).reset_index(drop=True)
tbl.to_csv(OUT / "cv_results_final.csv", index=False); display(tbl.round(4))

all_label = "+".join(c.split("_")[0] for c in names_cv)
oof, test_p = cand[all_label]
print("FINAL = average of", all_label, "| OOF macro-F1", round(f1m(y_pool[mask], oof[mask]), 4))
print(classification_report(y_pool[mask], oof[mask].argmax(1), target_names=CLASSES, digits=4))
cm = confusion_matrix(y_pool[mask], oof[mask].argmax(1))
plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="Blues"); plt.xticks(range(4), CLASSES, rotation=30); plt.yticks(range(4), CLASSES)
for i in range(4):
    for j in range(4): plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Out-of-fold confusion matrix"); plt.tight_layout()
plt.savefig(OUT / "cv_confusion_matrix_final.png", dpi=150); plt.show()

final_pred = test_p.argmax(1)
sub = pd.DataFrame({"new_filename": test_names})
for i, c in enumerate(CLASSES): sub[c] = (final_pred == i).astype(int)
sub.to_csv(OUT / "submission_final.csv", index=False)
print(sub.head()); print("rows:", len(sub), "| predicted class share:", dict(zip(CLASSES, (np.bincount(final_pred, minlength=4) / len(final_pred)).round(3))))
print("official test share  :", dict(zip(CLASSES, [0.266, 0.470, 0.253, 0.010])))
old = OUT / "submission_lung.csv"
if old.exists():
    o = pd.read_csv(old); print("same prediction as round-7 submission_lung.csv:", round(float((o[CLASSES].values.argmax(1) == final_pred).mean()), 4))

import shutil
PACK = Path("/content/pack_v8"); PACK.mkdir(exist_ok=True)
json.dump(dict(models=list(CV), e_cfg=E_CFG, e_folds_done=rE["done"], evaluated_folds=common, oof_macroF1=float(f1m(y_pool[mask], oof[mask]))), open(PACK / "run_config_v8.json", "w"), indent=2)
for f in ["submission_final.csv", "cv_results_final.csv", "cv_confusion_matrix_final.png"]: shutil.copy(OUT / f, PACK / f)
zip_path = shutil.make_archive("/content/lab3_results_v8", "zip", PACK)
print("zip:", zip_path, f"({os.path.getsize(zip_path)/1e6:.1f} MB)")
from google.colab import files
files.download(zip_path)

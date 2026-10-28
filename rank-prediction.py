import sys
import glob
import os
import time
from multiprocessing import Pool, Process
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import webdataset as wds
from torch.utils.data import DataLoader

from utils import SGFParseRankPrediction
from network import GoRankResNet
'''
10/27修改內容（加入目標玩家執色 color 資訊）
1. utils.py (SGFParseRankPrediction)：
   - features_from_sgf_content 與 _extract_features 新增接收 color ('B' 或 'W')
   - 偶數通道固定放「目標玩家棋子」、奇數通道固定放「對手棋子」，並新增第 501 個通道 (執黑填 1、執白填 0)，特徵形狀改為 (501, 19, 19)
2. rank-prediction.py：
   - extract_worker 與 predict 推論皆傳入對應的 color 欄位；若偵測到現有分片形狀與 (501, 19, 19) 不符則自動重抽分片
3. network.py (GoRankResNet)：
   - in_channels 由 500 改為 501 (參數量 4,160,138)
'''
# --- 加入日誌記錄功能 ---
class Logger(object):
    def __init__(self, filename="output_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8') # 每次執行會覆寫，若要附加請改 "a"

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 將標準輸出重導向到 Logger
sys.stdout = Logger("output_log.txt")
# ------------------------

print('torch', torch.__version__, '| cuda', torch.version.cuda,
      '| webdataset', wds.__version__)


# 官方資料放置位置：training/ 與 tests/ 依官方發佈的結構擺放
TRAIN_DIR = './training'
TEST_DIR = './tests'
FEAT_DIR = './rank-features'
CKPT_DIR = './rank-trained-models'

# 等級由弱到強。順序很重要：評分時的「相鄰一級」就是這個順序上的相鄰。
LEVELS = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']
RANK_TO_ID = {lv: i for i, lv in enumerate(LEVELS)}
ID_TO_RANK = {i: lv for lv, i in RANK_TO_ID.items()}

NPROC = 8              # 特徵抽取的進程數，設成 CPU 核數
PER_SHARD = 1000       # 每個 webdataset 分片的樣本數
SEED = 42
VAL_SIZE = 100_000     # 從訓練集切出來的驗證集大小

BATCH_SIZE = 128
LEARNING_RATE = 7e-4
NUM_EPOCHS = 30
NUM_CLASSES = len(LEVELS)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs(CKPT_DIR, exist_ok=True)
print('device', DEVICE)
RANK_TO_ID


df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f'{TRAIN_DIR}/train_*.csv'))],
               ignore_index=True)
df_train, df_val = train_test_split(
    df,
    test_size=VAL_SIZE,
    random_state=SEED,
    stratify=df['rank']
)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

print('整體比例')
print(df['rank'].value_counts(normalize=True).sort_index())

print('訓練集比例')
print(df_train['rank'].value_counts(normalize=True).sort_index())

print('驗證集比例')
print(df_val['rank'].value_counts(normalize=True).sort_index())
df.head(3)


_parser = SGFParseRankPrediction()
_demo = df.iloc[0]['sgf_content']
_feat, _err = _parser.features_from_sgf_content(_demo, df.iloc[0]['color'])
print('一局棋的特徵形狀:', _feat.shape, _feat.dtype)


def extract_worker(df_part, pattern, wid):
    """抽一份切片的特徵並寫成分片。每個進程獨立持有 parser 與 ShardWriter。"""
    parser = SGFParseRankPrediction()
    written = bad = 0
    t0 = time.time()
    with wds.ShardWriter(pattern, maxcount=PER_SHARD, verbose=0) as sink:
        for _, row in df_part.iterrows():
            feat, err = parser.features_from_sgf_content(row['sgf_content'], row['color'])
            if err is not None:
                bad += 1
                continue
            sink.write({
                '__key__': '%s_%s' % (row['rank'], row['game_id']),
                'features.npy': feat,
                'label.txt': str(row['rank']),
                'rank_idx.txt': str(RANK_TO_ID[row['rank']]),
                'game_name.txt': str(row['game_id']),
            })
            written += 1
            if written % 20000 == 0:
                print('  [w%d] %d/%d  %.0f 分'
                      % (wid, written, len(df_part), (time.time() - t0) / 60),
                      flush=True)
    print('  [w%d] 寫入 %d 筆（解析失敗 %d）%.1f 分'
          % (wid, written, bad, (time.time() - t0) / 60), flush=True)


def extract_split(df_part, tag):
    """跑完一個 split，並確認產出真的有內容。"""
    outdir = f'{FEAT_DIR}/{tag}'
    os.makedirs(outdir, exist_ok=True)
    for old_f in glob.glob(f'{outdir}/*.tar.gz'):
        os.remove(old_f)

    procs = []
    for i in range(NPROC):
        pat = '%s/%s-w%d-%%06d.tar.gz' % (outdir, tag, i)
        p = Process(target=extract_worker, args=(df_part.iloc[i::NPROC], pat, i))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise SystemExit('%s: worker 非正常結束（exitcode %s）' % (tag, p.exitcode))

    shards = glob.glob(f'{outdir}/*.tar.gz')
    total = sum(os.path.getsize(s) for s in shards)
    avg = total / len(shards) if shards else 0
    print('%s: %d 個分片, %.2f GB, 平均 %.1f KB/分片'
          % (tag, len(shards), total / 1e9, avg / 1024), flush=True)
    # 平均分片大小是最簡單的健全性檢查：寫入迴圈若出錯，計數器仍會累加、log 看起來
    # 正常，但 tar 是空的（gzip 後僅 46 bytes）。只數分片數量抓不到這種情況。
    if avg < 10 * 1024:
        raise SystemExit('%s: 分片平均僅 %.0f bytes，判定為空分片' % (tag, avg))


def shards_match_expected_shape():
    train_s = sorted(glob.glob(f'{FEAT_DIR}/train/*.tar.gz'))
    val_s = sorted(glob.glob(f'{FEAT_DIR}/val/*.tar.gz'))
    if not train_s or not val_s:
        return False
    try:
        sample = next(iter(wds.WebDataset([val_s[0]], shardshuffle=False).decode()))
        return sample['features.npy'].shape == _feat.shape
    except Exception:
        return False


# 一次性作業。分片產出且特徵形狀相符時可直接跳過；若特徵通道數改變則自動重抽分片。
if shards_match_expected_shape():
    print('偵測到分片已存在且形狀相符 %s，跳過特徵抽取！' % (str(_feat.shape),))
else:
    print('開始抽取特徵分片（目標形狀 %s）...' % (str(_feat.shape),), flush=True)
    t0 = time.time()
    extract_split(df_val, 'val')
    extract_split(df_train, 'train')
    print('完成，總耗時 %.1f 分' % ((time.time() - t0) / 60))


def decode_sample(sample):
    return (torch.from_numpy(sample['features.npy']).float(),
            torch.tensor(int(sample['rank_idx.txt']), dtype=torch.long),
            sample['game_name.txt'])


def make_loader(shards, workers, shuffle):
    ds = wds.WebDataset(shards, shardshuffle=1000 if shuffle else 0, detshuffle=shuffle)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.decode().map(decode_sample).batched(BATCH_SIZE, partial=True)
    return DataLoader(ds, batch_size=None, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0)


def pick_workers(val_shards, candidate=4):
    """數一遍驗證集：筆數對得上才用多 worker，否則退回單執行緒。"""
    total = sum(len(y) for _x, y, _n in make_loader(val_shards, candidate, False))
    if total == VAL_SIZE:
        print('num_workers=%d 驗證通過（%d 筆）' % (candidate, total))
        return candidate
    print('num_workers=%d 讀出 %d 筆（應為 %d），退回 0' % (candidate, total, VAL_SIZE))
    return 0


train_shards = sorted(glob.glob(f'{FEAT_DIR}/train/*.tar.gz'))
val_shards = sorted(glob.glob(f'{FEAT_DIR}/val/*.tar.gz'))
print('分片：train %d / val %d' % (len(train_shards), len(val_shards)))

workers = pick_workers(val_shards)
train_loader = make_loader(train_shards, workers, True)
val_loader = make_loader(val_shards, workers, False)

_x, _y, _n = next(iter(val_loader))
print('一個 batch:', _x.shape, _y[:8].tolist())


model = GoRankResNet(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

print('參數量 {:,}'.format(sum(p.numel() for p in model.parameters())))
print('batch %d | lr %s | epochs %d' % (BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS))

def validate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total, batches = 0.0, 0, 0, 0
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            loss_sum += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batches += 1
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())
    model.train()

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
    )
    print('各段位準確率：')
    for i, rank in enumerate(LEVELS):
        rank_total = cm[i].sum()
        rank_acc = cm[i, i] / rank_total if rank_total else 0.0
        print('  %s: %.2f%% (%d/%d)' %
              (rank, 100 * rank_acc, cm[i, i], rank_total))
    print('混淆矩陣（列=實際段位，欄=預測段位；順序=%s）:' % LEVELS)
    print(cm)

    return (loss_sum / batches if batches else 0.0,
            100 * correct / total if total else 0.0)


# 訓練期間不能中斷 kernel。
history = []
model.train()
best_val_acc = 0.0
train_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    total_loss, correct, total, steps = 0.0, 0, 0, 0

    for inputs, labels, _ in train_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # 圍棋棋盤 8 重對稱隨機增強（2種翻轉 × 4種旋轉 = 8種等價方向）
        if torch.rand(1).item() < 0.5:
            inputs = torch.flip(inputs, dims=[-1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            inputs = torch.rot90(inputs, k, dims=[-2, -1])

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        steps += 1
        if steps % 500 == 0:
            print('  ep%d step %d  loss %.4f  acc %.2f%%  %.1f 分'
                  % (epoch + 1, steps, total_loss / steps,
                     100 * correct / total, (time.time() - t0) / 60), flush=True)

    train_acc = 100 * correct / total if total else 0.0
    val_loss, val_acc = validate(model, val_loader, criterion)
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    epoch_minutes = (time.time() - t0) / 60

    history.append({'epoch': epoch + 1, 'train_acc': train_acc, 'val_acc': val_acc})
    print('Epoch %d/%d  lr %.6f  |  train acc %.2f%%  |  val loss %.4f acc %.2f%%  |  %.1f 分'
          % (epoch + 1, NUM_EPOCHS, current_lr, train_acc, val_loss, val_acc,
             epoch_minutes), flush=True)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f'{CKPT_DIR}/best_model.pth')
        print('  新的最佳 val acc %.2f%% —— 已存 best_model.pth' % best_val_acc)
    torch.save(model.state_dict(), f'{CKPT_DIR}/epoch_%d.pth' % (epoch + 1))

total_train_minutes = (time.time() - train_start_time) / 60
pd.DataFrame(history).to_csv(f'{CKPT_DIR}/history.csv', index=False)
print('訓練完成，總耗時 %.1f 分（%.2f 小時），最佳 val acc %.2f%%'
      % (total_train_minutes, total_train_minutes / 60, best_val_acc))


import matplotlib.pyplot as plt

hist = pd.read_csv(f'{CKPT_DIR}/history.csv')
best = hist.loc[hist['val_acc'].idxmax()]

plt.figure(figsize=(9, 4))
plt.plot(hist['epoch'], hist['train_acc'], marker='o', ms=3, label='train acc')
plt.plot(hist['epoch'], hist['val_acc'], marker='o', ms=3, label='val acc')
plt.axvline(best['epoch'], ls='--', c='gray', lw=1)
plt.annotate('best %.2f%%' % best['val_acc'],
             xy=(best['epoch'], best['val_acc']),
             xytext=(best['epoch'] + 1.5, best['val_acc'] - 4),
             arrowprops=dict(arrowstyle='->', color='gray'))
plt.xlabel('epoch'); plt.ylabel('accuracy (%)')
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); 
plt.savefig('training_curve.png') # 將圖片儲存下來
print("訓練曲線已儲存為 training_curve.png")
# plt.show() # 如果要在無 UI 環境執行，建議註解掉 show()

GPU_BATCH = 64
_infer_parser = SGFParseRankPrediction()


def sgf_to_features(args):
    """給 multiprocessing.Pool 用的頂層函式，接收 (sgf_content, color)。"""
    sgf_content, color = args
    feat, err = _infer_parser.features_from_sgf_content(sgf_content, color)
    return None if err is not None else feat


def predict(weights_path, out_path):
    net = GoRankResNet(num_classes=NUM_CLASSES)
    net.load_state_dict(torch.load(weights_path, map_location='cpu'))
    net.eval().to(DEVICE)

    rows = []
    with Pool(NPROC) as pool:
        # 兩份題目都要跑，結果串接成同一份 submission
        for split in ('public', 'private'):
            test = pd.read_csv(f'{TEST_DIR}/rank_prediction_test_{split}.csv')
            sgf_cols = [c for c in test.columns if c.startswith('sgf_')]
            t0 = time.time()

            for idx, r in test.iterrows():
                # num_games 之後的 sgf_ 欄位是空字串，要濾掉，並配對對應的 color_i
                sgf_color_pairs = [
                    (r[c], r[c.replace('sgf_', 'color_')])
                    for c in sgf_cols
                    if pd.notna(r[c]) and str(r[c]).strip()
                ]
                feats = [f for f in pool.map(sgf_to_features, sgf_color_pairs) if f is not None]
                if not feats:
                    rows.append([r['question_id'], LEVELS[0]])
                    continue

                x = torch.from_numpy(np.stack(feats)).float()
                probs = []
                with torch.no_grad():
                    for i in range(0, len(x), GPU_BATCH):
                        out = net(x[i:i + GPU_BATCH].to(DEVICE))
                        probs.append(out.softmax(1).cpu().numpy())

                avg = np.concatenate(probs).mean(0)      # 同題多局的機率平均
                rows.append([r['question_id'], LEVELS[int(avg.argmax())]])
                if (idx + 1) % 100 == 0:
                    print('  %s %d/%d  %.0f 秒'
                          % (split, idx + 1, len(test), time.time() - t0), flush=True)
            print('  %s 完成 %d 題' % (split, len(test)), flush=True)

    sub = pd.DataFrame(rows, columns=['question_id', 'pred_rank'])
    sub.to_csv(out_path, index=False)
    print('寫出 %s（%d 列）' % (out_path, len(sub)))
    print('預測分布:', sub['pred_rank'].value_counts().to_dict())
    return sub


test_files = [
    f'{TEST_DIR}/rank_prediction_test_public.csv',
    f'{TEST_DIR}/rank_prediction_test_private.csv',
]
if all(os.path.isfile(path) for path in test_files):
    submission = predict(f'{CKPT_DIR}/best_model.pth', './submission_rank.csv')
    print("預測結果前 5 筆：")
    print(submission.head())
else:
    print('測試集檔案尚未提供，略過預測與 submission 輸出。')

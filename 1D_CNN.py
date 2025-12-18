# ============================================================
# GCC-CNN (HƯỚNG A) — Regression cos(theta_fold) for DOA 0..90
# Input : GCC-PHAT vector (2*MAX_LAG+1)
# Target: cos(theta_fold) in [0,1]  (theta_fold = min(theta,180-theta))
# Output: theta_hat_fold = arccos(pred_cos) in degrees (0..90)
# ============================================================

import math
import random
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIG (FAST preset)
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Folder chứa 20 file: "audio 1.wav" ... (có space vẫn ok)
AUDIO_DIR_CANDIDATES = [
    Path(__file__).resolve().parent,                # cùng folder script
    Path(__file__).resolve().parent / "audio",      # ./audio
    Path.cwd(),                                     # working dir
    Path.cwd() / "audio",
]
AUDIO_DIR = None
for p in AUDIO_DIR_CANDIDATES:
    if p.exists() and len(list(p.glob("*.wav"))) > 0:
        AUDIO_DIR = p
        break
if AUDIO_DIR is None:
    raise FileNotFoundError("Không tìm thấy *.wav. Hãy để wav cùng folder script hoặc trong ./audio")

# Audio / Frame
FRAME_LEN = 4096          # tăng giúp GCC ổn định hơn
HOP_LEN   = 2048

# Array params (2 mics)
D = 0.20                  # meters
C = 343.0                 # m/s

# GCC
MAX_LAG = 64              # -> feature length L = 129
EPS = 1e-9

# Dataset sampling (FAST)
ANGLES_DEG = list(range(0, 181, 10))   # 0..180 step 10 (fast)
SNRS_DB = [10, 20]                      # bỏ SNR=0 để dễ học
MAX_FRAMES_PER_COMBO = 30               # cap frames per (wav, angle, snr)
MAX_WAV_FILES = 20                      # set 5 để debug nhanh hơn

# Training
BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-5
TEST_SIZE = 0.2
VAL_SIZE_IN_TRAIN = 0.2
PATIENCE = 7
PRINT_EVERY = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Utils
# ============================================================
def to_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0)
    if x.dtype == np.int32:
        return (x.astype(np.float32) / 2147483648.0)
    return x.astype(np.float32)

def ensure_2ch(x: np.ndarray) -> np.ndarray:
    # return shape (N,2)
    if x.ndim == 1:
        # mono -> duplicate
        return np.stack([x, x], axis=1)
    if x.ndim == 2 and x.shape[1] >= 2:
        return x[:, :2]
    if x.ndim == 2 and x.shape[0] >= 2 and x.shape[1] == 1:
        return np.concatenate([x, x], axis=1)
    raise ValueError(f"Audio shape không hợp lệ: {x.shape}")

def split_frames(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    # x shape: (N,) -> frames: (n_frames, frame_len)
    n = len(x)
    if n < frame_len:
        return np.zeros((0, frame_len), dtype=np.float32)
    n_frames = 1 + (n - frame_len) // hop_len
    frames = np.zeros((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop_len
        frames[i] = x[s:s+frame_len]
    return frames

def energy_ok(frame: np.ndarray, thr: float = 1e-4) -> bool:
    return float(np.mean(frame * frame)) > thr

def add_awgn(x: np.ndarray, snr_db: float) -> np.ndarray:
    if snr_db is None:
        return x
    p = np.mean(x * x) + 1e-12
    snr = 10.0 ** (snr_db / 10.0)
    noise_p = p / snr
    noise = np.random.randn(*x.shape).astype(np.float32) * np.sqrt(noise_p)
    return x + noise

def fractional_delay_fft(x: np.ndarray, d_samp: float) -> np.ndarray:
    """
    Fractional delay using frequency-domain phase shift.
    d_samp: positive => delay (shift right), negative => advance.
    """
    n = len(x)
    X = np.fft.rfft(x)
    k = np.arange(len(X), dtype=np.float32)
    # phase = -j*2*pi*k/N * d
    phase = np.exp(-1j * 2.0 * np.pi * k * d_samp / n)
    Y = X * phase
    y = np.fft.irfft(Y, n=n).astype(np.float32)
    return y

def gcc_phat(x: np.ndarray, y: np.ndarray, max_lag: int, eps: float = 1e-9) -> np.ndarray:
    """
    GCC-PHAT between x and y. Output length = 2*max_lag+1.
    """
    n = len(x) + len(y)
    nfft = 1
    while nfft < n:
        nfft *= 2

    X = np.fft.rfft(x, nfft)
    Y = np.fft.rfft(y, nfft)
    R = X * np.conj(Y)
    R /= (np.abs(R) + eps)
    cc = np.fft.irfft(R, nfft)

    # shift
    cc = np.concatenate((cc[-max_lag:], cc[:max_lag+1]))
    cc = cc.astype(np.float32)

    # normalize
    cc = cc / (np.max(np.abs(cc)) + eps)
    return cc

def theta_fold_deg(theta_deg: float) -> float:
    # fold 0..180 -> 0..90
    return float(min(theta_deg, 180.0 - theta_deg))

def build_delay_samples(theta_deg: float, fs: int, d: float = D, c: float = C) -> float:
    # Using folded theta (0..90), delay is non-negative magnitude only (consistent with ambiguity)
    th = math.radians(theta_fold_deg(theta_deg))
    tau = (d * math.cos(th)) / c  # seconds, >=0
    return float(tau * fs)


# ============================================================
# Dataset builder (HƯỚNG A)
# X: GCC vector (L=2*MAX_LAG+1)
# y: cos(theta_fold) in [0,1]
# also store theta_fold for evaluation in degrees
# ============================================================
def build_dataset():
    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if len(wavs) == 0:
        raise FileNotFoundError(f"Không thấy *.wav trong {AUDIO_DIR}")

    if MAX_WAV_FILES is not None:
        wavs = wavs[:MAX_WAV_FILES]

    print("Device:", DEVICE)
    print("Audio folder:", AUDIO_DIR.resolve())
    print("Found", len(wavs), "wav files")

    X_list = []
    ycos_list = []
    ydeg_list = []

    total_silent = 0

    for wi, w in enumerate(wavs, start=1):
        fs, x = wavfile.read(str(w))
        x = ensure_2ch(to_float32(x))   # (N,2)
        # dùng kênh 0 làm "source frame" (giống bạn đang làm)
        s = x[:, 0].astype(np.float32)

        # cắt 1 đoạn giữa cho ổn định
        N = len(s)
        mid = N // 2
        seg_len = min(N, fs * 10)  # tối đa 10s từ giữa
        a = max(0, mid - seg_len // 2)
        b = min(N, a + seg_len)
        s = s[a:b]

        frames = split_frames(s, FRAME_LEN, HOP_LEN)
        if len(frames) == 0:
            continue

        for theta in ANGLES_DEG:
            d_samp = build_delay_samples(theta, fs)  # >= 0
            th_fold = theta_fold_deg(theta)
            y_cos = math.cos(math.radians(th_fold))  # in [0,1]

            for snr in SNRS_DB:
                idxs = np.arange(len(frames))
                np.random.shuffle(idxs)
                idxs = idxs[:min(MAX_FRAMES_PER_COMBO, len(idxs))]

                for idx in idxs:
                    fr = frames[idx].copy()

                    if not energy_ok(fr):
                        total_silent += 1
                        continue

                    # Two-mic synthetic pair:
                    x0 = add_awgn(fr, snr)
                    x1 = fractional_delay_fft(fr, d_samp)
                    x1 = add_awgn(x1, snr)

                    cc = gcc_phat(x1, x0, MAX_LAG, EPS)  # (L,)
                    X_list.append(cc)
                    ycos_list.append(y_cos)
                    ydeg_list.append(th_fold)

        if wi % 2 == 0:
            print(f"[{wi}/{len(wavs)}] {w.name} -> samples so far: {len(X_list)}")

    X = np.stack(X_list, axis=0).astype(np.float32)                 # (N,L)
    y_cos = np.array(ycos_list, dtype=np.float32).reshape(-1, 1)    # (N,1)
    y_deg = np.array(ydeg_list, dtype=np.float32).reshape(-1, 1)    # (N,1)

    print("Silent skipped:", total_silent)
    print("Dataset:", "X", X.shape, "y_cos", y_cos.shape, "y_deg", y_deg.shape)
    return X, y_cos, y_deg


# ============================================================
# Model: 1D CNN (3x1 64 -> 3x1 24 -> 3x1 8 -> GAP -> 10 -> 1)
# Output with sigmoid to keep [0,1] (cos folded)
# ============================================================
class GCC_CNN_A(nn.Module):
    def __init__(self, L: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(24, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(8, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        # x: (B,1,L)
        z = self.net(x)
        z = self.gap(z).squeeze(-1)      # (B,8)
        z = torch.relu(self.fc1(z))      # (B,10)
        z = torch.sigmoid(self.fc2(z))   # (B,1) in [0,1]
        return z


@torch.no_grad()
def eval_loader(model, loader):
    model.eval()
    preds = []
    trues = []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        yp = model(xb)
        preds.append(yp.cpu().numpy())
        trues.append(yb.cpu().numpy())
    y_pred = np.vstack(preds).reshape(-1)
    y_true = np.vstack(trues).reshape(-1)

    # convert cos -> theta_fold
    y_pred = np.clip(y_pred, 0.0, 1.0)
    th_pred = np.degrees(np.arccos(y_pred))
    th_true = np.degrees(np.arccos(np.clip(y_true, 0.0, 1.0)))  # consistent

    rmse = float(np.sqrt(np.mean((th_true - th_pred) ** 2)))
    mae = float(np.mean(np.abs(th_true - th_pred)))
    return rmse, mae, th_true, th_pred


def train():
    X, y_cos, y_deg = build_dataset()

    # Train/Val/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cos, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE_IN_TRAIN, random_state=SEED, shuffle=True
    )

    print(f"Split: train={len(X_tr)}, val={len(X_val)}, test={len(X_test)}")
    L = X.shape[1]
    print(f"GCC length L={L}  (MAX_LAG={MAX_LAG})")
    print("CNN: 3x1(64)->3x1(24)->3x1(8)->GAP->10->1 (sigmoid)")

    def make_loader(Xa, ya, bs, shuffle):
        xb = torch.from_numpy(Xa).float().unsqueeze(1)  # (N,1,L)
        yb = torch.from_numpy(ya).float()               # (N,1)
        ds = TensorDataset(xb, yb)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

    train_loader = make_loader(X_tr, y_tr, BATCH_SIZE, True)
    val_loader   = make_loader(X_val, y_val, BATCH_SIZE, False)
    test_loader  = make_loader(X_test, y_test, BATCH_SIZE, False)

    model = GCC_CNN_A(L).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            yp = model(xb)
            loss = loss_fn(yp, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        val_rmse, val_mae, _, _ = eval_loader(model, val_loader)

        if ep % PRINT_EVERY == 0 or ep == 1:
            print(f"Epoch {ep:03d} | TrainLoss={np.mean(losses):.6f} | ValRMSE={val_rmse:.3f}° | ValMAE={val_mae:.3f}°")

        # early stop on RMSE
        if val_rmse + 1e-6 < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stop at epoch {ep} (best ValRMSE={best_val:.3f}°)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_rmse, test_mae, th_true, th_pred = eval_loader(model, test_loader)
    print("\n================ FINAL TEST (A: cos(theta_fold)) ================")
    print(f"Test RMSE: {test_rmse:.3f}°")
    print(f"Test MAE : {test_mae:.3f}°")

    print("\n===== 10 samples (test) =====")
    idx = np.random.choice(len(th_true), size=min(10, len(th_true)), replace=False)
    for i, k in enumerate(idx):
        print(f"{i:02d} | True={th_true[k]:6.2f}° | Pred={th_pred[k]:6.2f}° | Err={abs(th_true[k]-th_pred[k]):6.2f}°")


def main():
    train()


if __name__ == "__main__":
    main()

# ============================================================
# 2D CNN DOA (0..180) - LIGHT + FIX "stuck at 90deg"
# Key fixes:
#   - Predict theta_norm = theta/180 (0..1), NOT cos(theta)
#   - Sigmoid output + SmoothL1 loss
#   - Per-sample normalization (zero-mean / std)
#   - BatchNorm for stability
#   - ReduceLROnPlateau scheduler
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
# CONFIG (still LIGHT)
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Geometry
M = 3
D = 0.20
C = 343.0

# Frame (nhẹ)
FRAME_LEN = 512
HOP_LEN = 512

# Dataset
ANGLES_DEG = list(range(0, 181, 15))     # debug: 0..180 step 15
SNRS_DB = [10, 20]                       # 2 mức SNR (vẫn nhẹ)
MAX_FRAMES_PER_COMBO = 12                # tăng nhẹ để học được
MAX_WAV_FILES = 20                       # set 5 để test siêu nhanh

# Train
BATCH_SIZE = 128
EPOCHS = 30
LR = 2e-3
WEIGHT_DECAY = 1e-5
TEST_SIZE = 0.2
VAL_SIZE_IN_TRAIN = 0.2
PATIENCE = 6
PRINT_EVERY = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Auto find wav folder
AUDIO_DIR_CANDIDATES = [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent / "audio",
    Path.cwd(),
    Path.cwd() / "audio",
]
AUDIO_DIR = None
for p in AUDIO_DIR_CANDIDATES:
    if p.exists() and len(list(p.glob("*.wav"))) > 0:
        AUDIO_DIR = p
        break
if AUDIO_DIR is None:
    raise FileNotFoundError("Không tìm thấy *.wav. Hãy để wav cùng folder script hoặc trong ./audio")

# -----------------------------
# Utils
# -----------------------------
def to_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    if x.dtype == np.int32:
        return x.astype(np.float32) / 2147483648.0
    return x.astype(np.float32)

def ensure_1d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x[:, 0]
    raise ValueError(f"Audio shape không hợp lệ: {x.shape}")

def split_frames(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    n = len(x)
    if n < frame_len:
        return np.zeros((0, frame_len), dtype=np.float32)
    n_frames = 1 + (n - frame_len) // hop_len
    frames = np.zeros((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop_len
        frames[i] = x[s:s + frame_len]
    return frames

def energy_ok(frame: np.ndarray, thr: float = 1e-4) -> bool:
    return float(np.mean(frame * frame)) > thr

def add_awgn(x: np.ndarray, snr_db: float) -> np.ndarray:
    p = np.mean(x * x) + 1e-12
    snr = 10.0 ** (snr_db / 10.0)
    noise_p = p / snr
    noise = np.random.randn(*x.shape).astype(np.float32) * np.sqrt(noise_p)
    return x + noise

def fractional_delay_fft(x: np.ndarray, d_samp: float) -> np.ndarray:
    n = len(x)
    X = np.fft.rfft(x)
    k = np.arange(len(X), dtype=np.float32)
    phase = np.exp(-1j * 2.0 * np.pi * k * d_samp / n)
    y = np.fft.irfft(X * phase, n=n).astype(np.float32)
    return y

def simulate_multimic_frame(s: np.ndarray, theta_deg: float, fs: int, snr_db: float) -> np.ndarray:
    th = math.radians(theta_deg)
    tau_adj = (D * math.cos(th)) / C
    d_adj_samp = tau_adj * fs

    Xm = np.zeros((M, len(s)), dtype=np.float32)
    # augmentation: random gain (giúp model không dính amplitude)
    gain = 10 ** np.random.uniform(-0.3, 0.3)
    base = (s * gain).astype(np.float32)

    for m in range(M):
        xm = fractional_delay_fft(base, m * d_adj_samp)
        xm = add_awgn(xm, snr_db)
        Xm[m] = xm
    return Xm  # (M,L)

def normalize_sample(Xm: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # Xm: (M,L) normalize per channel
    Xm = Xm - Xm.mean(axis=1, keepdims=True)
    std = Xm.std(axis=1, keepdims=True) + eps
    Xm = Xm / std
    return Xm.astype(np.float32)

# -----------------------------
# Build dataset
# -----------------------------
def build_dataset():
    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if len(wavs) == 0:
        raise FileNotFoundError(f"Không thấy *.wav trong {AUDIO_DIR}")

    wavs = wavs[:MAX_WAV_FILES]

    print("Device:", DEVICE)
    print("Audio folder:", AUDIO_DIR.resolve())
    print("Found", len(wavs), "wav files")

    X_list, y_list = [], []
    silent = 0

    for wi, w in enumerate(wavs, start=1):
        fs, x = wavfile.read(str(w))
        x = to_float32(x)
        s = ensure_1d(x)

        # lấy đoạn giữa ~4s để nhẹ
        N = len(s)
        seg_len = min(N, int(fs * 4))
        mid = N // 2
        a = max(0, mid - seg_len // 2)
        b = min(N, a + seg_len)
        s = s[a:b].astype(np.float32)

        frames = split_frames(s, FRAME_LEN, HOP_LEN)
        if len(frames) == 0:
            continue

        for theta in ANGLES_DEG:
            y_norm = float(theta) / 180.0  # target in [0,1]
            for snr in SNRS_DB:
                idxs = np.arange(len(frames))
                np.random.shuffle(idxs)
                idxs = idxs[:min(MAX_FRAMES_PER_COMBO, len(idxs))]

                for idx in idxs:
                    fr = frames[idx].copy()
                    if not energy_ok(fr):
                        silent += 1
                        continue

                    Xm = simulate_multimic_frame(fr, theta, fs, snr)  # (M,L)
                    Xm = normalize_sample(Xm)                         # critical FIX
                    X_list.append(Xm)
                    y_list.append(y_norm)

        if wi % 2 == 0:
            print(f"[{wi}/{len(wavs)}] {w.name} -> samples: {len(X_list)}")

    X = np.stack(X_list, axis=0).astype(np.float32)            # (N,M,L)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)      # (N,1)

    print("Silent skipped:", silent)
    print("Dataset:", "X", X.shape, "y", y.shape)
    return X, y

# -----------------------------
# 2D CNN model (still light but stable)
# -----------------------------
class LightCNN2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((M, 8))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * M * 8, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()  # theta_norm in [0,1]
        )

    def forward(self, x):
        z = self.features(x)
        z = self.gap(z)
        return self.head(z)

@torch.no_grad()
def eval_loader(model, loader):
    model.eval()
    yp_all, yt_all = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        yp = model(xb)
        yp_all.append(yp.cpu().numpy())
        yt_all.append(yb.cpu().numpy())

    y_pred = np.vstack(yp_all).reshape(-1)
    y_true = np.vstack(yt_all).reshape(-1)

    # degrees
    th_pred = np.clip(y_pred, 0.0, 1.0) * 180.0
    th_true = np.clip(y_true, 0.0, 1.0) * 180.0

    rmse = float(np.sqrt(np.mean((th_true - th_pred) ** 2)))
    mae  = float(np.mean(np.abs(th_true - th_pred)))
    return rmse, mae

def train():
    X, y = build_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE_IN_TRAIN, random_state=SEED, shuffle=True
    )
    print(f"Split: train={len(X_tr)}, val={len(X_val)}, test={len(X_test)}")
    print("2D CNN FIX: predict theta_norm (sigmoid) + SmoothL1 + normalize")

    def make_loader(Xa, ya, bs, shuffle):
        xb = torch.from_numpy(Xa).float().unsqueeze(1)  # (N,1,M,L)
        yb = torch.from_numpy(ya).float()               # (N,1)
        ds = TensorDataset(xb, yb)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

    train_loader = make_loader(X_tr, y_tr, BATCH_SIZE, True)
    val_loader   = make_loader(X_val, y_val, BATCH_SIZE, False)
    test_loader  = make_loader(X_test, y_test, BATCH_SIZE, False)

    model = LightCNN2D().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss(beta=0.02)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2
    )

    best = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            yp = model(xb)
            loss = loss_fn(yp, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        val_rmse, val_mae = eval_loader(model, val_loader)
        scheduler.step(val_rmse)

        if ep % PRINT_EVERY == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(f"Epoch {ep:03d} | lr={lr_now:.2e} | TrainLoss={np.mean(losses):.6f} | ValRMSE={val_rmse:.3f}° | ValMAE={val_mae:.3f}°")

        if val_rmse + 1e-6 < best:
            best = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stop at epoch {ep} (best ValRMSE={best:.3f}°)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_rmse, test_mae = eval_loader(model, test_loader)
    print("\n================ FINAL TEST (2D CNN FIX) ================")
    print(f"Test RMSE: {test_rmse:.3f}°")
    print(f"Test MAE : {test_mae:.3f}°")

def main():
    train()

if __name__ == "__main__":
    main()

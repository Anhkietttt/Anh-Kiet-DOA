# ANN_METHOD_A_RAWFRAME_SINLABEL.py
# ==========================================================
# METHOD A (RAW FRAME) - COLUMN FLATTEN (3072)
# Label = sin(theta) to avoid ULA ambiguity (theta vs 180-theta)
# Evaluate RMSE/MAE on theta_eval = min(theta, 180-theta) in degrees (0..90)
#
# Input  : frame (1024, 3) -> flatten by column => (3072,)
# Output : beta = sin(theta) in [0,1]
#
# Notes:
# - Works best for ULA/linear 3-mic because physics only provides sin(theta).
# - If you insist 0..180 unique DOA, you need non-linear array (triangle) or extra cues.
# ==========================================================

import math
import random
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -------------------------
# CONFIG (FAST PRESET)
# -------------------------
AUDIO_DIR = Path(".")          # folder chứa audio 1.wav ... audio 20.wav
FRAME_LEN = 1024
HOP_LEN = 1024                # không overlap. muốn overlap set 512

M = 3
d = 0.20
c = 343.0

ANGLES_DEG = list(range(0, 181, 5))     # 0,5,10,...,180 (nhẹ)
SNRS_DB = [0, 10, 20]                   # nhẹ
MAX_FRAMES_PER_COMBO = 100              # theo yêu cầu 100 times/angle
SEED = 42

# limit wav files for quick test (set None to use all)
MAX_WAV_FILES = None  # ví dụ test nhanh: 5

# ANN training
BATCH_SIZE = 256
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAL_SIZE_IN_TRAIN = 0.2
TEST_SIZE = 0.2

ACTIVATION = "relu"  # "relu" | "tanh" | "sigmoid"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# UTILITIES
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mono_wav(path: Path):
    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]  # take left if stereo
    # normalize to float32 [-1,1] if int
    if np.issubdtype(data.dtype, np.integer):
        maxv = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / maxv
    else:
        data = data.astype(np.float32)
    return fs, data


def fractional_delay_linear(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """
    Simple fractional delay via linear interpolation.
    y[n] = x[n - delay]
    """
    n = np.arange(len(x), dtype=np.float32)
    idx = n - delay_samples
    # np.interp expects xp increasing, fill outside with 0
    y = np.interp(idx, n, x, left=0.0, right=0.0).astype(np.float32)
    return y


def add_noise_snr(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if snr_db is None:
        return signal
    s = signal.astype(np.float32)
    p_signal = np.mean(s**2) + 1e-12
    snr_lin = 10.0 ** (snr_db / 10.0)
    p_noise = p_signal / snr_lin
    noise = rng.normal(0.0, np.sqrt(p_noise), size=s.shape).astype(np.float32)
    return s + noise


def simulate_ula_3mic_frame(frame: np.ndarray, fs: int, theta_deg: float, snr_db: float, rng: np.random.Generator):
    """
    Simulate 3-mic linear array (ULA) receiving a far-field source at angle theta.
    - mic positions: m=0,1,2 along x with spacing d
    - delay for mic m: tau_m = m * d * sin(theta) / c
    """
    theta = np.deg2rad(theta_deg)
    base = frame.astype(np.float32)

    X = np.zeros((len(base), M), dtype=np.float32)
    for m in range(M):
        tau = (m * d * math.sin(theta)) / c  # seconds
        delay_samp = tau * fs
        xm = fractional_delay_linear(base, delay_samp)
        xm = add_noise_snr(xm, snr_db, rng)
        X[:, m] = xm
    return X  # (FRAME_LEN, 3)


def column_flatten(X_frame: np.ndarray) -> np.ndarray:
    """
    Flatten by column: (1024,3) -> (3072,)
    This matches your note: from row vector to column vector conceptually.
    """
    return X_frame.reshape(-1, order="F").astype(np.float32)


# -------------------------
# DATASET BUILD (Method A)
# -------------------------
def build_dataset_rawframes():
    rng = np.random.default_rng(SEED)

    wav_paths = sorted(AUDIO_DIR.glob("audio *.wav"))
    if MAX_WAV_FILES is not None:
        wav_paths = wav_paths[:MAX_WAV_FILES]

    if len(wav_paths) == 0:
        raise FileNotFoundError(f"Không thấy .wav trong folder: {AUDIO_DIR.resolve()} (pattern: 'audio *.wav')")

    print("Device:", DEVICE)
    print("Audio folder:", AUDIO_DIR.resolve())
    print("Found", len(wav_paths), "wav files")

    X_list = []
    y_beta_list = []
    y_theta_list = []  # keep original theta for evaluation

    total_target = len(wav_paths) * len(ANGLES_DEG) * len(SNRS_DB) * MAX_FRAMES_PER_COMBO
    done = 0

    for wi, wp in enumerate(wav_paths, start=1):
        fs, sig = load_mono_wav(wp)
        if len(sig) < FRAME_LEN:
            print(f"[skip] {wp.name} too short: len={len(sig)}")
            continue

        # make frames
        num_frames = 1 + (len(sig) - FRAME_LEN) // HOP_LEN
        # precompute indices for each combo
        all_frame_ids = np.arange(num_frames)

        print(f"[{wi}/{len(wav_paths)}] {wp.name} fs={fs}, len={len(sig)}, frames={num_frames}")

        for theta_deg in ANGLES_DEG:
            for snr_db in SNRS_DB:
                # sample at most MAX_FRAMES_PER_COMBO frames
                if num_frames <= MAX_FRAMES_PER_COMBO:
                    chosen = all_frame_ids
                else:
                    chosen = rng.choice(all_frame_ids, size=MAX_FRAMES_PER_COMBO, replace=False)

                for fid in chosen:
                    start = fid * HOP_LEN
                    frame = sig[start:start + FRAME_LEN]

                    X_mics = simulate_ula_3mic_frame(frame, fs, theta_deg, snr_db, rng)
                    x = column_flatten(X_mics)  # (3072,)

                    # label beta = sin(theta) in [0,1]
                    beta = math.sin(math.radians(theta_deg))

                    X_list.append(x)
                    y_beta_list.append(beta)
                    y_theta_list.append(theta_deg)

                    done += 1
                    if done % 5000 == 0:
                        print(f"[{done}/{total_target}] sample: {wp.name} theta={theta_deg} snr={snr_db}")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y_beta = np.array(y_beta_list, dtype=np.float32)
    y_theta = np.array(y_theta_list, dtype=np.float32)

    print("Built dataset:")
    print("X:", X.shape, " y_beta:", y_beta.shape, " y_theta:", y_theta.shape)
    return X, y_beta, y_theta


# -------------------------
# MODEL
# -------------------------
def make_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError("ACTIVATION must be 'relu'|'tanh'|'sigmoid'")


class MLP(nn.Module):
    def __init__(self, in_dim=3072, act="relu"):
        super().__init__()
        A = make_activation(act)
        # 3072 -> 1024 -> 256 -> 64 -> 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            A,
            nn.Linear(1024, 256),
            A,
            nn.Linear(256, 64),
            A,
            nn.Linear(64, 1),
            nn.Sigmoid()  # output beta in [0,1]
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# METRICS (in degrees)
# -------------------------
def beta_to_theta_deg(beta: np.ndarray) -> np.ndarray:
    beta = np.clip(beta, 0.0, 1.0)
    return np.rad2deg(np.arcsin(beta))  # 0..90


def mirror_theta_eval(theta_deg: np.ndarray) -> np.ndarray:
    # map 0..180 -> 0..90
    return np.minimum(theta_deg, 180.0 - theta_deg)


def rmse_mae_deg(theta_true_eval: np.ndarray, theta_pred_deg: np.ndarray):
    err = np.abs(theta_true_eval - theta_pred_deg)
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(err)
    return float(rmse), float(mae)


# -------------------------
# TRAINING LOOP
# -------------------------
def train_ann(X, y_beta, y_theta_deg):
    # split test first
    X_tmp, X_test, yb_tmp, yb_test, yt_tmp, yt_test = train_test_split(
        X, y_beta, y_theta_deg, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )

    # split train/val inside tmp
    X_train, X_val, yb_train, yb_val, yt_train, yt_val = train_test_split(
        X_tmp, yb_tmp, yt_tmp, test_size=VAL_SIZE_IN_TRAIN, random_state=SEED, shuffle=True
    )

    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"TRAIN ANN Method A | act={ACTIVATION} | frame={FRAME_LEN} | hop={HOP_LEN} | MAX_FRAMES_PER_COMBO={MAX_FRAMES_PER_COMBO}")

    # scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # torch datasets
    train_ds = TensorDataset(torch.from_numpy(X_train_sc).float(), torch.from_numpy(yb_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val_sc).float(), torch.from_numpy(yb_val).float())
    test_ds = TensorDataset(torch.from_numpy(X_test_sc).float(), torch.from_numpy(yb_test).float())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = MLP(in_dim=X.shape[1], act=ACTIVATION).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val_rmse = 1e9
    best_state = None
    patience = 10
    bad = 0

    def eval_loader(loader, y_theta_ref):
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                out = model(xb).squeeze(1).detach().cpu().numpy()
                preds.append(out)
        beta_pred = np.concatenate(preds, axis=0)
        theta_pred = beta_to_theta_deg(beta_pred)
        theta_true_eval = mirror_theta_eval(y_theta_ref)
        rmse, mae = rmse_mae_deg(theta_true_eval, theta_pred)
        return rmse, mae

    # training
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE).view(-1, 1)

            pred = model(xb)
            loss = criterion(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        # val
        val_rmse, val_mae = eval_loader(val_loader, yt_val)

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:03d} | TrainLoss={np.mean(losses):.6f} | Val RMSE={val_rmse:.3f}° | Val MAE={val_mae:.3f}°")

        # early stop on val RMSE
        if val_rmse + 1e-6 < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {epoch} (best val RMSE={best_val_rmse:.3f}°)")
                break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # test
    test_rmse, test_mae = eval_loader(test_loader, yt_test)
    print("====================================")
    print("FINAL TEST")
    print(f"Test RMSE: {test_rmse:.3f}°")
    print(f"Test MAE : {test_mae:.3f}°")
    print("====================================")

    # print 10 samples
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_test_sc[:10]).float().to(DEVICE)
        beta_pred = model(xb).squeeze(1).cpu().numpy()
    theta_pred = beta_to_theta_deg(beta_pred)
    theta_true_eval = mirror_theta_eval(yt_test[:10])

    print("===== 10 samples (test) =====")
    for i in range(len(theta_pred)):
        print(f"{i:02d} | True(eval)={theta_true_eval[i]:6.2f}° | Pred={theta_pred[i]:6.2f}° | Err={abs(theta_true_eval[i]-theta_pred[i]):6.2f}°")

    return model


def main():
    set_seed(SEED)
    X, y_beta, y_theta = build_dataset_rawframes()
    train_ann(X, y_beta, y_theta)


if __name__ == "__main__":
    main()

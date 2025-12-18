import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile

# ===================== CONFIG =====================
AUDIO_FILE = "audio 1.wav"
THETA_DEG  = 30.0     # DOA
M          = 3        # 3 mics
D          = 0.20     # spacing (m)
C          = 343.0    # speed of sound (m/s)

START_SEC    = 1.0
DURATION_SEC = 0.03   # 30 ms
# =================================================

def to_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    if x.dtype == np.int32:
        return x.astype(np.float32) / 2147483648.0
    return x.astype(np.float32)

def fractional_delay(x: np.ndarray, d_samp: float) -> np.ndarray:
    # delay d_samp samples (can be fractional) using linear interpolation
    n = np.arange(len(x), dtype=np.float32)
    t = n - d_samp
    x0 = np.floor(t).astype(np.int64)
    a  = (t - x0).astype(np.float32)

    y = np.zeros_like(x, dtype=np.float32)
    valid = (x0 >= 0) & (x0 + 1 < len(x))
    y[valid] = (1.0 - a[valid]) * x[x0[valid]] + a[valid] * x[x0[valid] + 1]
    return y

def doa_delays_samples(theta_deg: float, fs: int, M: int, d: float, c: float) -> np.ndarray:
    # ULA delays relative to mic 0
    theta = np.deg2rad(theta_deg)
    # tau_m = (m*d*cos(theta))/c  (m=0..M-1)
    taus = np.array([(m * d * np.cos(theta)) / c for m in range(M)], dtype=np.float32)
    return taus * fs

# ---- load ----
AUDIO_PATH = Path(__file__).resolve().parent / AUDIO_FILE
if not AUDIO_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy: {AUDIO_PATH}")

fs, x = wavfile.read(str(AUDIO_PATH))
x = to_float32(x)

# source: nếu stereo (N,2) thì lấy channel 0 làm nguồn
if x.ndim == 2:
    src = x[:, 0].copy()
else:
    src = x.copy()

# ---- cut segment ----
start = int(START_SEC * fs)
end   = min(len(src), start + int(DURATION_SEC * fs))
src_seg = src[start:end]
t = np.arange(len(src_seg)) / fs

# ---- simulate 3 mics at theta=30 ----
delays = doa_delays_samples(THETA_DEG, fs, M, D, C)  # in samples
mics = []
for m in range(M):
    mics.append(fractional_delay(src_seg, float(delays[m])))
mics = np.stack(mics, axis=1)  # (Nseg, 3)

print(f"fs={fs}, segment={len(src_seg)} samples, theta={THETA_DEG} deg")
print("delays(samples) =", delays)

# ---- plot ----
plt.figure(figsize=(12, 9))

plt.subplot(4, 1, 1)
plt.plot(t, src_seg)
plt.title("Source (from audio 1.wav, channel 0 if stereo)")
plt.ylabel("Amp")

for i in range(3):
    plt.subplot(4, 1, i + 2)
    plt.plot(t, mics[:, i])
    plt.title(f"Mic {i+1} (theta={THETA_DEG}°)")
    plt.ylabel("Amp")

plt.tight_layout()
plt.show()

# optional overlay
plt.figure(figsize=(12, 4))
plt.plot(t, src_seg, label="Source")
plt.plot(t, mics[:, 0], label="Mic1")
plt.plot(t, mics[:, 1], label="Mic2")
plt.plot(t, mics[:, 2], label="Mic3")
plt.title("Overlay: Source + 3 Mics (simulated at theta=30°)")
plt.xlabel("Time (s)")
plt.ylabel("Amp")
plt.legend()
plt.tight_layout()
plt.show()

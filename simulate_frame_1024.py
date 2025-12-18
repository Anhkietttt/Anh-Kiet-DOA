import numpy as np
from pathlib import Path
from scipy.io import wavfile
import csv

# ==========================
# CONFIG (bản nhẹ)
# ==========================
M = 3
d = 0.20          # m
c = 343.0         # m/s
FRAME_LEN = 1024

MAX_FRAMES_PER_COMBO = 50          # mỗi (file, góc, SNR) tối đa 50 frame
ANGLES = list(range(0, 181, 5))    # 0,5,10,...,180  (37 góc)
SNRS   = [-5, 0, 5, 10]            # 4 mức SNR

raw_dir = Path(".")                # thư mục hiện tại
out_csv = "All_Frames_Corr_1024_light.csv"

np.random.seed(0)


# ==========================
# HÀM PHỤ
# ==========================
def load_mono_wav(path: Path):
    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
    return fs, data


def fractional_delay(sig: np.ndarray, delay_samples: float):
    n = np.arange(len(sig), dtype=np.float32)
    new_n = n - delay_samples
    delayed = np.interp(new_n, n, sig, left=0.0, right=0.0).astype(np.float32)
    return delayed


def add_awgn(sig: np.ndarray, snr_db: float):
    power_sig = np.mean(sig ** 2)
    if power_sig <= 0:
        return sig
    snr_lin = 10 ** (snr_db / 10.0)
    power_noise = power_sig / snr_lin
    noise = np.sqrt(power_noise) * np.random.randn(*sig.shape).astype(np.float32)
    return sig + noise


def frame_mics(Y: np.ndarray, frame_len: int):
    N = Y.shape[0]
    num_frames = N // frame_len
    if num_frames == 0:
        return np.empty((0, frame_len, Y.shape[1]), dtype=np.float32)
    Y_cut = Y[:num_frames * frame_len, :]
    frames = Y_cut.reshape(num_frames, frame_len, Y.shape[1])
    return frames


def compute_corr_matrix(frame_3mic: np.ndarray):
    X = frame_3mic.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    R = np.corrcoef(X, rowvar=False)
    return R.astype(np.float32)


# ==========================
# MAIN
# ==========================
def main():
    wav_files = sorted(raw_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} wav files in {raw_dir.resolve()}")

    header = ["File_Name", "Alpha", "SNR", "Frame_ID"] + [f"Corr_{i}" for i in range(1, 10)]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        total_rows = 0

        for wav_idx, wav_path in enumerate(wav_files):
            fs, audio = load_mono_wav(wav_path)
            print(f"\n[{wav_idx+1}/{len(wav_files)}] {wav_path.name}  fs={fs}, len={len(audio)}")

            if len(audio) < FRAME_LEN:
                print("  -> File quá ngắn, bỏ qua.")
                continue

            for alpha_idx, alpha in enumerate(ANGLES):
                theta = np.deg2rad(alpha)

                delays_time = np.array(
                    [0,
                     d * np.sin(theta) / c,
                     2 * d * np.sin(theta) / c],
                    dtype=np.float32
                )
                delays_samples = delays_time * fs

                mic_signals = []
                for m in range(M):
                    delayed = fractional_delay(audio, delays_samples[m])
                    mic_signals.append(delayed)
                Y_clean = np.stack(mic_signals, axis=-1)  # (N,3)

                for snr_db in SNRS:
                    Y_noisy = np.stack(
                        [add_awgn(Y_clean[:, m], snr_db) for m in range(M)],
                        axis=-1
                    )

                    frames = frame_mics(Y_noisy, FRAME_LEN)
                    num_frames = frames.shape[0]
                    if num_frames == 0:
                        continue

                    if num_frames > MAX_FRAMES_PER_COMBO:
                        idxs = np.random.choice(num_frames,
                                                size=MAX_FRAMES_PER_COMBO,
                                                replace=False)
                    else:
                        idxs = np.arange(num_frames)

                    for fid in idxs:
                        frame = frames[fid]
                        R = compute_corr_matrix(frame)
                        corr_flat = R.flatten(order="C")

                        row = [wav_path.name, alpha, snr_db, int(fid)] + corr_flat.tolist()
                        writer.writerow(row)
                        total_rows += 1

                # in progress theo góc
                if (alpha_idx + 1) % 5 == 0:
                    print(f"  -> góc {alpha}° xong, total_rows ~ {total_rows}")

        print("\n==========================")
        print("Saved dataset to:", Path(out_csv).resolve())
        print("Total rows:", total_rows)


if __name__ == "__main__":
    main()

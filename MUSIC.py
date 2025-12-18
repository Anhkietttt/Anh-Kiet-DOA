import numpy as np
import pandas as pd
from pathlib import Path

# =====================================================
# 0. CẤU HÌNH
# =====================================================
CSV_PATH = Path("All_Frames_Corr_1024_light.csv")

# Hình học mảng 3 mic (linear array)
c = 343.0           # tốc độ âm (m/s)
d = 0.20            # khoảng cách giữa 2 mic (m)
f0 = 4000.0         # tần số "trung tâm" giả định (Hz) – chỉnh cũng được
M  = 3              # số mic

# Quét góc cho MUSIC (độ)
THETA_MIN = 0.0
THETA_MAX = 180.0   # nếu chỉ quan tâm 0..90 thì sửa lại
THETA_STEP = 1.0

# Nếu dataset lớn quá, có thể giới hạn số frame để test nhanh
MAX_FRAMES = None   # None = dùng hết; hoặc ví dụ 20000

# =====================================================
# 1. ĐỌC DATA
# =====================================================
df = pd.read_csv(CSV_PATH)
print("===== THÔNG TIN DATA =====")
print("Columns:", df.columns.tolist())
print("Số dòng (frames):", len(df))

# Lấy label góc Alpha (deg)
y_true_deg = df["Alpha"].to_numpy(dtype=np.float64)

# Lấy 9 phần tử ma trận tương quan
feature_cols = [f"Corr_{i}" for i in range(1, 10)]
X_corr = df[feature_cols].to_numpy(dtype=np.float64)

if MAX_FRAMES is not None:
    X_corr = X_corr[:MAX_FRAMES]
    y_true_deg = y_true_deg[:MAX_FRAMES]
    print(f"Dùng {len(y_true_deg)} frame đầu tiên (MAX_FRAMES={MAX_FRAMES})")

N = X_corr.shape[0]
print("Số frame dùng cho MUSIC:", N)

# =====================================================
# 2. HÌNH HỌC MẢNG & STEERING VECTOR
# =====================================================
# Mic đặt trên trục x: [0, d, 2d]
mic_pos = np.array([0.0, d, 2*d])   # shape (3,)

# Lưới góc
theta_grid_deg = np.arange(THETA_MIN, THETA_MAX + 1e-9, THETA_STEP)
theta_grid_rad = np.deg2rad(theta_grid_deg)
K = len(theta_grid_deg)

# Wavenumber
k0 = 2.0 * np.pi * f0 / c

# Precompute steering matrix A (M x K)
# a(theta) = exp(-j * k0 * x_m * sin(theta))
A = np.zeros((M, K), dtype=np.complex128)
for i, th in enumerate(theta_grid_rad):
    phase = k0 * mic_pos * np.sin(th)
    A[:, i] = np.exp(-1j * phase)

print("Steering matrix A shape:", A.shape)  # (3, K)

# =====================================================
# 3. HÀM HỖ TRỢ
# =====================================================
def angular_diff_deg(true_deg, pred_deg):
    """Sai số góc có wrap-around."""
    diff = np.abs(true_deg - pred_deg)
    diff = np.minimum(diff, 360.0 - diff)
    return diff

def row_to_R(row_corr):
    """Chuyển 9 số Corr thành ma trận 3x3 (row-major)."""
    R = row_corr.reshape(3, 3)
    return R

# =====================================================
# 4. MUSIC DOA ESTIMATION
# =====================================================
y_pred_deg = np.zeros_like(y_true_deg)

for n in range(N):
    corr_flat = X_corr[n]
    R = row_to_R(corr_flat)

    # Đảm bảo R đối xứng (trong thực tế nên vậy)
    R = 0.5 * (R + R.T)

    # Eigen-decomposition R (Hermitian)
    eigvals, eigvecs = np.linalg.eigh(R)

    # Sắp xếp eigenvalue tăng dần
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Giả sử 1 nguồn -> noise subspace có M-1 vector
    En = eigvecs[:, :M-1]         # shape (3, 2)

    # MUSIC spectrum:
    # P(theta) = 1 / || En^H a(theta) ||^2
    EnH_A = En.conj().T @ A       # (2, K)
    denom = np.sum(np.abs(EnH_A)**2, axis=0)  # (K,)
    P = 1.0 / denom

    # Góc ước lượng = vị trí đỉnh phổ
    idx_max = np.argmax(P)
    theta_hat = theta_grid_deg[idx_max]

    # Nếu bạn chỉ quan tâm 0..180 và muốn mirror về 0..180:
    # (ở đây lưới đã là 0..180 nên không cần)
    y_pred_deg[n] = theta_hat

    if (n+1) % 5000 == 0 or n == 0:
        print(f"[{n+1}/{N}]  Alpha_true={y_true_deg[n]:6.2f}°, "
              f"Alpha_hat={theta_hat:6.2f}°")

# =====================================================
# 5. ĐÁNH GIÁ RMSE / MAE
# =====================================================
# Nếu thực sự góc vật lý chỉ 0..180 thì có thể mirror:
y_true_eval = np.where(y_true_deg > 180.0, 360.0 - y_true_deg, y_true_deg)
y_pred_eval = np.where(y_pred_deg > 180.0, 360.0 - y_pred_deg, y_pred_deg)

err = angular_diff_deg(y_true_eval, y_pred_eval)
rmse = np.sqrt(np.mean(err**2))
mae  = np.mean(err)

print("\n===== KẾT QUẢ MUSIC TRÊN TOÀN BỘ DATA =====")
print(f"MUSIC RMSE: {rmse:.3f} độ")
print(f"MUSIC MAE : {mae:.3f} độ")

print("\n===== 10 MẪU ĐẦU TIÊN =====")
for i in range(min(10, N)):
    print(f"Sample {i:2d}: True={y_true_eval[i]:6.2f}°, "
          f"Pred={y_pred_eval[i]:6.2f}°, Err={err[i]:6.2f}°")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# =========================================
# 0. CẤU HÌNH TÊN FILE CSV
# =========================================
CSV_PATH = Path("All_Frames_Corr_1024_light.csv")   # file simulate bạn đã tạo


# =========================================
# 1. ĐỌC DATA GỐC (MỖI FRAME LÀ 1 SAMPLE)
# =========================================
df = pd.read_csv(CSV_PATH)

print("===== THÔNG TIN DATA GỐC =====")
print("Columns:", df.columns.tolist())
print("Số dòng (frame):", len(df))


# =========================================
# 2. GỘP THEO (File_Name, Alpha, SNR)
#    -> mỗi (file, góc, SNR) là 1 sample
# =========================================
corr_cols = [f"Corr_{i}" for i in range(1, 10)]
group_cols = ["File_Name", "Alpha", "SNR"]

df_group = (
    df.groupby(group_cols)[corr_cols]
      .mean()
      .reset_index()
)

print("\n===== SAU KHI GỘP FRAME =====")
print("Số sample (file, góc, SNR):", len(df_group))
print(df_group.head())


# =========================================
# 3. TẠO LABEL MỚI: Beta = min(alpha, 180-alpha)
#    -> chỉ 0..90 độ (giải quyết ambiguity front/back)
# =========================================
alpha_raw = df_group["Alpha"].values.astype(np.float32)
beta = np.minimum(alpha_raw, 180.0 - alpha_raw)   # label mới

df_group["Beta"] = beta

print("\nVí dụ Alpha -> Beta:")
print(df_group[["Alpha", "Beta"]].head(10))


# =========================================
# 4. TẠO X, y
# =========================================
# Feature: Corr_1..Corr_9
feature_cols = corr_cols

# Nếu muốn thêm SNR làm feature thì dùng:
# feature_cols = ["SNR"] + corr_cols

X = df_group[feature_cols].values.astype(np.float32)
y = df_group["Beta"].values.astype(np.float32)

print("\n===== SHAPE =====")
print("X shape:", X.shape)   # (N_sample, 9)
print("y shape:", y.shape)   # (N_sample,)


# =========================================
# 5. CHIA TRAIN / TEST
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("\n===== CHIA TẬP =====")
print("Train size:", X_train.shape[0])
print("Test size :", X_test.shape[0])


# =========================================
# 6. SCALE FEATURES
# =========================================
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


# =========================================
# 7. KHAI BÁO MODEL
# =========================================
models = {
    "LinearRegression": LinearRegression(),
    "SVR_rbf": SVR(kernel="rbf", C=10.0, gamma="scale"),
    "DecisionTree": DecisionTreeRegressor(
        max_depth=15,
        random_state=42
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),
}


# =========================================
# 8. HÀM TÍNH RMSE
# =========================================
def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


# =========================================
# 9. TRAIN + ĐÁNH GIÁ
# =========================================
results = []

for name, model in models.items():
    print(f"\n===== Training {name} =====")
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    score_rmse = rmse(y_test, y_pred)

    print(f"{name} RMSE (deg): {score_rmse:.3f}")
    results.append((name, score_rmse))


# =========================================
# 10. TỔNG KẾT
# =========================================
results_sorted = sorted(results, key=lambda x: x[1])

print("\n===== SUMMARY (RMSE càng thấp càng tốt, Beta 0..90°) =====")
for name, score in results_sorted:
    print(f"{name:15s}: {score:8.3f} độ")

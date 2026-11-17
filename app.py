"""Dự đoán năng suất lúa theo mùa vụ bằng SVR – chạy trên Streamlit Community Cloud."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

st.set_page_config(page_title="Dự đoán năng suất lúa – SVR", page_icon="🌾", layout="wide")

# ------------------------------------------------------------------ cấu trúc dữ liệu
TARGET = "nang_suat"
CAT = {"mua_vu": "Mùa vụ", "giong_lua": "Giống lúa"}
NUM = {  # cột: (nhãn, bước nhập)
    "nhiet_do_tb": ("Nhiệt độ TB (°C)", 0.1),
    "luong_mua": ("Tổng lượng mưa (mm/vụ)", 10.0),
    "gio_nang": ("Số giờ nắng (giờ/ngày)", 0.1),
    "do_am": ("Độ ẩm TB (%)", 1.0),
    "phan_N": ("Phân N (kg/ha)", 5.0),
    "phan_P2O5": ("Phân P₂O₅ (kg/ha)", 5.0),
    "phan_K2O": ("Phân K₂O (kg/ha)", 5.0),
    "luong_giong": ("Lượng giống (kg/ha)", 5.0),
}
GROUPS = {
    "🌤️ Thời tiết trong vụ": ["nhiet_do_tb", "luong_mua", "gio_nang", "do_am"],
    "🌱 Canh tác": ["phan_N", "phan_P2O5", "phan_K2O", "luong_giong"],
}
FEATURES = list(CAT) + list(NUM)
DATA_FILE = Path(__file__).with_name("du_lieu_lua.csv")  # (tuỳ chọn) dữ liệu thật của bạn


# ------------------------------------------------------------------ dữ liệu
def make_demo_data(n=1200, seed=42):
    """Dữ liệu MÔ PHỎNG theo 3 vụ ở ĐBSCL – chỉ để minh hoạ."""
    rng = np.random.default_rng(seed)
    cfg = {
        "Đông Xuân": dict(t=(25.5, 1.2), r=(180, 70), s=(7.4, 0.6), h=(74, 4), base=6.6),
        "Hè Thu": dict(t=(28.0, 1.0), r=(900, 200), s=(5.8, 0.7), h=(80, 4), base=5.4),
        "Thu Đông": dict(t=(27.5, 1.0), r=(1000, 220), s=(5.2, 0.7), h=(82, 4), base=5.0),
    }
    giong = {"OM18": 0.25, "OM5451": 0.15, "IR50404": 0.0, "Jasmine 85": -0.2, "ST25": -0.7}
    rows = []
    for _ in range(n):
        season = str(rng.choice(list(cfg)))
        variety = str(rng.choice(list(giong)))
        c = cfg[season]
        t, s, h = rng.normal(*c["t"]), rng.normal(*c["s"]), rng.normal(*c["h"])
        r = max(rng.normal(*c["r"]), 20)
        N, P, K, g = rng.uniform(60, 140), rng.uniform(20, 80), rng.uniform(20, 80), rng.uniform(80, 180)
        y = (c["base"] + giong[variety]
             - 0.00018 * (N - 105) ** 2 - 0.0002 * (P - 50) ** 2 - 0.0002 * (K - 50) ** 2
             - 0.00012 * (g - 110) ** 2 - 0.10 * (t - 26.5) ** 2 + 0.20 * (s - 6.0)
             - 0.0004 * max(r - 900, 0) + rng.normal(0, 0.25))
        rows.append(dict(mua_vu=season, giong_lua=variety, nhiet_do_tb=round(t, 1), luong_mua=round(r),
                         gio_nang=round(s, 1), do_am=round(h, 1), phan_N=round(N), phan_P2O5=round(P),
                         phan_K2O=round(K), luong_giong=round(g),
                         nang_suat=round(float(np.clip(y, 2.5, 9.0)), 2)))
    return pd.DataFrame(rows)


@st.cache_data
def load_data():
    """Ưu tiên file du_lieu_lua.csv trong repo; nếu không có thì dùng dữ liệu mô phỏng."""
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
        missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
        if missing:
            raise ValueError("File du_lieu_lua.csv thiếu cột: " + ", ".join(missing))
        df = df[FEATURES + [TARGET]].dropna().copy()
        for c in CAT:
            df[c] = df[c].astype(str)
        return df, True
    return make_demo_data(), False


# ------------------------------------------------------------------ mô hình SVR
@st.cache_resource(show_spinner="Đang huấn luyện mô hình SVR (chỉ lần đầu)...")
def train(df):
    X, y = df[FEATURES], df[TARGET].astype(float)
    vc = X["mua_vu"].value_counts()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X["mua_vu"] if vc.min() >= 5 else None)

    pre = ColumnTransformer([
        ("num", StandardScaler(), list(NUM)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(CAT)),
    ])
    base = TransformedTargetRegressor(
        regressor=Pipeline([("pre", pre), ("svr", SVR(kernel="rbf"))]),
        transformer=StandardScaler())
    grid = {"regressor__svr__C": [10, 30, 100],
            "regressor__svr__epsilon": [0.05, 0.15],
            "regressor__svr__gamma": ["scale", 0.01, 0.03, 0.1]}
    gs = GridSearchCV(base, grid, cv=3, scoring="neg_root_mean_squared_error")
    gs.fit(Xtr, ytr)

    pred = gs.predict(Xte)
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    info = {"R²": r2_score(yte, pred), "RMSE": rmse, "MAE": mean_absolute_error(yte, pred),
            "Số mẫu": len(df), "Tham số": {k.split("__")[-1]: v for k, v in gs.best_params_.items()}}
    final = clone(gs.best_estimator_).fit(X, y)  # huấn luyện lại trên toàn bộ dữ liệu
    return final, info


# ------------------------------------------------------------------ giao diện
st.title("🌾 Dự đoán năng suất lúa theo mùa vụ")
st.caption("Mô hình Support Vector Regression (SVR, kernel RBF)")

try:
    df, is_real = load_data()
except ValueError as e:
    st.error(str(e)); st.stop()

if not is_real:
    st.warning("Mô hình đang được huấn luyện trên **dữ liệu mô phỏng** nên kết quả chỉ mang tính minh hoạ. "
               "Thêm file `du_lieu_lua.csv` vào repo GitHub để dùng số liệu thực tế.")

model, info = train(df)

with st.sidebar:
    st.header("Thông tin mô hình")
    st.caption("Dữ liệu: " + ("số liệu thực tế (du_lieu_lua.csv)" if is_real else "mô phỏng"))
    st.metric("R² (tập kiểm tra)", f"{info['R²']:.3f}")
    st.metric("RMSE (tấn/ha)", f"{info['RMSE']:.3f}")
    st.metric("MAE (tấn/ha)", f"{info['MAE']:.3f}")
    st.caption(f"{info['Số mẫu']:,} mẫu · " + ", ".join(f"{k}={v}" for k, v in info["Tham số"].items()))

# ---- form nhập liệu
with st.form("form_nhap"):
    st.subheader("Nhập thông tin vụ lúa")
    c1, c2 = st.columns(2)
    inputs = {
        "mua_vu": c1.selectbox(CAT["mua_vu"], sorted(df["mua_vu"].unique())),
        "giong_lua": c2.selectbox(CAT["giong_lua"], sorted(df["giong_lua"].unique())),
    }
    for title, cols in GROUPS.items():
        st.markdown(f"**{title}**")
        for box, col in zip(st.columns(len(cols)), cols):
            label, step = NUM[col]
            lo, hi = float(df[col].min()), float(df[col].max())
            inputs[col] = box.number_input(label, min_value=0.0, step=step, format="%.1f",
                                           value=float(round(df[col].median(), 1)),
                                           help=f"Khoảng trong dữ liệu huấn luyện: {lo:g} – {hi:g}")
    area = st.number_input("Diện tích canh tác (ha)", min_value=0.0, value=1.0, step=0.5)
    submitted = st.form_submit_button("🔮 Dự đoán năng suất", type="primary")

if submitted:
    st.session_state["last"] = dict(inputs=inputs, area=area)

last = st.session_state.get("last")
if not last:
    st.info("Nhập thông tin ở trên rồi bấm **Dự đoán năng suất**.")
    st.stop()

# ---- kết quả
x, area = last["inputs"], last["area"]
y_hat = float(model.predict(pd.DataFrame([x]))[0])
rmse = info["RMSE"]
season_mean = df.loc[df["mua_vu"] == x["mua_vu"], TARGET].mean()

st.divider()
st.subheader(f"Kết quả: vụ {x['mua_vu']} · giống {x['giong_lua']}")
k1, k2, k3 = st.columns(3)
k1.metric("Năng suất dự đoán", f"{y_hat:.2f} tấn/ha",
          f"{y_hat - season_mean:+.2f} so với TB vụ {x['mua_vu']}")
k2.metric("Khoảng tham khảo (±RMSE)", f"{y_hat - rmse:.2f} – {y_hat + rmse:.2f}")
k3.metric("Sản lượng ước tính", f"{y_hat * area:,.1f} tấn")

out = [NUM[c][0] for c in NUM if not df[c].min() <= x[c] <= df[c].max()]
if out:
    st.warning("Giá trị nằm ngoài khoảng dữ liệu huấn luyện (kết quả kém tin cậy): " + ", ".join(out))

# ---- so sánh vụ / giống
a, b = st.columns(2)
seasons, varieties = sorted(df["mua_vu"].unique()), sorted(df["giong_lua"].unique())
a.caption("So sánh các mùa vụ (giữ nguyên các thông tin khác) – tấn/ha")
a.bar_chart(pd.Series(model.predict(pd.DataFrame([{**x, "mua_vu": s} for s in seasons])), index=seasons))
b.caption("So sánh các giống lúa (giữ nguyên các thông tin khác) – tấn/ha")
b.bar_chart(pd.Series(model.predict(pd.DataFrame([{**x, "giong_lua": v} for v in varieties])), index=varieties))

# ---- độ nhạy
st.subheader("Phân tích độ nhạy")
var = st.selectbox("Yếu tố cần thay đổi", list(NUM), format_func=lambda c: NUM[c][0])
xs = np.linspace(float(df[var].min()), float(df[var].max()), 40)
ys = model.predict(pd.DataFrame([{**x, var: v} for v in xs]))
st.line_chart(pd.DataFrame({NUM[var][0]: xs, "Năng suất dự đoán (tấn/ha)": ys}).set_index(NUM[var][0]))
st.caption("Đường cong cho thấy năng suất dự đoán thay đổi thế nào khi chỉ đổi yếu tố đã chọn, các yếu tố khác giữ nguyên.")

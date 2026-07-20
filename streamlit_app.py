"""
streamlit_app.py
=================
Dashboard interaktif analisis sintesis AgNPs (Streamlit).
Diadaptasi dari nanoparticle_analysis.py, dengan tambahan:
  - Model RSM kuadratik + ANOVA per respons (mirip Design-Expert)
  - Validasi model (Adjusted-R2 vs Predicted-R2 / LOO-CV)
  - Optimasi multi-respons pakai desirability function (bukan skor ad-hoc)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from itertools import combinations
from sklearn.model_selection import LeaveOneOut
from scipy.optimize import differential_evolution

st.set_page_config(page_title="AgNPs Synthesis Analyzer", layout="wide")

COLORS = {"primary": "#1F4E79", "accent": "#70AD47", "danger": "#C00000"}

st.title("🔬 AgNPs Synthesis Analyzer")
st.caption("Analisis statistik & optimasi multi-respons untuk sintesis nanopartikel perak")

# ── 1. Load data ────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload file Excel (sheet 'Raw_Data')", type=["xlsx"])
if uploaded is None:
    st.info("Silakan upload file .xlsx untuk memulai. Kolom minimal: konsentrasi, suhu, pH, waktu, "
            "ukuran partikel, PDI, zeta potensial. Yield & SPR opsional.")
    st.stop()

raw = pd.read_excel(uploaded, sheet_name="Raw_Data", header=1)
raw = raw.rename(columns=lambda c: str(c).strip())

st.subheader("Pratinjau Data")
st.dataframe(raw.head(10), use_container_width=True)
st.write(f"**{raw.shape[0]} baris x {raw.shape[1]} kolom**")

numeric_cols = raw.select_dtypes(include=np.number).columns.tolist()

# ── 2. Pilih kolom faktor & respons secara eksplisit (jangan asumsi nama) ──
st.subheader("Konfigurasi Kolom")
c1, c2 = st.columns(2)
with c1:
    factor_cols = st.multiselect("Kolom FAKTOR (variabel bebas)", numeric_cols,
                                  default=[c for c in numeric_cols if c.lower() not in
                                           ("run", "run #")][:4])
with c2:
    response_cols = st.multiselect("Kolom RESPON (variabel terikat)",
                                    [c for c in numeric_cols if c not in factor_cols])

if not factor_cols or not response_cols:
    st.warning("Pilih minimal 1 kolom faktor dan 1 kolom respons untuk lanjut.")
    st.stop()

goal_map = {}
st.write("Tujuan tiap respons (untuk optimasi):")
for r in response_cols:
    goal_map[r] = st.radio(f"Tujuan **{r}**", ["Minimalkan", "Maksimalkan"],
                            horizontal=True, key=f"goal_{r}")

# ── 3. Statistik deskriptif + normalitas ────────────────────────────────────
st.subheader("Statistik Deskriptif & Uji Normalitas (Shapiro-Wilk)")
desc = raw[factor_cols + response_cols].describe().T
sw_rows = []
for col in factor_cols + response_cols:
    W, p = stats.shapiro(raw[col].dropna())
    sw_rows.append({"Variabel": col, "Shapiro W": round(W, 4), "p-value": round(p, 4),
                     "Normal?": "Ya" if p > 0.05 else "Tidak"})
st.dataframe(desc.round(3), use_container_width=True)
st.dataframe(pd.DataFrame(sw_rows), use_container_width=True)
st.caption("Catatan: Shapiro-Wilk di sini menguji normalitas data mentah tiap variabel (screening awal). "
           "Uji yang relevan untuk validitas ANOVA/RSM adalah normalitas **residual model**, "
           "yang ditampilkan di bagian Model RSM di bawah.")

# ── 4. Korelasi ──────────────────────────────────────────────────────────────
st.subheader("Korelasi Faktor vs Respons")
fig, ax = plt.subplots(figsize=(6, 5))
corr = raw[factor_cols + response_cols].corr(method="pearson")
sns.heatmap(corr.loc[factor_cols, response_cols], annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Pearson r: Faktor (baris) vs Respons (kolom)")
st.pyplot(fig)

# ── 5. Model RSM kuadratik per respons ──────────────────────────────────────
st.subheader("Model RSM Kuadratik per Respons (mirip Design-Expert)")

def build_formula(resp, factors):
    terms = list(factors)
    terms += [f"I({a}*{b})" for a, b in combinations(factors, 2)]
    terms += [f"I({f}**2)" for f in factors]
    return f"{resp} ~ " + " + ".join(terms)

models, diag_rows = {}, []
data_clean = raw[factor_cols + response_cols].dropna()

for r in response_cols:
    formula = build_formula(r, factor_cols)
    m = smf.ols(formula, data=data_clean).fit()
    models[r] = m
    W, p_sw = stats.shapiro(m.resid)

    # Predicted-R2 via LOO-CV
    y = data_clean[r].values
    loo = LeaveOneOut()
    preds = np.zeros(len(data_clean))
    for tr, te in loo.split(data_clean):
        mm = smf.ols(formula, data=data_clean.iloc[tr]).fit()
        preds[te] = mm.predict(data_clean.iloc[te])
    pred_r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
    gap = m.rsquared_adj - pred_r2

    status = "OK"
    if abs(gap) >= 0.2:
        status = "⚠️ kemungkinan overfit"
    if p_sw < 0.05:
        status += " | ⚠️ residual tidak normal"

    diag_rows.append({
        "Respons": r, "R2": round(m.rsquared, 3), "Adj-R2": round(m.rsquared_adj, 3),
        "Pred-R2 (LOO)": round(pred_r2, 3), "p(model)": round(m.f_pvalue, 5),
        "Shapiro resid p": round(p_sw, 4), "Status": status
    })

diag_df = pd.DataFrame(diag_rows)
st.dataframe(diag_df, use_container_width=True)
st.caption("Adj-R2 vs Pred-R2: selisih < 0.2 dianggap wajar (aturan umum di Design-Expert). "
           "Kalau selisihnya besar, model overfit dan TIDAK boleh dipakai untuk ekstrapolasi/optimasi.")

with st.expander("Lihat koefisien signifikan tiap model (p<0.05)"):
    for r in response_cols:
        st.markdown(f"**{r}**")
        pv = models[r].pvalues.drop("Intercept", errors="ignore")
        sig = pv[pv < 0.05].sort_values()
        if len(sig) == 0:
            st.write("- Tidak ada term individual signifikan pada alpha=0.05")
        else:
            for term, p in sig.items():
                st.write(f"- `{term}`: coef={models[r].params[term]:+.4f}, p={p:.5f}")

# ── 6. Optimasi multi-respons (desirability function) ──────────────────────
st.subheader("Optimasi Multi-Respons (Desirability Function)")

valid_responses = diag_df[diag_df["Status"] == "OK"]["Respons"].tolist()
excluded = [r for r in response_cols if r not in valid_responses]
if excluded:
    st.warning(f"Respons berikut DIKELUARKAN dari optimasi karena model tidak valid (overfit / "
               f"residual tidak normal): {', '.join(excluded)}. Perbaiki model atau tambah data dulu.")

if not valid_responses:
    st.error("Tidak ada model respons yang valid untuk dioptimasi.")
else:
    resp_ranges = {r: (data_clean[r].min(), data_clean[r].max()) for r in valid_responses}
    bounds = [(data_clean[f].min(), data_clean[f].max()) for f in factor_cols]

    def d_min(y, lo, hi):
        return 1.0 if y <= lo else (0.0 if y >= hi else (hi - y) / (hi - lo))

    def d_max(y, lo, hi):
        return 1.0 if y >= hi else (0.0 if y <= lo else (y - lo) / (hi - lo))

    def neg_D(x):
        pt = pd.DataFrame([dict(zip(factor_cols, x))])
        ds = []
        for r in valid_responses:
            yhat = models[r].predict(pt).values[0]
            lo, hi = resp_ranges[r]
            ds.append(d_min(yhat, lo, hi) if goal_map[r] == "Minimalkan" else d_max(yhat, lo, hi))
        return -(np.prod(ds) ** (1 / len(ds)))

    if st.button("Jalankan Optimasi"):
        with st.spinner("Mencari titik optimum di dalam rentang data (tidak ekstrapolasi)..."):
            res = differential_evolution(neg_D, bounds, seed=42, maxiter=60,
                                          popsize=15, tol=1e-6, polish=True)
        opt = dict(zip(factor_cols, res.x))
        st.success(f"Desirability keseluruhan (D) = {-res.fun:.4f} (skala 0-1)")

        oc1, oc2 = st.columns(2)
        with oc1:
            st.markdown("**Kondisi faktor optimum:**")
            for f in factor_cols:
                st.write(f"- {f}: {opt[f]:.3f}")
        with oc2:
            st.markdown("**Prediksi respons pada titik optimum:**")
            pt = pd.DataFrame([opt])
            for r in valid_responses:
                yhat = models[r].predict(pt).values[0]
                st.write(f"- {r}: {yhat:.3f}")

        st.caption("⚠️ Ini adalah hasil MODEL STATISTIK, bukan jaminan hasil eksperimen. "
                   "Validasi ulang di laboratorium pada titik ini sebelum dijadikan formula final.")

st.divider()
st.caption("Dibangun di atas script analisis asli (nanoparticle_analysis.py), dengan optimasi "
           "diganti dari skor ad-hoc menjadi desirability function multi-respons + validasi model.")

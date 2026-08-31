"""Streamlit GUI for block-wise functional PCA of nasal-acoustics trajectories."""
from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from skfda.preprocessing.dim_reduction import FPCA
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.representation.basis import BSplineBasis
from skfda.representation.grid import FDataGrid


st.set_page_config(page_title="Nasal Acoustics fPCA", layout="wide")

CUE_BLOCKS = {
    "Formants and bandwidth": ["F1_Hz", "F2_Hz", "F3_Hz", "B1_Hz", "B2_Hz", "B3_Hz"],
    "Harmonic-formant contrasts": ["A1_P0_dB", "A1_P1_dB", "A3_P0_dB", "H1_H2_dB"],
    "Spectral shape": ["spectral_tilt_dB_per_kHz", "spectral_CoG_Hz", "spectral_SD_Hz", "spectral_skew", "spectral_kurtosis"],
    "Nasal energy and resonance": ["P0_prominence_dB", "P1_prominence_dB", "energy_low_nasal", "energy_high", "nasal_murmur_ratio_dB"],
    "MFCC": [f"MFCC_{i:02d}" for i in range(1, 13)],
}


def read_table(upload, sheet):
    if upload.name.lower().endswith(".csv"):
        return pd.read_csv(upload)
    return pd.read_excel(upload, sheet_name=sheet)


def available_sheets(upload):
    if upload.name.lower().endswith(".csv"):
        return ["CSV"]
    return pd.ExcelFile(upload).sheet_names


def make_curves(df, cue, grid_points, normalisation):
    required = {"token_id", "speaker", "time_pct", cue}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    data = df.copy()
    data[cue] = pd.to_numeric(data[cue], errors="coerce")
    data["time_pct"] = pd.to_numeric(data["time_pct"], errors="coerce")
    data = data.dropna(subset=["token_id", "speaker", "time_pct", cue])
    if data.empty:
        raise ValueError(f"{cue} has no valid rows.")

    # One set of reference parameters per speaker and cue: never normalise token-by-token.
    if normalisation != "Raw":
        if normalisation == "Oral CVC reference z-score" and "nasality_condition" in data:
            reference = data[data["nasality_condition"] == "oral_non_nasal_CVC"]
        else:
            reference = data
        params = reference.groupby("speaker")[cue].agg(["mean", "std"])
        data = data.join(params, on="speaker")
        data["value"] = (data[cue] - data["mean"]) / data["std"].replace(0, np.nan)
    else:
        data["value"] = data[cue]
    data = data.dropna(subset=["value"])

    token_frames = []
    for token_id, token in data.groupby("token_id"):
        token = token.sort_values("time_pct").drop_duplicates("time_pct")
        x = token["time_pct"].to_numpy(float)
        if len(x) >= 2 and x.max() > x.min():
            token_frames.append((token_id, token, x.min(), x.max()))
    if len(token_frames) < 3:
        raise ValueError("Need at least three token trajectories with two valid time points.")
    # The extractor deliberately omits edge windows. Analyse the observed common
    # domain rather than extrapolating values into 0% or 100%.
    domain_low = max(item[2] for item in token_frames)
    domain_high = min(item[3] for item in token_frames)
    if domain_high <= domain_low:
        raise ValueError("Token trajectories have no common observed time domain.")
    grid = np.linspace(domain_low, domain_high, int(grid_points))
    curves, metadata = [], []
    meta_cols = [c for c in ["speaker", "file", "vowel", "nasality_condition", "phonemic_nasality_01"] if c in data]
    for token_id, token, _, _ in token_frames:
        x, y = token["time_pct"].to_numpy(float), token["value"].to_numpy(float)
        curves.append(np.interp(grid, x, y))
        row = {"token_id": token_id}
        row.update(token.iloc[0][meta_cols].to_dict())
        metadata.append(row)
    if len(curves) < 3:
        raise ValueError("Need at least three complete token trajectories after QC.")
    return np.asarray(curves), grid, pd.DataFrame(metadata)


def run_fpca(df, cue, grid_points, n_basis, n_components, normalisation):
    curves, grid, metadata = make_curves(df, cue, grid_points, normalisation)
    n_components = min(int(n_components), len(curves), int(n_basis))
    fd = FDataGrid(curves, grid_points=grid)
    basis = BSplineBasis(domain_range=(0, 100), n_basis=int(n_basis), order=4)
    smooth = BasisSmoother(basis=basis).fit_transform(fd)
    model = FPCA(n_components=n_components).fit(smooth)
    scores = model.transform(smooth)
    score_df = metadata.copy()
    for index in range(n_components):
        score_df[f"PC{index + 1}"] = scores[:, index]
    return {"cue": cue, "grid": grid, "curves": curves, "metadata": metadata,
            "model": model, "scores": scores, "score_df": score_df,
            "normalisation": normalisation}


def eigen_plot(state, sd):
    model, grid = state["model"], state["grid"]
    n = len(model.explained_variance_)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    mean = model.mean_.data_matrix[0, :, 0]
    for i, ax in enumerate(axes.flat):
        component = model.components_[i].data_matrix[0, :, 0]
        delta = sd * np.sqrt(model.explained_variance_[i]) * component
        ax.plot(grid, mean, "k--", label="Mean")
        ax.plot(grid, mean + delta, color="#D55E00", label=f"+{sd} SD")
        ax.plot(grid, mean - delta, color="#0072B2", label=f"-{sd} SD")
        ax.set(title=f"PC{i + 1} ({model.explained_variance_ratio_[i] * 100:.1f}%)", xlabel="Time in vowel (%)")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def reconstruction_plot(state, grouping=None, selected=None):
    model, scores, grid, meta = state["model"], state["scores"], state["grid"], state["metadata"]
    reconstructed = model.inverse_transform(scores).data_matrix[:, :, 0]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if selected is not None:
        index = int(np.flatnonzero(meta["token_id"].eq(selected))[0])
        ax.plot(grid, state["curves"][index], "--", color="0.45", label="Input trajectory")
        ax.plot(grid, reconstructed[index], color="#D55E00", lw=2.5, label="fPCA reconstruction")
    elif grouping and grouping in meta:
        for group, idx in meta.groupby(grouping).groups.items():
            ax.plot(grid, reconstructed[list(idx)].mean(axis=0), lw=2.5, label=f"{group} (n={len(idx)})")
    else:
        for curve in reconstructed:
            ax.plot(grid, curve, color="#0072B2", alpha=.18)
        ax.plot(grid, reconstructed.mean(axis=0), color="black", lw=3, label="Grand mean")
    ax.set(xlabel="Time in vowel (%)", ylabel=state["normalisation"], title=f"{state['cue']} reconstruction")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    return fig


st.title("Nasal acoustics: block-wise fPCA")
st.caption("Long-format trajectory Excel → selected acoustic cue → B-spline smoothing → fPCA. One cue is analysed per run; select a cue within each information block.")

upload = st.file_uploader("Upload extractor workbook (.xlsx) or long CSV", type=["xlsx", "csv"])
if upload is None:
    st.info("Upload the extractor workbook and select its `Trajectories` sheet.")
    st.stop()

sheets = available_sheets(upload)
sheet = st.selectbox("Excel sheet", sheets, index=sheets.index("Trajectories") if "Trajectories" in sheets else 0)
df = read_table(upload, sheet)
st.caption(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")

left, right = st.columns([1, 1])
with left:
    block = st.selectbox("Information block", list(CUE_BLOCKS))
    options = [c for c in CUE_BLOCKS[block] if c in df.columns]
    cue = st.selectbox("Acoustic cue", options)
    normalisation = st.selectbox("Normalisation", ["Raw", "Speaker-wise z-score", "Oral CVC reference z-score"])
with right:
    grid_points = st.slider("Grid points", 50, 200, 100, 10)
    n_basis = st.slider("B-Spline basis functions", 4, 15, 5)
    st.text_input("B-Spline order (fixed)", "4 (cubic)", disabled=True)
    n_components = st.slider("FPCA components", 2, 5, 3)
    sd_mult = st.slider("Eigenfunction plot ± SD", .5, 3., 2., .5)

if st.button("Run fPCA", type="primary"):
    try:
        st.session_state["nasal_fpca"] = run_fpca(df, cue, grid_points, n_basis, n_components, normalisation)
    except ValueError as error:
        st.error(str(error))

state = st.session_state.get("nasal_fpca")
if state is None:
    st.stop()

scores = state["score_df"]
st.success(f"{state['cue']}: {len(scores)} complete token trajectories; {len(state['scores'][0])} FPCA components.")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Eigenfunctions", "All reconstructions", "PC scores", "Individual", "Group reconstruction"])
with tab1:
    st.pyplot(eigen_plot(state, sd_mult), clear_figure=True)
with tab2:
    st.pyplot(reconstruction_plot(state), clear_figure=True)
with tab3:
    pcs = [c for c in scores if c.startswith("PC")]
    x, y = st.columns(2)
    pc_x = x.selectbox("X axis", pcs, index=0, key="pcx")
    pc_y = y.selectbox("Y axis", pcs, index=min(1, len(pcs)-1), key="pcy")
    groups = ["none"] + [c for c in ["nasality_condition", "phonemic_nasality_01", "speaker", "vowel"] if c in scores]
    colour = st.selectbox("Colour by", groups)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, sub in scores.groupby(colour) if colour != "none" else [("all", scores)]:
        ax.scatter(sub[pc_x], sub[pc_y], label=str(label), alpha=.75)
    ax.axhline(0, color="0.6", ls="--"); ax.axvline(0, color="0.6", ls="--")
    ax.set(xlabel=pc_x, ylabel=pc_y); ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.download_button("Download PC scores (.csv)", scores.to_csv(index=False).encode(), "nasal_fpca_scores.csv", "text/csv")
with tab4:
    token = st.selectbox("Token", scores["token_id"].tolist())
    st.pyplot(reconstruction_plot(state, selected=token), clear_figure=True)
    st.dataframe(scores[scores["token_id"] == token], hide_index=True)
with tab5:
    group_options = [c for c in ["nasality_condition", "phonemic_nasality_01", "speaker", "vowel"] if c in scores]
    group = st.selectbox("Group variable", group_options)
    st.pyplot(reconstruction_plot(state, grouping=group), clear_figure=True)

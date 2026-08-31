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


def make_curves(df, cue, grid_points, normalisation, min_coverage: float = 0.70):
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
    # Harmonic and resonance candidates can be undefined at isolated frames.
    # A strict intersection across every token is therefore often empty. Use the
    # central observed domain and retain only adequately observed trajectories.
    domain_low = float(np.quantile([item[2] for item in token_frames], 0.10))
    domain_high = float(np.quantile([item[3] for item in token_frames], 0.90))
    if domain_high <= domain_low:
        raise ValueError("Token trajectories have no common observed time domain.")
    grid = np.linspace(domain_low, domain_high, int(grid_points))
    curves, metadata = [], []
    meta_cols = [c for c in ["speaker", "file", "vowel", "nasality_condition", "phonemic_nasality_01"] if c in data]
    for token_id, token, _, _ in token_frames:
        x, y = token["time_pct"].to_numpy(float), token["value"].to_numpy(float)
        observed = (grid >= x.min()) & (grid <= x.max())
        coverage = float(observed.mean())
        if coverage < min_coverage:
            continue
        # np.interp linearly fills internal gaps and holds only short missing
        # edges. Coverage is exported so users can identify imputed trajectories.
        curves.append(np.interp(grid, x, y))
        row = {"token_id": token_id}
        row.update(token.iloc[0][meta_cols].to_dict())
        row["observed_grid_coverage"] = coverage
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
    return {"mode": "Univariate cue fPCA", "cue": cue, "grid": grid, "curves": curves, "metadata": metadata,
            "model": model, "scores": scores, "score_df": score_df,
            "normalisation": normalisation}


def run_joint_fpca(
    df,
    cues,
    grid_points,
    n_basis,
    n_components,
    normalisation,
    equal_cue_weighting=True,
):
    """Joint multivariate fPCA over a selected acoustic information block."""
    if len(cues) < 2:
        raise ValueError("Joint MFPCA requires at least two available cues.")

    prepared = {}
    common_tokens = None
    for cue in cues:
        curves, cue_grid, metadata = make_curves(df, cue, grid_points, normalisation)
        token_ids = metadata["token_id"].astype(str).tolist()
        prepared[cue] = {
            "grid": cue_grid,
            "curves": {token: curve for token, curve in zip(token_ids, curves)},
            "metadata": metadata.assign(token_id=metadata["token_id"].astype(str)),
        }
        token_set = set(token_ids)
        common_tokens = token_set if common_tokens is None else common_tokens & token_set

    common_tokens = sorted(common_tokens or [])
    if len(common_tokens) < 3:
        raise ValueError(
            "Fewer than three tokens have usable trajectories for every selected cue. "
            "Remove the sparsest cue or improve candidate detection."
        )

    domain_low = max(prepared[cue]["grid"][0] for cue in cues)
    domain_high = min(prepared[cue]["grid"][-1] for cue in cues)
    if domain_high <= domain_low:
        raise ValueError("Selected cues have no shared observed time domain.")
    grid = np.linspace(domain_low, domain_high, int(grid_points))

    cue_arrays = []
    cue_scales = {}
    for cue in cues:
        cue_grid = prepared[cue]["grid"]
        raw = np.vstack([
            np.interp(grid, cue_grid, prepared[cue]["curves"][token])
            for token in common_tokens
        ])
        fd = FDataGrid(raw, grid_points=grid)
        basis = BSplineBasis(domain_range=(domain_low, domain_high), n_basis=int(n_basis), order=4)
        smooth = BasisSmoother(basis=basis).fit_transform(fd).data_matrix[:, :, 0]
        scale = float(np.std(smooth, ddof=1)) if equal_cue_weighting else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        cue_scales[cue] = scale
        cue_arrays.append(smooth / scale)

    joint = np.stack(cue_arrays, axis=1)  # token × cue × time
    mean = joint.mean(axis=0)
    centred = joint - mean
    flat = centred.reshape(len(common_tokens), -1)
    u, singular, vt = np.linalg.svd(flat, full_matrices=False)
    n_components = min(int(n_components), len(common_tokens) - 1, vt.shape[0])
    scores = u[:, :n_components] * singular[:n_components]
    components = vt[:n_components].reshape(n_components, len(cues), len(grid))
    eigenvalues = singular[:n_components] ** 2 / max(len(common_tokens) - 1, 1)
    total_variance = float(np.sum(singular ** 2 / max(len(common_tokens) - 1, 1)))
    ratios = eigenvalues / total_variance if total_variance > 0 else np.zeros_like(eigenvalues)
    reconstructed = mean + np.einsum("ik,kmn->imn", scores, components)

    base_meta = prepared[cues[0]]["metadata"].drop_duplicates("token_id").set_index("token_id")
    metadata = base_meta.reindex(common_tokens).reset_index()
    coverage_cols = []
    for cue in cues:
        coverage = prepared[cue]["metadata"].set_index("token_id")["observed_grid_coverage"]
        col = f"coverage_{cue}"
        metadata[col] = metadata["token_id"].map(coverage)
        coverage_cols.append(col)
    metadata["minimum_block_coverage"] = metadata[coverage_cols].min(axis=1)
    score_df = metadata.copy()
    for index in range(n_components):
        score_df[f"PC{index + 1}"] = scores[:, index]

    contribution = np.sum(components ** 2, axis=2)
    contribution /= contribution.sum(axis=1, keepdims=True)
    contribution_df = pd.DataFrame(
        contribution.T,
        index=cues,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    ).reset_index(names="cue")

    return {
        "mode": "Joint information-block MFPCA",
        "cues": list(cues),
        "grid": grid,
        "joint": joint,
        "mean": mean,
        "scores": scores,
        "components": components,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": ratios,
        "reconstructed": reconstructed,
        "metadata": metadata,
        "score_df": score_df,
        "contribution_df": contribution_df,
        "cue_scales": cue_scales,
        "normalisation": normalisation,
        "equal_cue_weighting": equal_cue_weighting,
    }


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


def joint_eigen_plot(state, sd):
    cues, grid = state["cues"], state["grid"]
    n_pc = len(state["eigenvalues"])
    fig, axes = plt.subplots(n_pc, len(cues), figsize=(4 * len(cues), 3.2 * n_pc), squeeze=False)
    for pc in range(n_pc):
        delta_scale = sd * np.sqrt(state["eigenvalues"][pc])
        for j, cue in enumerate(cues):
            ax = axes[pc, j]
            mean = state["mean"][j]
            delta = delta_scale * state["components"][pc, j]
            ax.plot(grid, mean, "k--", lw=1.4)
            ax.plot(grid, mean + delta, color="#D55E00")
            ax.plot(grid, mean - delta, color="#0072B2")
            contribution = state["contribution_df"].set_index("cue").loc[cue, f"PC{pc + 1}"]
            ax.set_title(f"PC{pc + 1} · {cue}\ncue contribution {contribution:.1%}", fontsize=9)
            ax.grid(alpha=.22)
            if pc == n_pc - 1:
                ax.set_xlabel("Time in vowel (%)")
    fig.suptitle("Joint eigenfunctions: mean ± SD in block-scaled units", fontweight="bold")
    fig.tight_layout()
    return fig


def joint_reconstruction_plot(state, grouping=None, selected=None):
    cues, grid, meta = state["cues"], state["grid"], state["metadata"]
    reconstructed = state["reconstructed"]
    fig, axes = plt.subplots(1, len(cues), figsize=(4 * len(cues), 4), squeeze=False)
    for j, cue in enumerate(cues):
        ax = axes[0, j]
        if selected is not None:
            idx = int(np.flatnonzero(meta["token_id"].astype(str).eq(str(selected)))[0])
            ax.plot(grid, state["joint"][idx, j], "--", color="0.45", label="Input")
            ax.plot(grid, reconstructed[idx, j], color="#D55E00", lw=2.3, label="Reconstruction")
        elif grouping and grouping in meta:
            for group, idx in meta.groupby(grouping).groups.items():
                ax.plot(grid, reconstructed[list(idx), j].mean(axis=0), lw=2.2, label=f"{group} (n={len(idx)})")
        else:
            for curve in reconstructed[:, j]:
                ax.plot(grid, curve, color="#0072B2", alpha=.15)
            ax.plot(grid, reconstructed[:, j].mean(axis=0), color="black", lw=2.5, label="Grand mean")
        ax.set(title=cue, xlabel="Time in vowel (%)")
        ax.grid(alpha=.22)
        if j == 0:
            ax.set_ylabel("Block-scaled value")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


st.title("Nasal acoustics: block-wise fPCA")
st.caption("Long trajectory table → univariate cue fPCA or joint information-block MFPCA → scores and reconstructions.")

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
    analysis_mode = st.radio("Analysis mode", ["Univariate cue fPCA", "Joint information-block MFPCA"])
    block = st.selectbox("Information block", list(CUE_BLOCKS))
    options = [c for c in CUE_BLOCKS[block] if c in df.columns]
    if not options:
        st.error("No cues from this block are present in the uploaded table.")
        st.stop()
    if analysis_mode == "Univariate cue fPCA":
        selected_cues = [st.selectbox("Acoustic cue", options)]
    else:
        selected_cues = st.multiselect("Cues in joint block", options, default=options)
        equal_cue_weighting = st.checkbox(
            "Equal cue weighting",
            value=True,
            help="Scales each smoothed cue by its total SD before joint decomposition so Hz-valued cues cannot dominate by units alone.",
        )
    normalisation = st.selectbox("Normalisation", ["Raw", "Speaker-wise z-score", "Oral CVC reference z-score"])
with right:
    grid_points = st.slider("Grid points", 50, 200, 100, 10)
    n_basis = st.slider("B-Spline basis functions", 4, 15, 5)
    st.text_input("B-Spline order (fixed)", "4 (cubic)", disabled=True)
    n_components = st.slider("FPCA components", 2, 5, 3)
    sd_mult = st.slider("Eigenfunction plot ± SD", .5, 3., 2., .5)

if st.button("Run fPCA", type="primary"):
    try:
        if analysis_mode == "Univariate cue fPCA":
            result = run_fpca(df, selected_cues[0], grid_points, n_basis, n_components, normalisation)
        else:
            result = run_joint_fpca(
                df, selected_cues, grid_points, n_basis, n_components,
                normalisation, equal_cue_weighting,
            )
        st.session_state["nasal_fpca"] = result
    except ValueError as error:
        st.error(str(error))

state = st.session_state.get("nasal_fpca")
if state is None:
    st.stop()

scores = state["score_df"]
if state["mode"] == "Univariate cue fPCA":
    coverage = scores["observed_grid_coverage"]
    result_name = state["cue"]
else:
    coverage = scores["minimum_block_coverage"]
    result_name = f"Joint block: {', '.join(state['cues'])}"
st.success(f"{result_name}: {len(scores)} tokens; {state['scores'].shape[1]} components. Coverage median {coverage.median():.0%}, minimum {coverage.min():.0%}.")
st.caption(
    "The fPCA grid uses the central observed time domain. Tokens with <70% observed coverage are excluded; "
    "remaining short gaps are interpolated. Inspect observed_grid_coverage before interpreting a noisy cue."
)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Eigenfunctions", "All reconstructions", "PC scores", "Individual", "Group reconstruction"])
with tab1:
    if state["mode"] == "Univariate cue fPCA":
        st.pyplot(eigen_plot(state, sd_mult), clear_figure=True)
    else:
        st.pyplot(joint_eigen_plot(state, sd_mult), clear_figure=True)
        st.markdown("#### Cue contribution to each joint PC")
        st.dataframe(state["contribution_df"].style.format({c: "{:.1%}" for c in state["contribution_df"] if c.startswith("PC")}), hide_index=True)
with tab2:
    plot = reconstruction_plot(state) if state["mode"] == "Univariate cue fPCA" else joint_reconstruction_plot(state)
    st.pyplot(plot, clear_figure=True)
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
    if state["mode"] == "Joint information-block MFPCA":
        st.download_button("Download cue contributions (.csv)", state["contribution_df"].to_csv(index=False).encode(), "joint_fpca_cue_contributions.csv", "text/csv")
with tab4:
    token = st.selectbox("Token", scores["token_id"].tolist())
    plot = reconstruction_plot(state, selected=token) if state["mode"] == "Univariate cue fPCA" else joint_reconstruction_plot(state, selected=token)
    st.pyplot(plot, clear_figure=True)
    st.dataframe(scores[scores["token_id"] == token], hide_index=True)
with tab5:
    group_options = [c for c in ["nasality_condition", "phonemic_nasality_01", "speaker", "vowel"] if c in scores]
    group = st.selectbox("Group variable", group_options)
    plot = reconstruction_plot(state, grouping=group) if state["mode"] == "Univariate cue fPCA" else joint_reconstruction_plot(state, grouping=group)
    st.pyplot(plot, clear_figure=True)

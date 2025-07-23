
# streamlit_app.py
# Extended Streamlit app with dual luminescence normalization using BetaGlo and BacTiter-Glo

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
from io import StringIO
from math import pi

# Sidebar navigation
st.sidebar.title("MoA Fingerprint App")
page = st.sidebar.radio("Go to", ["Generate Fingerprints", "Upload & Cluster", "Match Unknowns", "Radar Chart"])

# Session state init
for key in ["reference_data", "reference_scaled", "scaler", "reference_features"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Page 1: Generate Fingerprints
if page == "Generate Fingerprints":
    st.title("🧬 Generate Expression Profiles from Raw Plate Data")

    raw_file = st.file_uploader("Upload raw plate data (CSV: Well, BetaGlo, BacTiterGlo)", type="csv")
    map_file = st.file_uploader("Upload well map (CSV: Well, Compound, Reporter, Dose, Control)", type="csv")

    if raw_file and map_file:
        raw_df = pd.read_csv(raw_file)
        map_df = pd.read_csv(map_file)

        # Merge on well
        merged = pd.merge(map_df, raw_df, on="Well")

        # Compute BetaGlo normalized to BacTiterGlo
        merged["NormalizedSignal"] = merged["BetaGlo"] / merged["BacTiterGlo"]

        # Normalize: fold change over mean control for same Reporter + Dose
        norm_data = []
        grouped = merged.groupby(["Compound", "Reporter", "Dose"])
        for (compound, reporter, dose), group in grouped:
            control_vals = group[group["Control"] == True]["NormalizedSignal"]
            if control_vals.empty:
                continue
            control_mean = control_vals.mean()
            test_vals = group[group["Control"] == False]["NormalizedSignal"]
            fold_changes = test_vals / control_mean
            fc_mean = fold_changes.mean()
            norm_data.append({
                "Compound": compound,
                "Reporter": reporter,
                "Dose": dose,
                "FoldChange": fc_mean
            })

        norm_df = pd.DataFrame(norm_data)
        norm_df["Feature"] = norm_df["Reporter"] + "_" + norm_df["Dose"].astype(str) + "x"
        pivot_df = norm_df.pivot(index="Compound", columns="Feature", values="FoldChange").reset_index()

        st.subheader("Generated Expression Profile Matrix")
        st.dataframe(pivot_df)
        st.download_button("Download Expression Profile CSV", pivot_df.to_csv(index=False), file_name="expression_profiles.csv")

# The remaining app (Cluster, Match, Radar) remains unchanged and uses this generated output

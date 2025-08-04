
# streamlit_app.py
# Enhanced multi-plate fingerprint generator with optional merged output and direct pass to clustering tab

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from io import BytesIO
from zipfile import ZipFile

st.sidebar.title("MoA Fingerprint App")
page = st.sidebar.radio("Go to", ["Generate Fingerprints (Multi-Plate)", "Upload & Cluster"])

# Session state init
for key in ["reference_data", "reference_scaled", "scaler", "reference_features"]:
    if key not in st.session_state:
        st.session_state[key] = None

if page == "Generate Fingerprints (Multi-Plate)":
    st.title("🧬 Generate Expression Profiles from Multiple Single-Plate Files")

    uploaded_raws = st.file_uploader("Upload one or more raw plate files (CSV: Well, BetaGlo, BacTiterGlo)", type="csv", accept_multiple_files=True)
    uploaded_maps = st.file_uploader("Upload corresponding well maps (CSV: Well, Compound, Reporter, Dose, Control)", type="csv", accept_multiple_files=True)

    merge_profiles = st.checkbox("Merge all expression profiles into a single CSV", value=True)
    auto_push = st.checkbox("Make merged profile available in clustering tab", value=True)

    # Check if we have files to process
    if uploaded_raws and uploaded_maps:
        if len(uploaded_maps) == 1 and len(uploaded_raws) > 1:
            st.info(f"Using 1 well map for {len(uploaded_raws)} raw plate files.")
            # Use the same well map for all raw files
            map_df = pd.read_csv(uploaded_maps[0])
            zip_buf = BytesIO()
            merged_profiles = []
            with ZipFile(zip_buf, 'w') as zip_out:
                for i in range(len(uploaded_raws)):
                    raw_df = pd.read_csv(uploaded_raws[i])

                    if "BetaGlo" not in raw_df.columns or "BacTiterGlo" not in raw_df.columns:
                        st.warning(f"Missing required columns in raw data file {uploaded_raws[i].name}")
                        continue

                    merged = pd.merge(map_df, raw_df, on="Well")
                    merged["NormalizedSignal"] = merged["BetaGlo"] / merged["BacTiterGlo"]

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

                    compound_name = pivot_df['Compound'].iloc[0]
                    st.subheader(f"📊 Expression Profile: {compound_name}")
                    st.dataframe(pivot_df)

                    merged_profiles.append(pivot_df)
                    csv_name = f"{compound_name}_expression_profile.csv"
                    zip_out.writestr(csv_name, pivot_df.to_csv(index=False))

            st.download_button("Download All Profiles as ZIP", data=zip_buf.getvalue(), file_name="all_expression_profiles.zip")

            if merge_profiles and merged_profiles:
                merged_df = pd.concat(merged_profiles, ignore_index=True).fillna(np.nan)
                st.subheader("🧩 Merged Expression Matrix")
                st.dataframe(merged_df)
                st.download_button("Download Merged CSV", merged_df.to_csv(index=False), file_name="merged_expression_profiles.csv")

                if auto_push:
                    st.session_state.reference_data = merged_df

        elif len(uploaded_raws) == len(uploaded_maps):
            st.info(f"{len(uploaded_raws)} file pairs detected. Processing one compound per pair.")
            zip_buf = BytesIO()
            merged_profiles = []
            with ZipFile(zip_buf, 'w') as zip_out:
                for i in range(len(uploaded_raws)):
                    raw_df = pd.read_csv(uploaded_raws[i])
                    map_df = pd.read_csv(uploaded_maps[i])

                    if "BetaGlo" not in raw_df.columns or "BacTiterGlo" not in raw_df.columns:
                        st.warning(f"Missing required columns in raw data file {uploaded_raws[i].name}")
                        continue

                    merged = pd.merge(map_df, raw_df, on="Well")
                    merged["NormalizedSignal"] = merged["BetaGlo"] / merged["BacTiterGlo"]

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

                    compound_name = pivot_df['Compound'].iloc[0]
                    st.subheader(f"📊 Expression Profile: {compound_name}")
                    st.dataframe(pivot_df)

                    merged_profiles.append(pivot_df)
                    csv_name = f"{compound_name}_expression_profile.csv"
                    zip_out.writestr(csv_name, pivot_df.to_csv(index=False))

            st.download_button("Download All Profiles as ZIP", data=zip_buf.getvalue(), file_name="all_expression_profiles.zip")

            if merge_profiles and merged_profiles:
                merged_df = pd.concat(merged_profiles, ignore_index=True).fillna(np.nan)
                st.subheader("🧩 Merged Expression Matrix")
                st.dataframe(merged_df)
                st.download_button("Download Merged CSV", merged_df.to_csv(index=False), file_name="merged_expression_profiles.csv")

                if auto_push:
                    st.session_state.reference_data = merged_df

        else:
            st.error(f"❌ File count mismatch: {len(uploaded_raws)} raw files and {len(uploaded_maps)} well maps.")
            st.info("💡 You can either:")
            st.info("   • Upload 1 well map for multiple raw files (same layout)")
            st.info("   • Upload 1 well map for each raw file (different layouts)")
    elif uploaded_raws or uploaded_maps:
        st.warning("⚠️ Please upload both raw plate files and well maps to proceed.")

elif page == "Upload & Cluster":
    st.title("📊 Upload & Cluster Known Compound Profiles")
    if st.session_state.reference_data is not None:
        df = st.session_state.reference_data
        st.info("Using merged fingerprint matrix from previous tab.")
    else:
        uploaded_file = st.file_uploader("Upload known expression profiles (CSV)", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.stop()

    st.write("Raw Data:", df.head())
    compound_names = df.iloc[:, 0]
    features_df = df.drop(columns=[df.columns[0]])

    # Check if we have enough data for analysis
    if len(df) < 2:
        st.error("❌ Need at least 2 compounds for clustering analysis.")
        st.stop()
    
    if features_df.shape[1] < 1:
        st.error("❌ Need at least 1 feature column for analysis.")
        st.stop()

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from scipy.cluster.hierarchy import linkage, dendrogram
    import matplotlib.pyplot as plt

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    st.session_state.reference_data = df
    st.session_state.scaler = scaler
    st.session_state.reference_scaled = X_scaled
    st.session_state.reference_features = features_df

    # PCA with error handling
    try:
        # Determine appropriate number of components
        n_components = min(2, min(X_scaled.shape[0] - 1, X_scaled.shape[1]))
        if n_components < 1:
            st.warning("⚠️ Insufficient data for PCA. Need at least 2 samples and 1 feature.")
            st.stop()
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        st.subheader("PCA Projection")
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1])
        for i, name in enumerate(compound_names):
            ax.annotate(name, (X_pca[i, 0], X_pca[i, 1]), fontsize=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"❌ PCA failed: {str(e)}")
        st.info("This might be due to insufficient data or singular covariance matrix.")
        st.stop()

    # Hierarchical clustering with error handling
    try:
        st.subheader("Hierarchical Clustering")
        linked = linkage(X_scaled, method="ward")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        dendrogram(linked, labels=compound_names.values, leaf_rotation=90, ax=ax2)
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"❌ Hierarchical clustering failed: {str(e)}")
        st.stop()

    # K-means clustering with validation
    max_clusters = min(len(df), 10)  # Don't allow more clusters than samples
    if max_clusters < 2:
        st.error("❌ Need at least 2 samples for clustering.")
        st.stop()
    
    k = st.slider("Select number of clusters (K)", 2, max_clusters, min(4, max_clusters))
    
    try:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df["Cluster"] = clusters

        st.subheader("Cluster Annotations")
        cluster_labels = {}
        for cluster_id in sorted(df["Cluster"].unique()):
            label = st.text_input(f"Label for Cluster {cluster_id}", key=f"cluster_{cluster_id}")
            cluster_labels[cluster_id] = label

        if st.button("Apply Labels and Save Reference"):
            df["MoA"] = df["Cluster"].map(cluster_labels)
            df.drop(columns=["Cluster"], inplace=True)
            st.session_state.reference_data = df
            st.download_button("Download Annotated Reference", df.to_csv(index=False), file_name="annotated_reference.csv")
            st.success("Reference dataset updated and ready for matching.")
            
    except Exception as e:
        st.error(f"❌ K-means clustering failed: {str(e)}")
        st.stop()

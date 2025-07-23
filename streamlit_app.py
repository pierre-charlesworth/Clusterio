
# streamlit_app.py
# Web-based GUI for Mode of Action (MoA) fingerprint clustering and matching
# Built with Streamlit, ready for deployment on Streamlit Cloud or Hugging Face Spaces

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
page = st.sidebar.radio("Go to", ["Upload & Cluster", "Match Unknowns", "Radar Chart"])

# Session state init
if "reference_data" not in st.session_state:
    st.session_state.reference_data = None
if "reference_scaled" not in st.session_state:
    st.session_state.reference_scaled = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "reference_features" not in st.session_state:
    st.session_state.reference_features = None

# Page 1: Upload & Cluster
if page == "Upload & Cluster":
    st.title("📊 Upload & Cluster Known Compound Profiles")

    uploaded_file = st.file_uploader("Upload known compound expression profiles (CSV)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Raw Data:", df.head())

        compound_names = df.iloc[:, 0]
        features_df = df.drop(columns=[df.columns[0]])

        # Normalize and scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df)

        st.session_state.reference_data = df
        st.session_state.scaler = scaler
        st.session_state.reference_scaled = X_scaled
        st.session_state.reference_features = features_df

        # PCA Plot
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        st.subheader("PCA Projection")
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1])
        for i, name in enumerate(compound_names):
            ax.annotate(name, (X_pca[i, 0], X_pca[i, 1]), fontsize=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)

        # Dendrogram
        st.subheader("Hierarchical Clustering")
        linked = linkage(X_scaled, method="ward")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        dendrogram(linked, labels=compound_names.values, leaf_rotation=90, ax=ax2)
        st.pyplot(fig2)

        # KMeans clustering
        k = st.slider("Select number of clusters (K)", 2, 10, 4)
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df["Cluster"] = clusters

        # User annotation for MoA labels
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

# Page 2: Match Unknowns
elif page == "Match Unknowns":
    st.title("🔍 Match Unknown Compounds")

    if st.session_state.reference_data is None:
        st.warning("Please upload and annotate a reference dataset first.")
    else:
        unknown_file = st.file_uploader("Upload unknown compound fingerprints (CSV)", type="csv")
        if unknown_file:
            unknown_df = pd.read_csv(unknown_file)
            st.write("Unknown Data:", unknown_df.head())

            unknown_names = unknown_df.iloc[:, 0].values
            unknown_features = unknown_df.drop(columns=[unknown_df.columns[0]])
            unknown_scaled = st.session_state.scaler.transform(unknown_features)

            # Perform cosine similarity
            results = []
            ref_df = st.session_state.reference_data
            ref_names = ref_df["Compound"]
            ref_moas = ref_df["MoA"]
            ref_vectors = st.session_state.reference_scaled

            for i, uvec in enumerate(unknown_scaled):
                sims = cosine_similarity([uvec], ref_vectors).flatten()
                top_idx = np.argmax(sims)
                results.append({
                    "Unknown Compound": unknown_names[i],
                    "Best Match": ref_names[top_idx],
                    "Most Likely MoA": ref_moas[top_idx],
                    "Certainty (%)": round(sims[top_idx] * 100, 2)
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            st.download_button("Download Match Results", results_df.to_csv(index=False), file_name="moa_match_results.csv")

# Page 3: Radar Chart
elif page == "Radar Chart":
    st.title("📈 Radar Chart Comparison")

    if st.session_state.reference_data is None:
        st.warning("No reference data loaded.")
    else:
        ref_df = st.session_state.reference_data
        ref_features = st.session_state.reference_features
        scaler = st.session_state.scaler

        st.subheader("Compare an Unknown Compound to its Best Match")
        unknown_file = st.file_uploader("Upload unknown compound again (CSV)", type="csv", key="radar")
        if unknown_file:
            unknown_df = pd.read_csv(unknown_file)
            unknown_names = unknown_df.iloc[:, 0].values
            unknown_features = unknown_df.drop(columns=[unknown_df.columns[0]])
            unknown_scaled = scaler.transform(unknown_features)
            ref_vectors = st.session_state.reference_scaled

            selection = st.selectbox("Select Unknown Compound", unknown_names)

            idx = list(unknown_names).index(selection)
            uvec = unknown_scaled[idx]
            sims = cosine_similarity([uvec], ref_vectors).flatten()
            top_idx = np.argmax(sims)
            best_match_vector = ref_vectors[top_idx]
            best_match_name = ref_df["Compound"].iloc[top_idx]
            certainty = round(sims[top_idx] * 100, 2)

            features_list = list(ref_features.columns)
            u_vals = uvec.tolist() + [uvec[0]]
            m_vals = best_match_vector.tolist() + [best_match_vector[0]]
            angles = [n / float(len(features_list)) * 2 * pi for n in range(len(features_list))] + [0]

            fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
            ax.plot(angles, u_vals, label="Unknown", linewidth=2)
            ax.fill(angles, u_vals, alpha=0.25)
            ax.plot(angles, m_vals, label=f"Match: {best_match_name}", linestyle="dashed")
            ax.fill(angles, m_vals, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features_list, fontsize=8)
            ax.set_title(f"Radar Match (Certainty: {certainty}%)")
            ax.legend(loc="upper right")
            st.pyplot(fig)

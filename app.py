# Versi Final app6.py setelah perbaikan lengkap

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from collections import Counter
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.spatial import KDTree
from streamlit_option_menu import option_menu
from kmodes.kprototypes import KPrototypes
import io
import os
import pickle


# === SIMPAN & MUAT session_state ===
SESSION_FILE = "session_state.pkl"

def save_session():
    keys_to_save = [
        "df_original", "df", "df_preprocessed", "df_encoded",
        "num_cols", "cat_cols", "n_clusters", "gbest",
        "df_clustered", "df_result", "label_encoders"
    ]
    data_to_save = {key: st.session_state.get(key) for key in keys_to_save if key in st.session_state}
    with open(SESSION_FILE, "wb") as f:
        pickle.dump(data_to_save, f)

def load_session():
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "rb") as f:
            data = pickle.load(f)
            for key, value in data.items():
                st.session_state[key] = value

# Panggil load_session di awal
load_session()

# Inisialisasi flag kontrol antar menu
st.session_state.setdefault('data_uploaded', False)
st.session_state.setdefault('data_preprocessed', False)
st.session_state.setdefault('data_encoded', False)
st.session_state.setdefault('clustering_done', False)


# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        "Menu", 
        ["Home", "Dataset", "Preprocessing Data", "Exploratory Data Analysis (EDA)", "Pra-Pemodelan", "Clustering", "Hasil", "Prediksi"],
        icons=["house", "folder", "database", "bar-chart", "tools", "gear", "check-circle", "lightbulb"],
        menu_icon="cast",
        default_index=0
    )


st.title("🏠 Pengelompokan Calon Penerima Bantuan Rehabilitasi Rutilahu Kota Surabaya")


if selected == "Home":
    st.markdown(
        """
        <style>
        .justified-text {
            text-align: justify;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="justified-text">
        Aplikasi ini digunakan untuk mengelompokkan data calon penerima bantuan Rehabilitasi Rumah Tidak Layak Huni (Rutilahu) di Kota Surabaya. 
        Proses penentuan penerima bantuan rehabilitasi Rutilahu masih menghadapi banyak tantangan, terutama dalam memastikan bahwa bantuan diberikan kepada pihak yang benar-benar membutuhkan. 
        Subjektivitas dalam penilaian serta kriteria seleksi yang belum terstandarisasi secara objektif sering kali menyebabkan bantuan tidak tepat sasaran. 
        Kondisi ini menimbulkan potensi kurangnya efisiensi, rumah tangga yang layak menerima bantuan terlewatkan, sementara pihak yang kurang prioritas justru mendapatkan alokasi. 
        Oleh sebab itu, diperlukan sistem otomatisasi berbasis data untuk mengelompokkan masyarakat yang berhak menerima bantuan rehabilitasi Rutilahu.
        </p>

        <p class="justified-text">
        Aplikasi ini melakukan clustering atau pengelompokan data Rutilahu secara otomatis menggunakan model clustering K-Prototype yang dioptimasi. 
        Adapun variabel yang menjadi pertimbangan yaitu Pendapatan, Jumlah Tanggungan, Luas Rumah, Kondisi Atap, Kondisi Dinding, dan Kondisi Lantai.
        Proses clustering diawali dengan menyiapkan data Rutilahu dan dilakukan beberapa proses awal. 
        Dilakukannya proses cleansing, EDA, dan pra-pemodelan. Kemudian, dilakukan proses utama pengelompokan atau clustering 
        dengan optimasi pusat cluster menggunakan Particle Swarm Optimization (PSO).
        </p>
        """, 
        unsafe_allow_html=True
    )


elif selected == "Dataset":
    st.write("### 📤 Upload dataset")
    uploaded_file = st.file_uploader("Unggah dataset (CSV atau Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.session_state.df_original = df.copy()  # Simpan data asli
        st.session_state.df = df.copy()  # Salinan untuk diproses
        
        st.write("### 📑 Preview Data")
        st.dataframe(df)
        
        st.success("✅ Data berhasil dimuat!")
        st.session_state.data_uploaded = True


elif selected == "Preprocessing Data":
    if not st.session_state.get("data_uploaded", False):
        st.warning("⚠️ Harap unggah dataset terlebih dahulu di menu 'Dataset'.")
    else:
        if "df" not in st.session_state:
            st.warning("⚠️ Harap unggah dataset terlebih dahulu di menu 'Dataset'.")
        else:
            df = st.session_state.df.copy()

            st.write("### 🛠️ Proses preprocessing data")
            
            # Hapus duplikasi
            duplicate_count = df.duplicated().sum()
            df = df.drop_duplicates()
            st.success(f"✅ {duplicate_count} duplikat ditemukan dan telah dihapus!" if duplicate_count > 0 else "✅ Tidak ada data duplikat!")

            # Cek missing values
            missing_values = df.isnull().sum()
            missing_df = missing_values[missing_values > 0].reset_index()
            missing_df.columns = ["Kolom", "Jumlah Missing"]
        
            if not missing_df.empty:
                st.write("🔍 **Cek Missing Value:**")
                st.dataframe(missing_df)

                # Imputasi data
                num_cols = df.select_dtypes(include=['number']).columns
                cat_cols = df.select_dtypes(include=['object']).columns
                df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.mean()))
                df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))
                
                st.success("✅ Missing values telah diimputasi!")
            else:
                st.success("✅ Tidak ada missing values!")

        st.session_state.df_preprocessed = df.copy()  # Simpan hasil preprocessing
        st.session_state.data_preprocessed = True

        st.write("### 📋 Data setelah preprocessing")
        st.dataframe(df)
        st.success("✅ Preprocessing selesai")


elif selected == "Exploratory Data Analysis (EDA)":
    if not st.session_state.get("data_preprocessed", False):
        st.warning("⚠️ Silakan lakukan preprocessing terlebih dahulu di menu 'Preprocessing'.")
    else:
        if "df_preprocessed" not in st.session_state:
            st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu di menu 'Preprocessing'.")
        else:
            df = st.session_state.df_preprocessed.copy()

            st.write("### 📋 Exploratory Data Analysis (EDA)")

            # 1. Statistika Deskriptif
            st.write("### 🧮 Statistika Deskriptif")
            st.dataframe(df.describe())

            # Hilangkan kolom 'nama' jika ada
            cols_to_visualize = df.columns[df.columns.str.lower() != 'nama']

            # 2. Histogram Variabel Numerik
            num_cols = df[cols_to_visualize].select_dtypes(include=['number']).columns
            st.write("### 📊 Histogram Variabel Numerik")
            for col in num_cols:
                fig = px.histogram(
                    df,
                    x=col,
                    nbins=30,
                    marginal="box",
                    opacity=0.7,
                    color_discrete_sequence=['blue'],
                    labels={col: col, 'count': 'Jumlah'}
                )
                fig.update_layout(
                    title={
                        'text': f"Histogram Distribusi {col}",
                        'y': 0.9,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': dict(size=20)
                    },
                    xaxis_title=col,
                    yaxis_title="Jumlah"
                )
                st.plotly_chart(fig)

            # 3. Korelasi antar fitur numerik
            if len(num_cols) > 1:
                st.write("### ♻️ Korelasi Fitur Numerik")
                corr = df[num_cols].corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    origin='lower',
                    title="Heatmap Korelasi"
                )
                fig_corr.update_layout(
                    title={
                        'text': "Heatmap Korelasi Fitur Numerik",
                        'y': 0.9,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': dict(size=20)
                    }
                )
                st.plotly_chart(fig_corr)

            # 4. Bar Chart Variabel Kategorik
            cat_cols = df[cols_to_visualize].select_dtypes(include=['object']).columns
            st.write("### 📊 Bar Chart Variabel Kategorik")
            for col in cat_cols:
                df_count = df[col].value_counts().reset_index()
                df_count.columns = [col, "Jumlah"]
                fig = px.bar(
                    df_count,
                    x=col,
                    y="Jumlah",
                    text="Jumlah",
                    labels={col: col, "Jumlah": "Jumlah"}
                )
                fig.update_layout(
                    title={
                        'text': f"Frekuensi Kategori {col}",
                        'y': 0.9,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': dict(size=20)
                    },
                    xaxis_title=col,
                    yaxis_title="Jumlah"
                )
                st.plotly_chart(fig)

            st.success("✅ Semua variabel berhasil divisualisasikan")


elif selected == "Pra-Pemodelan":
    if not st.session_state.get("data_preprocessed", False):
        st.warning("⚠️ Harap lakukan preprocessing data terlebih dahulu di menu 'Preprocessing'.")
    else:
        st.title("⚙️ Pra-pemodelan data")
        if "df_preprocessed" not in st.session_state:
            st.warning("⚠️ Harap lakukan preprocessing data terlebih dahulu di menu 'Preprocessing'.")
        else:
            df = st.session_state.df_preprocessed.copy()

            # **1. Drop Kolom yang Tidak Digunakan**
            st.write("### 🔢 Menghapus kolom yang tidak digunakan")
            used_columns = ["Pendapatan", "Jumlah Tanggungan", "Luas Rumah", "Kondisi Atap", "Kondisi Dinding", "Kondisi Lantai"]
            df = df[used_columns]
            st.info(f"Kolom yang digunakan: {', '.join(used_columns)}")

            # **2. Pisahkan Kolom Numerik dan Kategorikal**
            categorical_cols = ["Kondisi Atap", "Kondisi Dinding", "Kondisi Lantai"]
            numeric_cols = ["Pendapatan", "Jumlah Tanggungan", "Luas Rumah"]

            # **3. Encoding Kolom Kategorikal (Ordinal)**
            st.write("### 🔢 Encoding data kategorikal")
            categories = [
                ["Buruk", "Sedang", "Baik"],  # Kondisi Atap
                ["Buruk", "Sedang", "Baik"],  # Kondisi Dinding
                ["Lebih Rendah dari Jalanan", "Sama Rata dengan Jalanan", "Lebih Tinggi dari Jalanan"]  # Kondisi Lantai
            ]

            ordinal_encoder = OrdinalEncoder(categories=categories)
            df[categorical_cols] = ordinal_encoder.fit_transform(df[categorical_cols]).astype(int)
            st.success("✅ Encoding kategorikal selesai!")

            # **4. Scaling Kolom Numerik**
            st.write("### 🔢 Normalisasi data numerik")
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            st.success("✅ Normalisasi numerik selesai!")

            # **5. Simpan Hasil Pra-Pemodelan ke session_state**
            st.session_state.df_encoded = df.copy()
            st.session_state.data_encoded = True
            st.session_state.num_cols = numeric_cols
            st.session_state.cat_cols = categorical_cols

            # **6. Tampilkan Data yang Telah Diproses**
            st.write("### 📑 Data yang telah diproses")
            st.dataframe(df)

            st.success("✅ Pra-pemodelan selesai")


elif selected == "Clustering":
    if not st.session_state.get("data_encoded", False) or "df_encoded" not in st.session_state:
        st.warning("⚠️ Silakan lakukan pra-pemodelan terlebih dahulu di menu 'Pra-pemodelan'.")
    else:
        st.title("🧪 Clustering K-Prototype dengan optimasi pusat cluster menggunakan PSO")

        if 'df_encoded' not in st.session_state or 'num_cols' not in st.session_state or 'cat_cols' not in st.session_state:
            st.warning("⚠️ Data hasil pra-pemodelan belum tersedia! Silakan lakukan preprocessing terlebih dahulu.")
        else:
            df = st.session_state.df_encoded.copy()
            num_cols = st.session_state.num_cols
            cat_cols = st.session_state.cat_cols

            st.write("### 📋 Data yang telah diproses dan siap untuk clustering")
            st.dataframe(df)

            # Konversi data kategori ke string sebelum encoding
            df[cat_cols] = df[cat_cols].astype(str)

            # Encoding kategori dengan LabelEncoder
            label_encoders = {}
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

            data_combined = df.values
            categorical_indices = [df.columns.get_loc(col) for col in cat_cols]

            st.write("### 🔢 Pilih jumlah K klaster yang dibentuk")
            col1, col2, col3 = st.columns([3, 2, 3])
            with col2:
                if 'sse_values' not in st.session_state:
                    st.session_state.sse_values = None

                if st.button("🔍 Rekomendasi jumlah cluster (Elbow Method)"):
                    with st.spinner("Menghitung jumlah cluster optimal..."):
                        def optimal_k_prototypes(data, categorical_indices, max_k=10):
                            sse = []
                            for k in range(1, max_k + 1):
                                kproto = KPrototypes(n_clusters=k, init='Cao', random_state=42)
                                kproto.fit(data, categorical=categorical_indices)
                                sse.append(kproto.cost_)
                            return sse

                        st.session_state.sse_values = optimal_k_prototypes(data_combined, categorical_indices)
                        st.session_state.recommended_k1 = 3

            if st.session_state.sse_values is not None:
                fig, ax = plt.subplots()
                ax.plot(range(1, 11), st.session_state.sse_values, marker='o', linestyle='--', color='b')
                ax.set_xlabel('Jumlah Klaster (k)')
                ax.set_ylabel('SSE (Cost)')
                ax.set_title('Elbow Method untuk Menentukan k Optimal')
                ax.grid(True)
                st.pyplot(fig)
                st.success(f"📌 Berdasarkan Elbow Method, jumlah cluster yang direkomendasikan adalah {st.session_state.recommended_k1}")

            st.write("Masukkan jumlah klaster yang akan dibentuk dalam rentang 2-10.")
            n_clusters = st.slider("Jumlah Cluster", min_value=2, max_value=10, value=3, key="n_clusters_slider")

            if st.button("🚀 Lakukan clustering"):
                st.session_state.n_clusters = n_clusters
                st.session_state.label_encoders = label_encoders

                with st.spinner("Proses clustering sedang berlangsung..."):
                    def assign_cluster(data, centroids, num_cols, gamma=0.3):
                        num_distances = np.linalg.norm(data[:, :len(num_cols), np.newaxis] - centroids[:, :len(num_cols)].T, axis=1)
                        cat_distances = np.sum(data[:, len(num_cols):, np.newaxis] != centroids[:, len(num_cols):].T, axis=1)
                        distances = num_distances + gamma * cat_distances
                        return np.argmin(distances, axis=1)

                    start_time = time.time()
                    np.random.seed(42)
                    num_particles = 1360
                    num_iterations = 100
                    w, c1, c2 = 0.6, 2.0, 2.0

                    particles = np.random.rand(num_particles, n_clusters, data_combined.shape[1])
                    velocities = np.random.rand(num_particles, n_clusters, data_combined.shape[1]) * 0.1
                    pbest = particles.copy()
                    gbest = min(particles, key=lambda centroids: np.sum(np.min(np.linalg.norm(data_combined[:, np.newaxis, :] - centroids, axis=2), axis=1)))
                    last_fitness = float('inf')

                    for iteration in range(num_iterations):
                        fitness_results = np.array([np.sum(np.min(np.linalg.norm(data_combined[:, np.newaxis, :] - p, axis=2), axis=1)) for p in particles])
                        better_pbest_mask = fitness_results < np.array([np.sum(np.min(np.linalg.norm(data_combined[:, np.newaxis, :] - p, axis=2), axis=1)) for p in pbest])
                        pbest[better_pbest_mask] = particles[better_pbest_mask]
                        gbest = pbest[np.argmin(fitness_results)]

                        r1, r2 = np.random.rand(), np.random.rand()
                        velocities = w * velocities + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest - particles)
                        particles += velocities

                        if abs(last_fitness - np.min(fitness_results)) < 0.01:
                            break
                        last_fitness = np.min(fitness_results)

                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    labels = assign_cluster(data_combined, gbest, num_cols)
                    df['Cluster'] = labels

                    silhouette_avg = silhouette_score(df[num_cols], labels)
                    dbi_score = davies_bouldin_score(df[num_cols], labels)

                    st.session_state.df_clustered = df.copy()
                    st.session_state.gbest = gbest
                    st.session_state.clustering_done = True
                    st.session_state.elapsed_time = elapsed_time
                    st.session_state.silhouette_score = silhouette_avg
                    st.session_state.dbi_score = dbi_score

                st.success("✅ Clustering selesai!")

            # Menampilkan hasil clustering jika sudah dilakukan
            if st.session_state.get("clustering_done", False):
                df = st.session_state.df_clustered.copy()
                gbest = st.session_state.gbest
                n_clusters = st.session_state.n_clusters
                label_encoders = st.session_state.label_encoders
                elapsed_time = st.session_state.elapsed_time
                silhouette_avg = st.session_state.silhouette_score
                dbi_score = st.session_state.dbi_score

                st.markdown(f"""
                    ### 📈 Evaluasi Model Clustering
                    - 🧭 **Silhouette Score**: `{silhouette_avg:.4f}` (semakin tinggi semakin baik)
                    - 📏 **Davies-Bouldin Index (DBI)**: `{dbi_score:.4f}` (semakin rendah semakin baik)
                    - ⏱️ **Waktu Komputasi**: `{elapsed_time:.2f}` detik
                """)

                # Visualisasi t-SNE
                columns_for_tsne = num_cols + cat_cols
                data_for_tsne = df[columns_for_tsne]
                tsne_model = TSNE(n_components=2, perplexity=30, random_state=42)
                data_tsne = tsne_model.fit_transform(data_for_tsne)
                data_tsne_shifted = data_tsne - np.min(data_tsne, axis=0)

                data_viz = pd.DataFrame(data_tsne_shifted, columns=['TSNE1', 'TSNE2'])
                data_viz['Cluster'] = df['Cluster']
                palette = sns.color_palette('tab10', n_colors=n_clusters)

                tree = KDTree(data_for_tsne)
                _, nearest_indices = tree.query(gbest)
                centroids_tsne = data_tsne[nearest_indices]
                centroids_tsne_shifted = centroids_tsne - np.min(data_tsne, axis=0)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x='TSNE1', y='TSNE2', hue=data_viz['Cluster'], palette=palette, data=data_viz, alpha=0.7, ax=ax)
                for i, centroid in enumerate(centroids_tsne_shifted):
                    ax.scatter(centroid[0], centroid[1], marker='X', s=200, color=palette[i], edgecolor='black', label=f'Centroid {i}')
                ax.set_title('Visualisasi Hasil Clustering dengan t-SNE')
                ax.set_xlabel('TSNE1')
                ax.set_ylabel('TSNE2')
                ax.grid(True)  # ✅ Tambahkan grid
                ax.legend(title='Cluster')
                st.pyplot(fig)

                # Interpretasi hasil clustering
                cluster_means = df.groupby('Cluster')[num_cols].mean().reset_index()
                cluster_scores = cluster_means.copy()
                cluster_scores['Skor Numerik'] = cluster_scores[num_cols].mean(axis=1)

                original_df = df.copy()
                for col in cat_cols:
                    if col in label_encoders:
                        le = label_encoders[col]
                        original_df[col] = le.inverse_transform(original_df[col])

                df_interpretasi = cluster_scores[['Cluster', 'Skor Numerik']].copy()
                df_interpretasi['Prioritas'] = df_interpretasi['Skor Numerik'].rank(ascending=True).astype(int)

                def kategorik_dari_skor(skor):
                    if skor >= df_interpretasi['Skor Numerik'].quantile(2/3):
                        return "Baik"
                    elif skor >= df_interpretasi['Skor Numerik'].quantile(1/3):
                        return "Sedang"
                    else:
                        return "Buruk"

                df_interpretasi['Fitur Kategorik'] = df_interpretasi['Skor Numerik'].apply(kategorik_dari_skor)

                df_tabel_interpretasi = pd.DataFrame({
                    "Cluster": df_interpretasi['Cluster'],
                    "Fitur Numerik": pd.cut(df_interpretasi['Skor Numerik'], bins=3, labels=["Rendah", "Sedang", "Tinggi"]),
                    "Fitur Kategorik": df_interpretasi['Fitur Kategorik'],
                    "Prioritas": df_interpretasi['Prioritas']
                })

                st.write("### 📋 Tabel interpretasi tiap cluster")
                st.dataframe(df_tabel_interpretasi)

                st.markdown("""
                - Semakin **rendah** rata-rata skor numerik suatu cluster, maka cluster tersebut memiliki kondisi rumah yang **lebih buruk** dan **lebih diprioritaskan**.
                - Semakin **tinggi** skor numerik, maka fitur kategorik cenderung **baik**, dan cluster tersebut **kurang diprioritaskan**.
                """)

                cluster_counts = df['Cluster'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Jumlah Anggota']
                st.write("### 📋 Jumlah anggota tiap cluster")
                st.table(cluster_counts)


elif selected == "Hasil":
    if not st.session_state.get("clustering_done", False):
        st.warning("⚠️ Belum ada hasil clustering. Silakan lakukan clustering terlebih dahulu di menu 'Clustering'.")
    else:
        st.title("📄 Hasil clustering PSO")

        # ✅ **Perbaikan: Gunakan session_state.get() untuk menghindari error jika hasil clustering belum ada**
        df_clustered = st.session_state.get("df_clustered", None)
        
        # Pastikan hasil clustering tersedia
        if 'df_clustered' in st.session_state and 'df_preprocessed' in st.session_state:
            st.write("### 📋 Tabel hasil clustering")

            # Memuat data hasil preprocessing
            df_preprocessed = st.session_state.df_preprocessed.copy()

            # Menambahkan hasil clustering
            df_clustered = st.session_state.df_clustered.copy()
            df_preprocessed["Cluster"] = df_clustered["Cluster"].values  

            # Simpan hasil clustering ke session_state
            st.session_state.df_result = df_preprocessed.copy()

            # Menampilkan tabel hasil clustering
            st.dataframe(df_preprocessed)

            # Menampilkan ringkasan jumlah anggota tiap cluster
            st.write("### 📋 Ringkasan jumlah anggota tiap cluster")
            cluster_counts = df_preprocessed["Cluster"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Jumlah Anggota"]
            st.table(cluster_counts)

            # **Menyiapkan data untuk diunduh**
            csv = df_preprocessed.to_csv(index=False).encode('utf-8')

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_preprocessed.to_excel(writer, index=False, sheet_name='Hasil Clustering')
                writer.close()
            excel_data = output.getvalue()

            # **Tombol Download**
            st.write("### 📥 Unduh hasil clustering")
            st.download_button(label="📥 Download sebagai CSV", data=csv, file_name="hasil_clustering.csv", mime='text/csv')
            st.download_button(label="📥 Download sebagai Excel", data=excel_data, file_name="hasil_clustering.xlsx", 
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        else:
            st.warning("⚠️ Belum ada hasil clustering. Silakan lakukan clustering terlebih dahulu.")


elif selected == "Prediksi":
    if not st.session_state.get("data_encoded", False) or not st.session_state.get("clustering_done", False):
        st.warning("⚠️ Lakukan pra-pemodelan dan clustering terlebih dahulu.")
    else:
        st.title("🧠 Prediksi klaster data baru")

        if "df_encoded" not in st.session_state or "n_clusters" not in st.session_state:
            st.warning("⚠️ Data training atau jumlah cluster belum tersedia. Silakan lakukan pra-pemodelan dan clustering terlebih dahulu.")
        else:
            df_encoded = st.session_state.df_encoded.copy()
            num_cols = st.session_state.num_cols
            cat_cols = st.session_state.cat_cols
            n_clusters_prediksi = st.session_state.n_clusters

            if "manual_data_list" not in st.session_state:
                st.session_state.manual_data_list = []

            input_method = st.radio("🕹️ Pilih metode input data:", ["Unggah File", "Input Manual"], horizontal=True)

            new_data = None

            if input_method == "Unggah File":
                uploaded_file = st.file_uploader("📤 Unggah data baru (CSV/XLSX)", type=["csv", "xlsx"])
                if uploaded_file is not None:
                    ext = uploaded_file.name.split('.')[-1]
                    try:
                        if ext == "csv":
                            new_data = pd.read_csv(uploaded_file)
                        elif ext == "xlsx":
                            new_data = pd.read_excel(uploaded_file)
                    except Exception as e:
                        st.error(f"Gagal membaca file: {e}")
            else:
                st.write("### 📝 Input Data Manual")
                with st.form("manual_input_form"):
                    nama = st.text_input("Nama")
                    kecamatan = st.selectbox("Kecamatan", [
                        "Asemrowo", "Benowo", "Bubutan", "Bulak", "Dukuh Pakis", "Gayungan", "Genteng", "Gubeng", "Gunung Anyar", "Jambangan",
                        "Karang Pilang", "Kenjeran", "Krembangan", "Lakarsantri", "Mulyorejo", "Pabean Cantian", "Pakal", "Rungkut", "Sambikerep", "Sawahan",
                        "Semampir", "Simokerto", "Sukolilo", "Sukomanunggal", "Tambaksari", "Tandes", "Tegalsari", "Tenggilis Mejoyo", "Wiyung", "Wonocolo", "Wonokromo"
                    ])
                    pendapatan = st.number_input("Pendapatan", min_value=0, step=100000)
                    tanggungan = st.number_input("Jumlah Tanggungan", min_value=0, step=1)
                    luas_rumah = st.number_input("Luas Rumah (m²)", min_value=0, step=1)
                    kondisi_atap = st.selectbox("Kondisi Atap", ["Baik", "Sedang", "Buruk"])
                    kondisi_dinding = st.selectbox("Kondisi Dinding", ["Baik", "Sedang", "Buruk"])
                    kondisi_lantai = st.selectbox("Kondisi Lantai", [
                        "Lebih Rendah dari Jalanan", "Sama Rata dengan Jalanan", "Lebih Tinggi dari Jalanan"
                    ])
                    submit = st.form_submit_button("💾 Simpan data")

                    if submit:
                        if nama and pendapatan and luas_rumah:
                            new_entry = {
                                "Nama": nama,
                                "Kecamatan": kecamatan,
                                "Pendapatan": pendapatan,
                                "Jumlah Tanggungan": tanggungan,
                                "Luas Rumah": luas_rumah,
                                "Kondisi Atap": kondisi_atap,
                                "Kondisi Dinding": kondisi_dinding,
                                "Kondisi Lantai": kondisi_lantai
                            }
                            st.session_state.manual_data_list.append(new_entry)
                            st.success("✅ Data berhasil disimpan!")
                        else:
                            st.warning("⚠️ Pastikan semua kolom wajib diisi.")

                if st.session_state.manual_data_list:
                    st.write("### 📋 Data manual yang tersimpan")
                    df_manual = pd.DataFrame(st.session_state.manual_data_list)
                    st.dataframe(df_manual)
                    new_data = df_manual.copy()

            if new_data is not None:
                st.write("### 📋 Data yang akan diproses")
                st.dataframe(new_data)

                if st.button("🚀 Lakukan clustering data baru"):
                    gbest = st.session_state.get("gbest", None)

                    if gbest is None:
                        st.error("❌ Data centroid (gbest) belum tersedia. Silakan lakukan proses clustering terlebih dahulu.")
                    else:
                        with st.spinner("🔄 Memproses data baru..."):
                            try:
                                data_pred = new_data[["Pendapatan", "Jumlah Tanggungan", "Luas Rumah",
                                                    "Kondisi Atap", "Kondisi Dinding", "Kondisi Lantai"]].copy()

                                categories = [
                                    ["Buruk", "Sedang", "Baik"],
                                    ["Buruk", "Sedang", "Baik"],
                                    ["Lebih Rendah dari Jalanan", "Sama Rata dengan Jalanan", "Lebih Tinggi dari Jalanan"]
                                ]
                                ordinal_encoder = OrdinalEncoder(categories=categories)
                                data_pred[cat_cols] = ordinal_encoder.fit_transform(data_pred[cat_cols]).astype(int)

                                scaler = StandardScaler()
                                data_pred[num_cols] = scaler.fit_transform(data_pred[num_cols])

                                data_array = data_pred[num_cols + cat_cols].values

                                def assign_cluster(data, centroids, num_cols, gamma=0.3):
                                    num_distances = np.linalg.norm(data[:, :len(num_cols), np.newaxis] - centroids[:, :len(num_cols)].T, axis=1)
                                    cat_distances = np.sum(data[:, len(num_cols):, np.newaxis] != centroids[:, len(num_cols):].T, axis=1)
                                    distances = num_distances + gamma * cat_distances
                                    return np.argmin(distances, axis=1)

                                predicted_clusters = assign_cluster(data_array, gbest, num_cols)
                                new_data["Prediksi Cluster"] = predicted_clusters

                                st.session_state.hasil_prediksi = new_data.copy()
                                st.success("✅ Clustering selesai! Data baru berhasil dikelompokkan.")
                            except Exception as e:
                                st.error(f"❌ Terjadi kesalahan saat proses clustering: {e}")

            if "hasil_prediksi" in st.session_state:
                st.write("### 🧾 Hasil clustering data baru")
                st.dataframe(st.session_state.hasil_prediksi)

                cluster_counts = st.session_state.hasil_prediksi["Prediksi Cluster"].value_counts().reset_index()
                cluster_counts.columns = ["Cluster", "Jumlah Anggota"]
                st.write("### 📋 Jumlah anggota tiap cluster pada data baru")
                st.table(cluster_counts)

                # Hitung rata-rata skor numerik per cluster
                numeric_means = st.session_state.hasil_prediksi.groupby("Prediksi Cluster")[["Pendapatan", "Jumlah Tanggungan", "Luas Rumah"]].mean()
                numeric_means["Rata-rata Skor"] = numeric_means.mean(axis=1)

                # Urutkan cluster berdasarkan skor rata-rata numerik (dari rendah ke tinggi)
                urutan_cluster = numeric_means["Rata-rata Skor"].sort_values().index.tolist()

                # Prioritas hanya berdasarkan urutan skor rata-rata numerik
                label_prioritas = ["Tinggi", "Sedang", "Rendah"][:len(urutan_cluster)]

                # Buat tabel interpretasi hanya dengan "Cluster" dan "Prioritas"
                interpretasi_df = pd.DataFrame({
                    "Cluster": urutan_cluster,
                    "Prioritas": label_prioritas
                }).sort_values("Cluster").reset_index(drop=True)

                st.write("### 📋 Tabel interpretasi tiap cluster")
                st.dataframe(interpretasi_df)

                st.markdown("""  
                - Cluster dengan prioritas **Tinggi** adalah yang memiliki kondisi rumah yang lebih buruk berdasarkan skor numerik yang lebih rendah.
                - Cluster dengan prioritas **Rendah** adalah yang memiliki kondisi rumah yang lebih baik berdasarkan skor numerik yang lebih tinggi.
                """)

                # Tombol download
                csv = st.session_state.hasil_prediksi.to_csv(index=False).encode('utf-8')
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state.hasil_prediksi.to_excel(writer, index=False, sheet_name='Data Baru')
                    writer.close()
                excel_data = output.getvalue()

                st.write("### 📥 Unduh hasil clustering data baru")
                st.download_button("📥 Download sebagai CSV", data=csv, file_name="hasil_prediksi_data_baru.csv", mime='text/csv')
                st.download_button("📥 Download sebagai Excel", data=excel_data, file_name="hasil_prediksi_data_baru.xlsx",
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

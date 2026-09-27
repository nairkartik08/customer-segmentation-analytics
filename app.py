import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/customer_segmentation_results.csv"
    )


@st.cache_resource
def load_models():

    kmeans = joblib.load(
        "models/kmeans_model.pkl"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    return kmeans, scaler


df = load_data()
kmeans, scaler = load_models()

# ============================================================
# ML ANALYSIS DATA
# ============================================================

X = df[
    ['Annual Income (k$)', 'Spending Score (1-100)']
]

X_scaled = scaler.transform(X)


# Elbow Method
inertia_values = []

K_range = range(2, 11)

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia_values.append(
        model.inertia_
    )


# Silhouette Analysis
silhouette_values = []

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_values.append(score)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ==============================
       MAIN APPLICATION
       ============================== */

    .main {
        background-color: #f6f8fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }


    /* ==============================
       SIDEBAR
       ============================== */

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    section[data-testid="stSidebar"] .stRadio label {
        padding: 8px 5px;
        border-radius: 8px;
    }


    /* ==============================
       DASHBOARD TITLE
       ============================== */

    .dashboard-title {
        font-size: 42px;
        font-weight: 750;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }

    .dashboard-subtitle {
        font-size: 17px;
        color: #667085;
        margin-bottom: 30px;
    }


    /* ==============================
       SECTION HEADINGS
       ============================== */

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }


    /* ==============================
       METRIC CARDS
       ============================== */

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e4e7ec;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }


    /* ==============================
       BUTTONS
       ============================== */

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
    }

    .stDownloadButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
    }


    /* ==============================
       DATAFRAME
       ============================== */

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e4e7ec;
    }


    /* ==============================
       INPUT ELEMENTS
       ============================== */

    div[data-baseweb="select"] > div {
        border-radius: 9px;
    }

    div[data-baseweb="input"] > div {
        border-radius: 9px;
    }


    /* ==============================
       ALERT / INFO BOXES
       ============================== */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }


    /* ==============================
       DIVIDERS
       ============================== */

    hr {
        margin-top: 25px;
        margin-bottom: 25px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Customer Profiles",
        "Customer Predictor",
        "Algorithm Analysis",
        "Customer Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Customer segmentation using "
    "K-Means and Hierarchical Clustering."
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="dashboard-title">'
        '📊 Customer Segmentation Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Machine Learning based customer behavior analysis using '
        'K-Means and Hierarchical Clustering'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "📌 Dataset: Mall Customer Dataset  |  "
        "Features: Annual Income & Spending Score  |  "
        "Clustering: K-Means + Hierarchical"
)

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    total_customers = len(df)

    avg_income = df[
        'Annual Income (k$)'
    ].mean()

    avg_spending = df[
        'Spending Score (1-100)'
    ].mean()

    total_segments = df[
        'KMeans_Cluster'
    ].nunique()


    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👥 Total Customers",
            total_customers
        )

    with col2:

        st.metric(
            "💰 Avg Income",
            f"${avg_income:.2f}k"
        )

    with col3:

        st.metric(
            "🛒 Avg Spending",
            f"{avg_spending:.2f}"
        )

    with col4:

        st.metric(
            "🎯 Segments",
            total_segments
        )


    st.divider()

    # --------------------------------------------------------
    # ELBOW AND SILHOUETTE ANALYSIS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Clustering Model Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)


    # Elbow Curve
    with col1:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            list(K_range),
            inertia_values,
            marker='o'
        )

        ax.set_title(
            'Elbow Method'
        )

        ax.set_xlabel(
            'Number of Clusters (K)'
        )

        ax.set_ylabel(
            'Inertia'
        )

        ax.grid(True)

        plt.tight_layout()

        st.pyplot(fig)


    # Silhouette Curve
    with col2:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            list(K_range),
            silhouette_values,
            marker='o'
        )

        ax.set_title(
            'Silhouette Analysis'
        )

        ax.set_xlabel(
            'Number of Clusters (K)'
        )

        ax.set_ylabel(
            'Silhouette Score'
        )

        ax.grid(True)

        plt.tight_layout()

        st.pyplot(fig)

        st.divider()

    st.subheader("🤖 Machine Learning Summary")

    best_silhouette_index = np.argmax(
        silhouette_values
    )

    best_silhouette_k = list(K_range)[
        best_silhouette_index
    ]

    best_silhouette_score = max(
        silhouette_values
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Selected Clusters",
            "5"
        )


    with col2:

        st.metric(
            "Best Silhouette K",
            best_silhouette_k
        )


    with col3:

        st.metric(
            "Best Silhouette Score",
            f"{best_silhouette_score:.3f}"
        )

        st.divider()

    st.subheader("💡 Business Insights")

    st.markdown("""
    **High-Value Customers**

    Customers with relatively high income and high spending scores.
    They represent a potentially valuable customer segment.

    **Young High-Spenders**

    Younger customers with relatively lower income but high spending
    scores. This group demonstrates strong spending activity despite
    lower average income.

    **High-Income Low-Spenders**

    Customers with high average income but comparatively low spending
    scores. This group may represent an opportunity for increased
    customer engagement.

    **Low-Income Low-Spenders**

    Customers with relatively low income and low spending scores.

    **Average Customers**

    The largest segment in this dataset, showing moderate income and
    spending behavior.
    """)

    # --------------------------------------------------------
    # CLUSTER VISUALIZATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Customer Segmentation'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    # Scatter plot
    with col1:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.scatterplot(
            data=df,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            hue='Customer_Segment',
            s=80,
            ax=ax
        )

        ax.set_title(
            'Income vs Spending'
        )

        ax.set_xlabel(
            'Annual Income (k$)'
        )

        ax.set_ylabel(
            'Spending Score'
        )

        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            fontsize=8
        )

        plt.tight_layout()

        st.pyplot(fig)


    # Segment distribution
    with col2:

        segment_counts = (
            df['Customer_Segment']
            .value_counts()
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.barplot(
            x=segment_counts.index,
            y=segment_counts.values,
            ax=ax
        )

        ax.set_title(
            'Customers by Segment'
        )

        ax.set_xlabel(
            'Customer Segment'
        )

        ax.set_ylabel(
            'Number of Customers'
        )

        ax.tick_params(
            axis='x',
            rotation=25
        )

        plt.tight_layout()

        st.pyplot(fig)


    st.divider()


    # --------------------------------------------------------
    # QUICK CUSTOMER PROFILE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Customer Segment Overview'
        '</div>',
        unsafe_allow_html=True
    )


    profile = df.groupby(
        'Customer_Segment'
    ).agg(
        Customers=('CustomerID', 'count'),
        Average_Age=('Age', 'mean'),
        Average_Income=(
            'Annual Income (k$)',
            'mean'
        ),
        Average_Spending=(
            'Spending Score (1-100)',
            'mean'
        )
    ).round(2)


    st.dataframe(
        profile,
        use_container_width=True
    )


# ============================================================
# CUSTOMER PROFILES
# ============================================================

elif page == "Customer Profiles":

    st.title("👥 Customer Profiles")

    st.write(
        "Explore the characteristics and behavior of each customer segment."
    )

    # --------------------------------------------------------
    # SEGMENT PROFILE
    # --------------------------------------------------------

    profile = df.groupby(
        'Customer_Segment'
    ).agg(
        Customers=('CustomerID', 'count'),
        Average_Age=('Age', 'mean'),
        Average_Income=('Annual Income (k$)', 'mean'),
        Average_Spending=('Spending Score (1-100)', 'mean')
    ).round(2)

    selected_segment = st.selectbox(
        "Select a Customer Segment",
        profile.index
    )

    segment_data = profile.loc[selected_segment]

    st.divider()

    st.subheader(f"📌 {selected_segment}")

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 Customers",
            int(segment_data['Customers'])
        )

    with col2:
        st.metric(
            "🎂 Average Age",
            f"{segment_data['Average_Age']:.1f}"
        )

    with col3:
        st.metric(
            "💰 Average Income",
            f"${segment_data['Average_Income']:.1f}k"
        )

    with col4:
        st.metric(
            "🛍️ Average Spending",
            f"{segment_data['Average_Spending']:.1f}"
        )

    # --------------------------------------------------------
    # GET SELECTED CUSTOMERS
    # --------------------------------------------------------

    segment_customers = df[
        df['Customer_Segment'] == selected_segment
    ]

    st.divider()

    # --------------------------------------------------------
    # VISUAL ANALYSIS
    # --------------------------------------------------------

    st.subheader("📊 Segment Visual Analysis")

    col1, col2 = st.columns(2)

    # Income vs Spending
    with col1:

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.scatterplot(
            data=df,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            hue='Customer_Segment',
            alpha=0.25,
            s=60,
            legend=False,
            ax=ax
        )

        sns.scatterplot(
            data=segment_customers,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            s=100,
            ax=ax
        )

        ax.set_title(
            f"Income vs Spending — {selected_segment}"
        )

        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score")

        plt.tight_layout()

        st.pyplot(fig)

    # Age Distribution
    with col2:

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.histplot(
            segment_customers['Age'],
            bins=10,
            kde=True,
            ax=ax
        )

        ax.set_title(
            f"Age Distribution — {selected_segment}"
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Customers")

        plt.tight_layout()

        st.pyplot(fig)

    # --------------------------------------------------------
    # GENDER DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("👤 Gender Distribution")

    gender_counts = (
        segment_customers['Gender']
        .value_counts()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        x=gender_counts.index,
        y=gender_counts.values,
        ax=ax
    )

    ax.set_title(
        f"Gender Distribution — {selected_segment}"
    )

    ax.set_xlabel("Gender")
    ax.set_ylabel("Number of Customers")

    plt.tight_layout()

    st.pyplot(fig)

    # --------------------------------------------------------
    # SEGMENT INTERPRETATION
    # --------------------------------------------------------

    st.subheader("💡 Segment Interpretation")

    interpretations = {
        "High-Value Customers":
            "Customers with high income and high spending scores. "
            "This segment shows strong purchasing activity.",

        "Young High-Spenders":
            "Relatively young customers with high spending scores "
            "despite having lower average income.",

        "High-Income Low-Spenders":
            "Customers with high income but relatively low spending "
            "scores.",

        "Low-Income Low-Spenders":
            "Customers with relatively low income and low spending "
            "scores.",

        "Average Customers":
            "Customers showing moderate income and spending behavior."
    }

    st.info(
        interpretations.get(
            selected_segment,
            "This segment represents a distinct customer group "
            "identified by the clustering model."
        )
    )

    # --------------------------------------------------------
    # CUSTOMER DETAILS
    # --------------------------------------------------------

    st.subheader("📋 Customers in This Segment")

    st.dataframe(
        segment_customers[
            [
                'CustomerID',
                'Gender',
                'Age',
                'Annual Income (k$)',
                'Spending Score (1-100)'
            ]
        ],
        use_container_width=True
    )


# ============================================================
# CUSTOMER PREDICTOR
# ============================================================

elif page == "Customer Predictor":

    st.title("🔍 Customer Segment Predictor")

    st.write(
        "Enter a customer's annual income and spending score "
        "to predict their customer segment."
    )

    st.divider()

    # --------------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------------

    st.subheader("📝 Customer Information")

    col1, col2 = st.columns(2)

    with col1:

        income = st.number_input(
            "Annual Income (k$)",
            min_value=0.0,
            max_value=200.0,
            value=60.0,
            step=1.0
        )

    with col2:

        spending = st.number_input(
            "Spending Score (1-100)",
            min_value=1.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🎯 Predict Customer Segment",
        use_container_width=True
    ):

        # Prepare input
        input_data = np.array(
            [[income, spending]]
        )

        # Scale input using trained scaler
        input_scaled = scaler.transform(
            input_data
        )

        # Predict cluster
        prediction = kmeans.predict(
            input_scaled
        )[0]

        # Cluster names
        segment_name = {
            0: 'Average Customers',
            1: 'High-Value Customers',
            2: 'Young High-Spenders',
            3: 'High-Income Low-Spenders',
            4: 'Low-Income Low-Spenders'
        }

        predicted_segment = segment_name[
            prediction
        ]

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.success(
            f"🎯 Predicted Segment: **{predicted_segment}**"
        )

        st.divider()

        # ----------------------------------------------------
        # INPUT METRICS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "💰 Annual Income",
                f"${income:.1f}k"
            )

        with col2:

            st.metric(
                "🛍️ Spending Score",
                f"{spending:.1f}"
            )

        with col3:

            st.metric(
                "🎯 Cluster",
                f"Cluster {prediction}"
            )

        st.divider()

        # ----------------------------------------------------
        # PREDICTION VISUALIZATION
        # ----------------------------------------------------

        st.subheader("📊 Customer Position in Segmentation")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Existing customers
        sns.scatterplot(
            data=df,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            hue='Customer_Segment',
            s=70,
            alpha=0.55,
            ax=ax
        )

        # Predicted customer
        ax.scatter(
            income,
            spending,
            s=300,
            marker='*',
            edgecolors='black',
            linewidths=2,
            label='New Customer'
        )

        ax.annotate(
            'New Customer',
            (income, spending),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold'
        )

        ax.set_title(
            'Predicted Customer on Segmentation Map'
        )

        ax.set_xlabel(
            'Annual Income (k$)'
        )

        ax.set_ylabel(
            'Spending Score (1-100)'
        )

        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left'
        )

        ax.grid(alpha=0.2)

        plt.tight_layout()

        st.pyplot(fig)

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        st.subheader("💡 Prediction Interpretation")

        interpretations = {

            "High-Value Customers":
                "The customer has relatively high income and "
                "high spending behavior.",

            "Young High-Spenders":
                "The customer's spending score is relatively high "
                "despite a lower income level.",

            "High-Income Low-Spenders":
                "The customer has relatively high income but "
                "comparatively low spending behavior.",

            "Low-Income Low-Spenders":
                "The customer has relatively low income and "
                "low spending behavior.",

            "Average Customers":
                "The customer shows moderate income and "
                "spending behavior."
        }

        st.info(
            interpretations.get(
                predicted_segment,
                "The customer belongs to a segment identified "
                "by the K-Means clustering model."
            )
        )


# ============================================================
# ALGORITHM ANALYSIS
# ============================================================

elif page == "Algorithm Analysis":

    st.title("📈 Algorithm Analysis")

    st.write(
        "Comparison of K-Means and Hierarchical Clustering "
        "using clustering evaluation techniques."
    )

    # --------------------------------------------------------
    # ALGORITHM SCORES
    # --------------------------------------------------------

    kmeans_score = 0.554657
    hierarchical_score = 0.553809

    comparison = pd.DataFrame({
        'Algorithm': [
            'K-Means',
            'Hierarchical Clustering'
        ],
        'Silhouette Score': [
            kmeans_score,
            hierarchical_score
        ]
    })

    # --------------------------------------------------------
    # SCORE CARDS
    # --------------------------------------------------------

    st.subheader("📊 Clustering Performance")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "K-Means Silhouette Score",
            f"{kmeans_score:.4f}"
        )

    with col2:

        st.metric(
            "Hierarchical Silhouette Score",
            f"{hierarchical_score:.4f}"
        )

    st.divider()

    # --------------------------------------------------------
    # SILHOUETTE COMPARISON
    # --------------------------------------------------------

    st.subheader("📊 Silhouette Score Comparison")

    col1, col2 = st.columns(2)

    with col1:

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

    with col2:

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        sns.barplot(
            data=comparison,
            x='Algorithm',
            y='Silhouette Score',
            ax=ax
        )

        ax.set_ylim(
            0,
            0.7
        )

        ax.set_title(
            "Algorithm Silhouette Comparison"
        )

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Silhouette Score")

        ax.tick_params(
            axis='x',
            rotation=15
        )

        plt.tight_layout()

        st.pyplot(fig)

    st.divider()

    # --------------------------------------------------------
    # ELBOW METHOD
    # --------------------------------------------------------

    st.subheader("📉 Elbow Method")

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        list(K_range),
        inertia_values,
        marker='o'
    )

    # Highlight selected K
    selected_k = 5

    selected_index = list(K_range).index(
        selected_k
    )

    ax.scatter(
        selected_k,
        inertia_values[selected_index],
        s=180,
        marker='*',
        edgecolors='black',
        linewidths=1.5,
        label='Selected K = 5'
    )

    ax.set_title(
        "Elbow Method for Selecting Number of Clusters"
    )

    ax.set_xlabel(
        "Number of Clusters (K)"
    )

    ax.set_ylabel(
        "Inertia"
    )

    ax.set_xticks(
        list(K_range)
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        "The clustering experiment uses K = 5 as the selected "
        "number of customer segments."
    )

    st.divider()

    # --------------------------------------------------------
    # HIERARCHICAL CLUSTERING DENDROGRAM
    # --------------------------------------------------------

    st.subheader("🌳 Hierarchical Clustering Dendrogram")

    st.write(
        "The dendrogram illustrates the hierarchical merging "
        "of customers based on their scaled income and "
        "spending behavior."
    )

    linkage_matrix = linkage(
        X_scaled,
        method='ward'
    )

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode='lastp',
        p=30,
        leaf_rotation=90,
        leaf_font_size=8
    )

    ax.set_title(
        "Hierarchical Clustering Dendrogram"
    )

    ax.set_xlabel(
        "Customer Groups"
    )

    ax.set_ylabel(
        "Distance"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        "Ward linkage combines groups while minimizing the "
        "increase in within-cluster variance. The dendrogram "
        "helps visualize how customer groups are progressively merged."
    )

    st.divider()

    # --------------------------------------------------------
    # ALGORITHM EXPLANATION
    # --------------------------------------------------------

    st.subheader("🧠 Algorithm Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            ### 🔵 K-Means Clustering

            - Divides customers into a predefined number of clusters.
            - Uses distance between data points and cluster centroids.
            - K = 5 was used for the final segmentation.
            - Suitable for identifying distinct customer groups.
            """
        )

    with col2:

        st.markdown(
            """
            ### 🌳 Hierarchical Clustering

            - Builds a hierarchy of customer groups.
            - Starts with individual observations.
            - Similar groups are progressively merged.
            - A dendrogram visualizes the merging process.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # FINAL ANALYSIS
    # --------------------------------------------------------

    st.subheader("📌 Analysis Summary")

    st.markdown(
        f"""
        **K-Means Silhouette Score:** `{kmeans_score:.4f}`

        **Hierarchical Clustering Silhouette Score:** `{hierarchical_score:.4f}`

        Both methods produced similar silhouette scores for this
        dataset. The final customer segmentation uses **5 clusters**
        based on the clustering experiment and the selected K value.

        The Elbow Method provides an additional view of clustering
        structure, while the dendrogram shows the hierarchical
        relationships between customer groups.
        """
    )


# ============================================================
# CUSTOMER EXPLORER
# ============================================================

elif page == "Customer Explorer":

    st.title("🔎 Customer Explorer")

    st.write(
        "Search and filter customers based on their "
        "income, spending behavior, and customer segment."
    )

    st.divider()

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.subheader("🎛️ Customer Filters")

    col1, col2, col3 = st.columns(3)

    # Segment filter
    with col1:

        segment_filter = st.multiselect(
            "Customer Segment",
            options=sorted(
                df['Customer_Segment'].unique()
            ),
            default=sorted(
                df['Customer_Segment'].unique()
            )
        )

    # Income filter
    with col2:

        min_income = float(
            df['Annual Income (k$)'].min()
        )

        max_income = float(
            df['Annual Income (k$)'].max()
        )

        income_range = st.slider(
            "Annual Income (k$)",
            min_value=min_income,
            max_value=max_income,
            value=(min_income, max_income),
            step=1.0
        )

    # Spending filter
    with col3:

        min_spending = float(
            df['Spending Score (1-100)'].min()
        )

        max_spending = float(
            df['Spending Score (1-100)'].max()
        )

        spending_range = st.slider(
            "Spending Score",
            min_value=min_spending,
            max_value=max_spending,
            value=(min_spending, max_spending),
            step=1.0
        )

    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    filtered_df = df[
        (df['Customer_Segment'].isin(segment_filter))
        &
        (df['Annual Income (k$)'] >= income_range[0])
        &
        (df['Annual Income (k$)'] <= income_range[1])
        &
        (df['Spending Score (1-100)'] >= spending_range[0])
        &
        (df['Spending Score (1-100)'] <= spending_range[1])
    ]

    # --------------------------------------------------------
    # FILTERED CUSTOMER COUNT
    # --------------------------------------------------------

    st.divider()

    st.subheader("📊 Filter Results")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "👥 Customers Found",
            len(filtered_df)
        )

    with col2:

        if len(filtered_df) > 0:
            avg_filtered_income = filtered_df[
                'Annual Income (k$)'
            ].mean()

            st.metric(
                "💰 Average Income",
                f"${avg_filtered_income:.1f}k"
            )

        else:

            st.metric(
                "💰 Average Income",
                "N/A"
            )

    with col3:

        if len(filtered_df) > 0:
            avg_filtered_spending = filtered_df[
                'Spending Score (1-100)'
            ].mean()

            st.metric(
                "🛍️ Average Spending",
                f"{avg_filtered_spending:.1f}"
            )

        else:

            st.metric(
                "🛍️ Average Spending",
                "N/A"
            )

    # --------------------------------------------------------
    # FILTERED VISUALIZATION
    # --------------------------------------------------------

    if len(filtered_df) > 0:

        st.divider()

        st.subheader("📈 Filtered Customer Distribution")

        fig, ax = plt.subplots(
            figsize=(11, 6)
        )

        sns.scatterplot(
            data=filtered_df,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            hue='Customer_Segment',
            s=90,
            ax=ax
        )

        ax.set_title(
            "Income vs Spending — Filtered Customers"
        )

        ax.set_xlabel(
            "Annual Income (k$)"
        )

        ax.set_ylabel(
            "Spending Score"
        )

        ax.grid(
            alpha=0.2
        )

        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left'
        )

        plt.tight_layout()

        st.pyplot(fig)

    else:

        st.warning(
            "No customers match the selected filters. "
            "Try widening the income or spending range."
        )

    # --------------------------------------------------------
    # CUSTOMER TABLE
    # --------------------------------------------------------

    st.divider()

    st.subheader("📋 Customer Records")

    st.write(
        f"Showing **{len(filtered_df)}** customers"
    )

    display_columns = [
        'CustomerID',
        'Gender',
        'Age',
        'Annual Income (k$)',
        'Spending Score (1-100)',
        'Customer_Segment'
    ]

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD FILTERED DATA
    # --------------------------------------------------------

    if len(filtered_df) > 0:

        csv_data = filtered_df[
            display_columns
        ].to_csv(index=False)

        st.download_button(
            label="⬇️ Download Filtered Customer Data",
            data=csv_data,
            file_name="filtered_customer_data.csv",
            mime="text/csv",
            use_container_width=True
        )

        # ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <hr>

    <div style="
        text-align:center;
        color:#667085;
        font-size:14px;
        padding:15px;
    ">
        <b>Customer Segmentation Analytics</b><br>
        K-Means & Hierarchical Clustering |
        Machine Learning & Business Analytics
    </div>
    """,
    unsafe_allow_html=True
)
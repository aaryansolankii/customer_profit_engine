import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

ARTIFACT_DIR = "artifacts"

@st.cache_data
def load_data():
    rfm = pd.read_csv(f"{ARTIFACT_DIR}/rfm.csv", index_col=0)
    final_df = pd.read_csv(f"{ARTIFACT_DIR}/final_df.csv", index_col=0)
    uplift_df_sorted = pd.read_csv(f"{ARTIFACT_DIR}/uplift_df_sorted.csv")
    eligible_df = pd.read_csv(f"{ARTIFACT_DIR}/eligible_df.csv", index_col=0)
    return rfm, final_df, uplift_df_sorted, eligible_df

@st.cache_resource
def load_models():
    bgf = joblib.load(f"{ARTIFACT_DIR}/bgf.pkl")
    ggf = joblib.load(f"{ARTIFACT_DIR}/ggf.pkl")
    model_treat = joblib.load(f"{ARTIFACT_DIR}/uplift_model_treat.pkl")
    model_ctrl = joblib.load(f"{ARTIFACT_DIR}/uplift_model_ctrl.pkl")
    return bgf, ggf, model_treat, model_ctrl

st.set_page_config(page_title="Customer Profitability Engine", layout="wide")

st.title("📊 Customer Profitability Engine (RFM + CLV + Uplift)")

rfm, final_df, uplift_df_sorted, eligible_df = load_data()
bgf, ggf, model_treat, model_ctrl = load_models()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "RFM Segments", "CLV", "Uplift Targeting"])

with tab1:
    st.subheader("Overview")
    st.metric("Total Customers", len(final_df))
    st.metric("Total Revenue (Monetary sum)", f"{rfm['Monetary'].sum():,.2f}")
    st.metric("Avg CLV (12m)", f"{final_df['CLV_12m'].mean():,.2f}")

    st.write("Segment distribution:")
    seg_counts = rfm["Segment"].value_counts()
    st.bar_chart(seg_counts)

with tab2:
    st.subheader("RFM Segments")
    st.write("RFM head:")
    st.dataframe(rfm.head())

    # heatmap pivot
    heatmap_data = (
        rfm.groupby(["R_score", "F_score"])
        .size()
        .reset_index(name="count")
        .pivot(index="R_score", columns="F_score", values="count")
    )

    fig, ax = plt.subplots()
    ax.set_title("RFM Heatmap (Recency vs Frequency)")
    import seaborn as sns
    sns.heatmap(heatmap_data, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader("CLV Distribution")
    st.write("CLV (12 months) histogram")

    fig, ax = plt.subplots()
    final_df["CLV_12m"].dropna().hist(bins=30, ax=ax)
    ax.set_xlabel("CLV 12m")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.write("Avg CLV by Segment")
    clv_by_seg = final_df.groupby("Segment")["CLV_12m"].mean().sort_values(ascending=False)
    st.bar_chart(clv_by_seg)

    st.subheader("Customer lookup")
    customer_id = st.selectbox("Select Customer ID", final_df.index.astype(str))
    if customer_id:
        row = final_df.loc[int(customer_id)]
        st.write("RFM + CLV for this customer:")
        st.json({
            "Recency": int(row["Recency"]),
            "Frequency": int(row["Frequency"]),
            "Monetary": float(row["Monetary"]),
            "Segment": row["Segment"],
            "CLV_6m": float(row["CLV_6m"]) if not pd.isna(row["CLV_6m"]) else None,
            "CLV_12m": float(row["CLV_12m"]) if not pd.isna(row["CLV_12m"]) else None,
        })

with tab4:
    st.subheader("Uplift Targeting")

    st.write("Qini / Uplift curve")
    fig, ax = plt.subplots()
    ax.plot(uplift_df_sorted["percent"], uplift_df_sorted["incremental_resp"])
    ax.set_xlabel("% of customers targeted")
    ax.set_ylabel("Incremental responses")
    ax.set_title("Qini Curve")
    st.pyplot(fig)

    st.write("Top 20% customers by uplift")
    cutoff = int(0.2 * len(uplift_df_sorted))
    top20 = uplift_df_sorted.iloc[:cutoff].copy()
    st.dataframe(top20[["CustomerID", "uplift", "p_treat", "p_ctrl"]].head(20))

    st.download_button(
        "Download full target list (CSV)",
        data=top20.to_csv(index=False),
        file_name="target_customers.csv",
        mime="text/csv"
    )

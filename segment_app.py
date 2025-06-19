import streamlit as st
import pandas as pd
import plotly.express as px

# Load the personas CSV
@st.cache_data
def load_data():
    return pd.read_csv("models/customer_segments.csv")

df = load_data()

# Set page config
st.set_page_config(page_title="Customer Segments Viewer", layout="wide")
st.title("📊 Customer Segmentation Explorer")

# Sidebar filters
st.sidebar.header("🔍 Filter Segments")
selected_cluster = st.sidebar.selectbox("Select Segment (Cluster)", options=sorted(df['cluster'].unique()))

# Filter data by segment
filtered_df = df[df['cluster'] == selected_cluster]

# Segment overview
st.subheader(f"📁 Segment {selected_cluster} Overview")
st.markdown("### Key Stats")
st.dataframe(filtered_df.describe().T.style.format("{:.2f}"))

# Persona Description
persona = filtered_df["persona"].iloc[0]
st.markdown(f"### 🧠 Persona: `{persona}`")

# Scatter plot
fig = px.scatter(
    df,
    x="pca1",
    y="pca2",
    color="cluster",
    hover_data=["persona", "monthly_avg", "payment_delay"],
    title="Customer Segments via PCA",
    template="plotly_dark",
    color_continuous_scale="Turbo",
    symbol="cluster"
)
fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# Display data
with st.expander("🔬 Show Raw Data"):
    st.dataframe(filtered_df.reset_index(drop=True))

# Footer
st.markdown("""
---
<div style='text-align:center; font-size:13px; color:#888;'>
    Built with ❤️ for customer insight and segmentation by <b>Prudhvi Raj</b>.
</div>
""", unsafe_allow_html=True)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv("data/model_features.csv").dropna()

# Select behavioral features for clustering
features = ['recency', 'monthly_avg', 'support_calls', 'payment_delay']
X = df[features]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal clusters with Elbow Method
def plot_elbow(X_scaled):
    distortions = []
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        distortions.append(km.inertia_)
    plt.figure(figsize=(6, 4))
    plt.plot(range(2, 10), distortions, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Uncomment to visualize elbow plot
# plot_elbow(X_scaled)

# Cluster using KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Add PCA components for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

# Optional: Plot with Plotly
fig = px.scatter(
    df,
    x='pca1',
    y='pca2',
    color='cluster',
    title='Customer Segments (KMeans + PCA)',
hover_data=['recency', 'monthly_avg', 'support_calls', 'payment_delay', 'is_churned']
)
fig.write_html("models/customer_segments.html")
# fig.show()
# Segment Summary
summary = df.groupby('cluster').agg({
    'is_churned': 'mean',   # ✅ correct column name
    'recency': 'mean',
    'monthly_avg': 'mean',
    'payment_delay': 'mean',
    'support_calls': 'mean',
    'monetary': 'mean',
    'customer_id': 'count'
}).rename(columns={'customer_id': 'count', 'is_churned': 'churn_rate'})

summary.columns = [
    'ChurnRate',     # churn_rate
    'Recency',
    'MonthlyAvg',
    'PaymentDelay',
    'SupportCalls',
    'Monetary',
    'Count'          # count
]

summary = summary.reset_index()


print("\n📊 Segment Summary:\n")
print(summary)
print("✅ Segmentation plot saved to models/customer_segments.html")
# Define persona labels
personas = {
    0: "High Risk / Low Value",
    1: "Low Risk / Medium Value",
    2: "Medium Risk / Medium Value",
    3: "Low Risk / High Value"
}

# Map persona labels
df["persona"] = df["cluster"].map(personas)

# Save final dataframe with all info
df.to_csv("models/customer_segments.csv", index=False)
print("✅ Saved: models/customer_segments.csv")

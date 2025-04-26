import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Load both CSVs
game_stats = pd.read_csv('Cleaned Data/Finished/PlayerStatistics.csv')
injuries = pd.read_csv('Cleaned Data/Finished/injury_locations_categorized.csv')

# 2. Summarize game stats per player
player_summary = game_stats.groupby('personId').agg({
    'firstName': 'first',
    'lastName': 'first',
    'numMinutes': ['count', 'mean'],
    'points': 'mean',
    'assists': 'mean',
    'reboundsTotal': 'mean'
}).reset_index()

player_summary.columns = [
    'personId', 'firstName', 'lastName',
    'games_played', 'avg_minutes',
    'avg_points', 'avg_assists', 'avg_rebounds'
]

# 3. Summarize injury count per player
injury_summary = injuries.groupby('Relinquished').size().reset_index(name='injury_count')

# 4. Merge injury count into player_summary
# Create a 'full_name' to match
player_summary['full_name'] = player_summary['firstName'] + ' ' + player_summary['lastName']
merged = pd.merge(player_summary, injury_summary, how='left', left_on='full_name', right_on='Relinquished')

# Fill players with no injuries as 0
merged['injury_count'] = merged['injury_count'].fillna(0)

# 5. Prepare features for clustering
features = merged[['games_played', 'avg_minutes', 'avg_points', 'avg_assists', 'avg_rebounds', 'injury_count']]

# 6. Drop NaNs
features = features.dropna()

# NEW: match merged to features so they align!
merged = merged.loc[features.index].reset_index(drop=True)

# 7. Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 8. Find best k using Elbow Method
sse = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 10), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Method For Optimal k')
plt.show()

# 9. Run final KMeans
kmeans = KMeans(n_clusters=4, random_state=42)  # you chose 4 clusters
clusters = kmeans.fit_predict(X_scaled)

# 10. Assign clusters correctly
merged['injury_risk_cluster'] = clusters

# 11. Analyze
print(merged.groupby('injury_risk_cluster').mean())


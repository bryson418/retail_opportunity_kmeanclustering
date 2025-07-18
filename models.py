import osmnx as ox
import geopandas as gpd
import numpy as np
import scipy.io
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from shapely.validation import explain_validity
from shapely.ops import unary_union
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from shapely.validation import explain_validity
# === Load and Process Traffic Dataset ===
import scipy.io

mat = scipy.io.loadmat('data/traffic_dataset.mat')


tra_adj_mat = mat['tra_adj_mat']
tra_X_tr = mat['tra_X_tr']
tra_X_te = mat['tra_X_te']
tra_X_tr = np.concatenate([tra_X_tr, tra_X_te], axis=1)

# Cluster Sensors
n_clusters = 3
sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
labels = sc.fit_predict(tra_adj_mat)

# Process Traffic Volume Data
num_time_steps = 2016
num_sensors = 36
intervals_per_day = 96
num_days = 21

volumes = []
for i in range(num_time_steps):
    sparse_mat = tra_X_tr[0, i + 1]
    dense_mat = sparse_mat.toarray()
    volume_vector = dense_mat[:, 0]
    volumes.append(volume_vector)

traffic_data = np.stack(volumes, axis=0).T
traffic_data_trimmed = traffic_data[:, :num_days * intervals_per_day]
traffic_data_reshaped = traffic_data_trimmed.reshape(num_sensors, num_days, intervals_per_day)
daily_avg = traffic_data_reshaped.mean(axis=2)
mean_daily_avg = daily_avg.mean(axis=1)

# Morning & Evening Peaks
avg_morning_peak = traffic_data_reshaped[:, :, 28:36].mean(axis=(1, 2))
avg_evening_peak = traffic_data_reshaped[:, :, 68:76].mean(axis=(1, 2))
sensor_ids = ["Sensor " + str(i) for i in range(1, num_sensors + 1)]

df = pd.DataFrame({
    'sensor_id': sensor_ids,
    'morning_peak_avg': avg_morning_peak,
    'evening_peak_avg': avg_evening_peak,
    'daily_avg': mean_daily_avg
})

# === Load and Process Sensor Locations ===
sensors = pd.read_csv("data/senslatlon - Sheet1.csv")


# Keep sensor_gdf in lat/lon for geopy distances
sensor_gdf_4326 = gpd.GeoDataFrame(
    sensors,
    geometry=gpd.points_from_xy(sensors.Longitude, sensors.Latitude),
    crs="EPSG:4326"
)

# Projected GeoDataFrame in meters for spatial operations like buffering
sensor_gdf_3857 = sensor_gdf_4326.to_crs(epsg=3857)

buffer_radius_m = 500

# Create buffer zones around sensors in projected CRS (meters)
sensor_union = unary_union(sensor_gdf_3857.geometry.buffer(buffer_radius_m))


# if sensor_union.is_empty or not sensor_union.is_valid or np.isnan(sensor_union.area):
#     print("❌ sensor_union geometry invalid:", explain_validity(sensor_union))
#     raise ValueError("Invalid sensor_union geometry. Check sensor coordinates or CRS.")
# else:
#     print("✅ sensor_union geometry is valid.")

    # Convert polygon to lat/lon CRS for OSM query
sensor_union_4326 = gpd.GeoSeries([sensor_union], crs="EPSG:3857").to_crs(epsg=4326).unary_union

    # Query OSM for shops and amenities
tags = {'shop': True, 'amenity': True}
businesses = ox.features_from_polygon(sensor_union_4326.convex_hull, tags=tags)

# Clean businesses data
businesses = businesses[~businesses['name'].isna()]
businesses = businesses[~businesses.geometry.is_empty]
businesses = businesses[['name', 'amenity', 'shop', 'geometry']]

# Convert businesses to GeoDataFrame in meters for spatial operations
businesses = gpd.GeoDataFrame(businesses, geometry='geometry', crs='EPSG:4326').to_crs(epsg=3857)

# Use centroids of businesses for points
businesses['geometry'] = businesses.geometry.centroid

# Combine shop and amenity columns to single business_type
businesses['business_type'] = businesses['shop'].fillna(businesses['amenity'])

# Filter relevant business types
relevant_types = [
    'clothes', 'fashion', 'shoes', 'jewelry', 'cosmetics', 'bakery', 'supermarket', 'convenience',
    'electronics', 'mobile_phone', 'books', 'gift', 'florist', 'furniture', 'hardware',
    'car_repair', 'stationery', 'toys', 'kiosk', 'department_store', 'butcher',
    'greengrocer', 'hairdresser', 'optician', 'travel_agency', 'pet', 'retail'
]
businesses = businesses[businesses['business_type'].isin(relevant_types)].copy()

# Keep only businesses within 500m of sensors using projected CRS distance
businesses['within_500m'] = businesses.geometry.apply(
    lambda geom: sensor_gdf_3857.distance(geom).min() <= buffer_radius_m
)
businesses = businesses[businesses['within_500m']].copy()
businesses.drop(columns=['within_500m'], inplace=True)

# Assign nearest sensor and distance in meters (using projected CRS)
def get_nearest_sensor(business_geom):
    distances = sensor_gdf_3857.distance(business_geom)
    nearest_idx = distances.idxmin()
    return sensor_gdf_3857.loc[nearest_idx, 'Sensor']

businesses['nearest_sensor_id'] = businesses.geometry.apply(get_nearest_sensor)
businesses['distance_m'] = businesses.geometry.apply(lambda g: sensor_gdf_3857.distance(g).min())

# Convert businesses back to lat/lon for downstream use
businesses = businesses.to_crs("EPSG:4326")
businesses['longitude'] = businesses.geometry.x
businesses['latitude'] = businesses.geometry.y

# Drop unused columns for clarity
businesses = businesses.drop(columns=['geometry', 'amenity', 'shop'])

# Reorder columns
businesses = businesses[['name', 'business_type', 'latitude', 'longitude', 'nearest_sensor_id', 'distance_m']]

# === Competition Score Calculation ===
def competition_score_idw(lat, lon, business_df, radius_m=5000, category=None):
    # if ((business_df['latitude'] < -90) | (business_df['latitude'] > 90)).any():
    #     raise ValueError("Businesses DataFrame contains invalid latitude values.")
    # if ((business_df['longitude'] < -180) | (business_df['longitude'] > 180)).any():
    #     raise ValueError("Businesses DataFrame contains invalid longitude values.")

    score = 0.0
    for _, row in business_df.iterrows():
        if category and row['business_type'] != category:
            continue
        dist = geodesic((lat, lon), (row['latitude'], row['longitude'])).meters
        if dist <= radius_m and dist > 0:
            score += 1
    return score

def calculate_competition_scores(df, radius_m=5000, category_col='business_type'):
    scores = []
    for _, row in df.iterrows():
        score = competition_score_idw(
            lat=row['latitude'],
            lon=row['longitude'],
            business_df=df,
            radius_m=radius_m,
            category=row[category_col]
        )
        scores.append(score)
    return scores

businesses['competition_score'] = calculate_competition_scores(businesses)

# Merge traffic and business data
merged_df = businesses.merge(
    df,
    how='left',
    left_on='nearest_sensor_id',
    right_on='sensor_id'
)

# Clustering and opportunity levels
features = merged_df[['daily_avg', 'competition_score']].copy()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
merged_df['cluster'] = kmeans.fit_predict(features_scaled)

cluster_map = {
    0: 'High Opportunity',
    1: 'Low Opportunity',
    2: 'Medium Opportunity'
}
merged_df['opportunity_level'] = merged_df['cluster'].map(cluster_map)

# === Helper functions for API ===
def get_nearest_sensor_id(lat, lon, sensor_gdf=sensor_gdf_4326):
    point = (lat, lon)
    distances = sensor_gdf.geometry.apply(lambda g: geodesic(point, (g.y, g.x)).meters)
    nearest_idx = distances.idxmin()
    return sensor_gdf.loc[nearest_idx, 'Sensor']

def get_traffic_features_for_sensor(sensor_id, df_traffic=df):
    row = df_traffic[df_traffic['sensor_id'] == sensor_id]
    if row.empty:
        return None
    return {
        'morning_peak_avg': float(row['morning_peak_avg']),
        'evening_peak_avg': float(row['evening_peak_avg']),
        'daily_avg': float(row['daily_avg'])
    }

def predict_opportunity(lat, lon, category=None):
    # if not (-90 <= lat <= 90):
    #     raise ValueError("Latitude must be in [-90, 90]")
    # if not (-180 <= lon <= 180):
    #     raise ValueError("Longitude must be in [-180, 180]")

    comp_score = competition_score_idw(lat, lon, businesses, radius_m=5000, category=category)
    nearest_sensor_id = get_nearest_sensor_id(lat, lon)
    traffic_features = get_traffic_features_for_sensor(nearest_sensor_id, df)

    if traffic_features is None:
        return None

    feature_vec = np.array([[traffic_features['daily_avg'], comp_score]])
    feature_vec_scaled = scaler.transform(feature_vec)

    cluster_label = kmeans.predict(feature_vec_scaled)[0]
    opportunity = cluster_map.get(cluster_label, "Unknown")

    return {
        "competition_score": comp_score,
        "daily_avg_traffic": traffic_features['daily_avg'],
        "morning_peak_avg": traffic_features['morning_peak_avg'],
        "evening_peak_avg": traffic_features['evening_peak_avg'],
        "opportunity_level": opportunity
    }

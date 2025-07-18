from flask import Flask, render_template, request, jsonify
from geopy.distance import geodesic
import models 
import json

app = Flask(__name__)

# --- Helper Function ---
def find_nearest_sensor(lat, lon):
    min_dist = float("inf")
    nearest_sensor = None
    for _, row in models.sensor_gdf_4326.iterrows():
        dist = geodesic((lat, lon), (row['Latitude'], row['Longitude'])).meters
        if dist < min_dist:
            min_dist = dist
            nearest_sensor = row['Sensor']
    return nearest_sensor

# --- Core Analysis ---
def get_opportunity_level(lat, lon, business_type):
    try:
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude {lat} out of range [-90, 90]")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Longitude {lon} out of range [-180, 180]")

        result = models.predict_opportunity(lat, lon, category=business_type)
        
        if result is None:
            return 0, "No sensor data nearby"
        
        return result['competition_score'], result['opportunity_level']
    except Exception as e:
        print(f"Error in get_opportunity_level: {e}")
        return 0, f"Error: {str(e)}"

# --- Main Page ---
@app.route('/')
def index():
    business_types = models.relevant_types
    sensors = models.sensor_gdf_4326.dropna(subset=['Latitude', 'Longitude'])
    center_lat = sensors['Latitude'].mean()
    center_lon = sensors['Longitude'].mean()

    # Prepare sensor data for JavaScript
    sensor_data = []
    for _, row in sensors.iterrows():
        sensor_data.append({
            'name': row['Sensor'],
            'lat': float(row['Latitude']),
            'lon': float(row['Longitude'])
        })

    traffic_data = []

    return render_template(
        "index.html", 
        business_types=business_types,
        center_lat=center_lat,
        center_lon=center_lon,
        sensor_data=json.dumps(sensor_data),
        traffic_data=json.dumps(traffic_data)
    )

# --- Backend Analysis Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        
        lat = float(data['lat'])
        lon = float(data['lon'])
        business_type = data.get('business_type', models.relevant_types[0])

        comp_score, opportunity = get_opportunity_level(lat, lon, business_type)

        result = {
            'success': True,
            'lat': lat,
            'lon': lon,
            'business_type': business_type,
            'competition_score': round(comp_score, 2),
            'opportunity': opportunity
        }
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

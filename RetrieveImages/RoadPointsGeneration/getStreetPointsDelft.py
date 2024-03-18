import requests
import numpy as np
import math
import pandas as pd
import geopandas
from shapely.geometry import LineString, Point
from pyproj import Transformer
from shapely.ops import nearest_points
import json
from datetime import datetime
import os
from visualize_roads import visualize_roads

# This script reads a shape file that contains points.
# Then, it uses overpass API to retrieve road data of Delft, visualized in an HTML file.
# It finds for each point in the shape file, the nearest road segment. 
# That road segment is then used to calculate the bearing of the road in degrees.
# Two field points that are perpendictular to the road on either side are found.
# The results are saved to a CSV file.
# The headers in the CSV file mean respectively:  
# y: latitude of the point, x: longitude of the point, b: bearing of the road,
# y1: latitude of the first field point, x1: longitude of the first field point,
# y2: latitude of the second field point, x2: longitude of the second field point.


# Constants
OVERPASS_API_URL = "http://overpass-api.de/api/interpreter"
SHAPEFILE_PATH = 'TreehealthDataset/bomen.shp'
OUTPUT_CSV = 'nearest_road_bearings_and_field_points.csv'
PERPENDICULAR_DISTANCE = 30  # Distance for calculating field points, in meters
EARTH_RADIUS = 6371e3  # in meters
TIMESTAMP_NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def convert_rd_to_wgs(lat, lon):
    """Convert RD New (EPSG:28992) coordinates to WGS 84 (EPSG:4326)."""
    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    lon_wgs, lat_wgs = transformer.transform(lon, lat)
    return lat_wgs, lon_wgs

def fetch_overpass_data_delft(from_api=False):
    if(from_api):
        """Fetch road data for Delft using a polygon."""
        query = f"""
        [out:json];
        area["name"="Delft"]->.searchArea;
        (
        way(area.searchArea)["highway"];
        );
        out geom;
        """
        response = requests.get(OVERPASS_API_URL, params={'data': query})
        if response.status_code == 200:
            data = response.json()


            # Save the api response to a file
            filename = f'TreehealthDataset/overpass_data_{TIMESTAMP_NOW}.json'
            with open(filename, 'w') as file:
                json.dump(data, file)
            return data
        else:
            return None
        
    else:
        with open('TreehealthDataset/overpass_data.json') as file:
            return json.load(file)

def compute_bearing(from_point, to_point):
    """Calculate the bearing from one geographic point to another."""
    y = math.sin(to_point[1] - from_point[1]) * math.cos(to_point[0])
    x = math.cos(from_point[0]) * math.sin(to_point[0]) - \
        math.sin(from_point[0]) * math.cos(to_point[0]) * math.cos(to_point[1] - from_point[1])
    θ = math.atan2(y, x)
    bearing = (θ * 180 / math.pi + 360) % 360
    return bearing

def compute_point_on_field(from_point, theta, distance):
    """Calculate a point at a certain distance and bearing from a given point."""
    angular_distance = distance / EARTH_RADIUS
    theta = math.radians(theta)
    lat1 = math.radians(from_point[0])
    lon1 = math.radians(from_point[1])
    lat2 = math.asin(math.sin(lat1) * math.cos(angular_distance) + \
                     math.cos(lat1) * math.sin(angular_distance) * math.cos(theta))
    lon2 = lon1 + math.atan2(math.sin(theta) * math.sin(angular_distance) * math.cos(lat1),
                             math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2))
    return (math.degrees(lat2), math.degrees(lon2))

def process_points_and_save_to_csv(geo_data, road_data):
    """Find the closest way to each point and compute the bearing."""
    results = []
    batch_size = 100  # Define the batch size
    all_roads = [LineString([(node['lon'], node['lat']) for node in element['geometry']]) for element in road_data['elements'] if element['type'] == 'way']
    for idx, row in geo_data.iterrows():
        if row.geometry.type == "Point":
            lat, lon = convert_rd_to_wgs(row.geometry.y, row.geometry.x)
            point = Point(lon, lat)

            # Find the closest road to the point
            closest_road = min(all_roads, key=lambda road: point.distance(road))
            
            # Use nearest_points to find the closest point on this road to the point
            _, nearest_road_point = nearest_points(point, closest_road)
            
            # Now find the segment of closest_road that is nearest to nearest_road_point
            min_distance = float('inf')
            for i in range(len(closest_road.coords) - 1):
                segment = LineString([closest_road.coords[i], closest_road.coords[i + 1]])
                distance = nearest_road_point.distance(segment)
                if distance < min_distance:
                    min_distance = distance
                    from_point = closest_road.coords[i]
                    to_point = closest_road.coords[i + 1]
            
            # Convert from_point and to_point to lat, lon for bearing calculation
            from_point = (from_point[1], from_point[0])  # Assuming (lon, lat) to (lat, lon)
            to_point = (to_point[1], to_point[0])
            if len(closest_road.coords) > 1:
                bearing = compute_bearing(from_point, to_point)
                p1 = compute_point_on_field(to_point, (bearing + 90) % 360, PERPENDICULAR_DISTANCE)
                p2 = compute_point_on_field(to_point, (bearing + 270) % 360, PERPENDICULAR_DISTANCE)
                results.append([lat, lon, bearing, p1[0], p1[1], p2[0], p2[1]])
                # print([lat, lon, bearing, p1[0], p1[1], p2[0], p2[1]], "from", from_point, "to", to_point)

                # Save the batch when it reaches the batch size
                if idx > 0 and idx % batch_size == 0:
                    save_batch_to_csv(results)
                    results = []  # Reset results for the next batch
                    print(f"Processed {idx} / {len(geo_data)} road points")
                    

    if results:
        save_batch_to_csv(results)
        print(f"Processing done")

def save_batch_to_csv(results, filename=f"roadPoints/roadPointsDelft{TIMESTAMP_NOW}.csv"):
    """Append a batch of results to a CSV file."""
    df = pd.DataFrame(results, columns=['y', 'x', 'b', 'y1', 'x1', 'y2', 'x2'])
    # Append mode, add header only if the file does not exist
    header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=header)


if __name__ == "__main__":
    # Fetch road data for Delft
    road_data = fetch_overpass_data_delft()
    visualize_roads(road_data)

    # Load the geo data from the shapefile
    geo_data = geopandas.read_file(SHAPEFILE_PATH)

    # Process the points and save the results to a CSV file
    process_points_and_save_to_csv(geo_data, road_data)

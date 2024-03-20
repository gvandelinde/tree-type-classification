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

# This script reads a shape file that contains location of trees (tree points).
# Then, it uses overpass API to retrieve road data of Delft, visualized in an HTML file.
# It finds for each point in the shape file, the nearest OSM "way" (a road)
# and computes the bearing from the tree location to the nearest road point. 
# The results are saved to a CSV file.
# The headers in the CSV file mean respectively:  
# tree_lon: longitude of the tree
# tree_lat: latitude of the tree
# bearing: bearing from the tree to the nearest road
# road_lon: longitude of the nearest point on the road
# road_lat: latitude of the nearest point on the road
# CONDITIE: condition of the tree
 


# Constants
OVERPASS_API_URL = "http://overpass-api.de/api/interpreter"
SHAPEFILE_PATH = 'RetrieveGSV/RoadPointsGeneration/TreehealthDataset/bomen.shp'
PERPENDICULAR_DISTANCE = 30  # Distance for calculating field points, in meters
EARTH_RADIUS = 6371e3  # in meters
USE_OVERPASS_API_INSTEAD_OF_FILE = True


def convert_rd_to_wgs(lat, lon):
    """Convert RD New (EPSG:28992) coordinates to WGS 84 (EPSG:4326)."""
    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    lon_wgs, lat_wgs = transformer.transform(lon, lat)
    return lat_wgs, lon_wgs

def fetch_overpass_data_delft(from_api=False):
    print("Fetching overpass road data...")
    if(from_api):
        """Fetch road data for Delft."""
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
            filename = f'RetrieveGSV/RoadPointsGeneration/OverpassData/overpass_data.json'
            with open(filename, 'w') as file:
                json.dump(data, file)
            return data
        else:
            return None
        
    else:
        with open('RetrieveGSV/RoadPointsGeneration/OverpassData/overpass_data.json') as file:
            return json.load(file)
def compute_bearing2(from_point, to_point):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = map(math.radians, from_point)
    lat2, lon2 = map(math.radians, to_point)
    # compute the difference in longitudes
    delta_lon = lon2 - lon1
    # compute the bearing
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))
    initial_bearing = math.atan2(x, y)
    #convert bearing from radians to degrees
    initial_bearing = math.degrees(initial_bearing)
    # no negative angles allowed, so 0 <= bearing < 360
    bearing = (initial_bearing + 360) % 360
    # print("computing bearing from ", from_point, " to ", to_point, "to be ", bearing, "initial_bearing", initial_bearing)
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
            tree_lat, tree_lon = convert_rd_to_wgs(row.geometry.y, row.geometry.x)
            point = Point(tree_lon, tree_lat)

            # Find the closest road to the point
            closest_road = min(all_roads, key=lambda road: point.distance(road))
            
            # Use nearest_points to find the closest point on this road to the point
            _, nearest_road_point = nearest_points(point, closest_road)

            if len(closest_road.coords) > 1:
                from_point = (nearest_road_point.y, nearest_road_point.x)
                to_point = (tree_lat, tree_lon)
                bearing = compute_bearing2(from_point, to_point)
                conditie = row.CONDITIE
                # print(nearest_road_point)
                results.append([tree_lat, tree_lon, bearing, nearest_road_point.y, nearest_road_point.x, conditie])

                # Save the batch when it reaches the batch size
                if idx > 0 and idx % batch_size == 0:
                    save_batch_to_csv(results)
                    results = []  # Reset results for the next batch
                    print(f"Processed {idx} / {len(geo_data)} road points")
                    

    if results:
        save_batch_to_csv(results)
        print(f"Processing done")

def save_batch_to_csv(results, filename=f"RetrieveGSV/RoadPointsGeneration/roadPoints/roadPointsDelft.csv"):
    """Append a batch of results to a CSV file."""
    df = pd.DataFrame(results, columns=['tree_lat', 'tree_lon', 'bearing', 'road_lat', 'road_lon', "CONDITIE"])
    # Append mode, add header only if the file does not exist
    header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=header)


if __name__ == "__main__":
    # Fetch road data for Delft
    road_data = fetch_overpass_data_delft(USE_OVERPASS_API_INSTEAD_OF_FILE)
    # visualize_roads(road_data)

    # Load the geo data from the shapefile
    geo_data = geopandas.read_file(SHAPEFILE_PATH)

    # If the output file already exists, rename it
    output_path = "RetrieveGSV/RoadPointsGeneration/roadPoints/roadPointsDelft.csv"
    if os.path.exists(output_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"RetrieveGSV/RoadPointsGeneration/roadPoints/roadPointsDelft_{timestamp}.csv"
        os.rename(output_path, new_filename)
        print(f"Existing file renamed to: {new_filename}")

    # Process the points and save the results to a CSV file
    process_points_and_save_to_csv(geo_data, road_data)

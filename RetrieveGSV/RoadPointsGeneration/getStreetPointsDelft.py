from cmath import cos
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

from visualizeOnMaps.visualize_road_segments import visualize_road_segments
from visualizeOnMaps.visualize_trees_and_cameras import visualize_trees_and_cameras

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
OUTPUT_FILE_RESULTS = 'RetrieveGSV/RoadPointsGeneration/roadPoints/roadPointsDelft.csv'
OUTPUT_FILE_SKIPPED = 'RetrieveGSV/RoadPointsGeneration/roadPoints/skippedPoints.csv'
OUTPUT_FILE_AUGMENTED = 'RetrieveGSV/RoadPointsGeneration/roadPoints/augmentedRoadPointsDelft.csv'
OUTPUT_FILE_OVERPASS = 'RetrieveGSV/RoadPointsGeneration/OverpassData/overpass_data.json'
USE_OVERPASS_API_INSTEAD_OF_FILE = False
CAMERA_TOO_FAR_THRESHOLD = 50  # Threshold in meters
BATCH_SIZE = 100
AUGMENTATION_DISTANCE = 5 # In meters


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

            rename_file_if_exists(OUTPUT_FILE_OVERPASS)
            # Save the api response to a file in case this script is run again
            with open(OUTPUT_FILE_OVERPASS, 'w') as file:
                json.dump(data, file)
            return data
        else:
            print("Failed to fetch data from overpass API", response)
            return None
        
    else:
        with open(OUTPUT_FILE_OVERPASS) as file:
            return json.load(file)
        
def compute_bearing(from_point, to_point):
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


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in meters between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in meters. Use 6371 for kilometers
    r = 6371000
    return c * r

def form_result(from_point, to_point, conditie):
    bearing = compute_bearing(from_point, to_point)
    point_lat, point_lon = from_point
    tree_lat, tree_lon = to_point
    return [tree_lat, tree_lon, bearing, point_lat, point_lon, conditie]

def process_points_and_save_to_csv(geo_data, road_data):
    """Find the closest way to each point and compute the bearing."""
    results = []
    augmented = []
    skipped = []
    batch_size = BATCH_SIZE # Define the batch size
    all_roads = [LineString([(node['lon'], node['lat']) for node in element['geometry']]) for element in road_data['elements'] if element['type'] == 'way']
    for idx, row in geo_data.iterrows():
        conditie = row.CONDITIE
        tree_lat, tree_lon = convert_rd_to_wgs(row.geometry.y, row.geometry.x)
        if row.geometry.type == "Point":
            treepoint = Point(tree_lon, tree_lat)

            # Find the closest road to the point
            closest_road = min(all_roads, key=lambda road: treepoint.distance(road))

            # ADD: 
            
            # print(f"all roads: {}\n")
            # Use nearest_points to find the closest point on this road to the point
            _, nearest_road_point = nearest_points(treepoint, closest_road)
            distance_of_nearest_road_point = closest_road.project(nearest_road_point)  
            augmentation_offset = AUGMENTATION_DISTANCE / 100000

            # Datapoints for data-augmentation
            interpolated_minus_offset = closest_road.interpolate(distance_of_nearest_road_point - augmentation_offset)
            interpolated_plus_offset= closest_road.interpolate(distance_of_nearest_road_point + augmentation_offset)

            # Calculate haversine distances
            distance = haversine_distance(tree_lat, tree_lon, nearest_road_point.y, nearest_road_point.x)
            augmented_distance_minus = haversine_distance(tree_lat, tree_lon, interpolated_minus_offset.y, interpolated_minus_offset.x)
            augmented_distance_plus = haversine_distance(tree_lat, tree_lon, interpolated_plus_offset.y, interpolated_plus_offset.x)

            if len(closest_road.coords) > 1:
                # Compute the distance from the tree to the nearest road point
                from_point = (nearest_road_point.y, nearest_road_point.x)
                to_point = (tree_lat, tree_lon)
                if distance >= CAMERA_TOO_FAR_THRESHOLD:
                    bearing = compute_bearing(from_point, to_point)
                    skipped.append([tree_lat, tree_lon, bearing, nearest_road_point.y, nearest_road_point.x, conditie, "Too far from road ({:.2f} m)".format(distance)])
                elif(conditie == None):
                    bearing = compute_bearing(from_point, to_point)
                    skipped.append([tree_lat, tree_lon, bearing, nearest_road_point.y, nearest_road_point.x, conditie, "No condition"])
                else:
                    results.append(form_result(from_point, to_point, conditie))
                    # Data augmentation! 

                    if not (conditie == "Matig" or conditie == "Redelijk"):
                        if augmented_distance_minus <= CAMERA_TOO_FAR_THRESHOLD: 
                            augmented.append(form_result((interpolated_minus_offset.y, interpolated_minus_offset.x), to_point, conditie) + [nearest_road_point.y, nearest_road_point.x, f"-{AUGMENTATION_DISTANCE}m"])
                        if augmented_distance_plus <= CAMERA_TOO_FAR_THRESHOLD: 
                            augmented.append(form_result((interpolated_plus_offset.y, interpolated_plus_offset.x), to_point, conditie) + [nearest_road_point.y, nearest_road_point.x, f"+{AUGMENTATION_DISTANCE}m"])

                # Save the batch when it reaches the batch size
                if idx > 0 and idx % batch_size == 0:
                    save_batch_to_csv(results, OUTPUT_FILE_RESULTS, ['tree_lat', 'tree_lon', 'bearing', 'road_lat', 'road_lon', "CONDITIE"])
                    save_batch_to_csv(skipped, OUTPUT_FILE_SKIPPED, ['tree_lat', 'tree_lon', 'bearing', 'road_lat', 'road_lon', "CONDITIE", "Reason"])
                    save_batch_to_csv(augmented, OUTPUT_FILE_AUGMENTED, ['tree_lat', 'tree_lon', 'bearing', 'road_lat', 'road_lon', "CONDITIE", "original_lat", "original_lon", "adjustment"])
                    print(f"Processed {idx} / {len(geo_data)} road points. Skipped {len(skipped)} due to null values or being further than {CAMERA_TOO_FAR_THRESHOLD} m from the road.")
                    results = []  # Reset results for the next batch
                    skipped = []  # Reset skipped for the next batch
                    augmented = []
        else:
            print("Skipping row with invalid geometry:", row)

    if results:
        save_batch_to_csv(results, OUTPUT_FILE_RESULTS, ['tree_lat', 'tree_lon', 'bearing', 'road_lat', 'road_lon', "CONDITIE"])
        save_batch_to_csv(skipped, OUTPUT_FILE_SKIPPED, ['tree_lat', 'tree_lon', 'bearing', 'road_lat', 'road_lon', "CONDITIE", "Reason"])
        print(f"Processing done. \nSkipped {len(skipped)} points cached in skipped.Points.csv.")

def save_batch_to_csv(results, filename, columns_header):
    """Append a batch of results to a CSV file."""
    df = pd.DataFrame(results, columns=columns_header)
    # Append mode, add header only if the file does not exist
    header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=header)

def rename_file_if_exists(filename):
    if os.path.exists(filename):
        # Generate a timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # Split the filename to add the timestamp before the file extension
        base, extension = os.path.splitext(filename)
        # Create a new filename with the timestamp
        new_filename = f"{base}_{timestamp}{extension}"
        # Rename the file
        os.rename(filename, new_filename)
        print(f"Renamed existing file to: {new_filename}")

if __name__ == "__main__":
    # Fetch road data for Delft
    road_data = fetch_overpass_data_delft(USE_OVERPASS_API_INSTEAD_OF_FILE)

    # Load the geo data from the shapefile
    geo_data = geopandas.read_file(SHAPEFILE_PATH)

    # If the output files already exists, rename it
    rename_file_if_exists(OUTPUT_FILE_RESULTS)
    rename_file_if_exists(OUTPUT_FILE_SKIPPED)
    rename_file_if_exists(OUTPUT_FILE_AUGMENTED)

    # Process the points and save the results to a CSV file
    process_points_and_save_to_csv(geo_data, road_data)
    visualize_road_segments(road_data)
    visualize_trees_and_cameras(OUTPUT_FILE_RESULTS, OUTPUT_FILE_SKIPPED)
    
import folium
import pandas as pd

OUTPUT_FILE = 'RetrieveGSV/RoadPointsGeneration/visualizeOnMaps/trees_and_cameras.html'
COLOR_TREE = 'green'
COLOR_SKIPPED_DISTANCE = 'orange'  # Using a dark orange for distance-related skips
COLOR_SKIPPED_NO_DATA = '#C65C1B'  # A bit darker shade of orange for no data skips

def visualize_trees_and_cameras(csv_path, skipped_csv_path):
    """
    Visualizes tree points and road points on a map. Tree points are in green, road points in blue.
    Skipped points due to distance are displayed with tree points in light grey and their corresponding 
    road points in dark grey, connected by a grey line. Skipped points due to no condition are displayed 
    in grey without a connecting line to a road point.

    Args:
        csv_path (str): Path to the CSV file containing tree and road points data.
        skipped_csv_path (str): Path to the CSV file containing skipped points data.
    """
    print("Visualizing points on a map...")
    # Load data from CSV
    df = pd.read_csv(csv_path)
    skipped_df = pd.read_csv(skipped_csv_path)

    # Create a map centered around the average location
    map_center = [df['tree_lat'].mean(), df['tree_lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=15)

    # Add processed points and lines to the map
    for index, row in df.iterrows():
        tree_location = [row['tree_lat'], row['tree_lon']]
        road_location = [row['road_lat'], row['road_lon']]
        
        folium.CircleMarker(location=tree_location, radius=3, color=COLOR_TREE, fill=True, fill_color=COLOR_TREE).add_to(m)
        folium.PolyLine(locations=[road_location, tree_location], color=COLOR_TREE, weight=1).add_to(m)
    
    # Add skipped points to the map with specific reasons
    for index, row in skipped_df.iterrows():
        tree_location = [row['tree_lat'], row['tree_lon']]
        road_location = [row['road_lat'], row['road_lon']]
        reason = row['Reason']
        
        if 'Too far' in reason:
            # Skipped due to distance - use light grey for trees, dark grey for roads

            folium.CircleMarker(location=tree_location, radius=3, color=COLOR_SKIPPED_DISTANCE, fill=True, fill_color=COLOR_SKIPPED_DISTANCE, popup=f'Skipped Tree: {reason}').add_to(m)
            folium.PolyLine(locations=[tree_location, road_location], color=COLOR_SKIPPED_DISTANCE, weight=1, dash_array='5, 5').add_to(m)
        elif 'No condition' in reason:
            # Skipped due to no condition - use grey for trees without adding road points or lines
            folium.CircleMarker(location=tree_location, radius=3, color=COLOR_SKIPPED_NO_DATA, fill=True, fill_color=COLOR_SKIPPED_NO_DATA, popup=f'Skipped Tree: {reason}').add_to(m)
    legend_html = '''
    <div style="position: fixed; 
    bottom: 20px; left: 20px; width: 200px; height: 130px; 
    background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
    overflow: hidden; padding: 10px;">  <!-- Adjusted overflow to hidden -->
    <b>Legend</b><br>
    <i style="background:{}; border-radius:50%; width:10px; height:10px; display:inline-block"></i> Tree Location<br>
    <i style="background:{}; border-radius:50%; width:10px; height:10px; display:inline-block"></i> Skipped (Distance to Road)<br>
    <i style="background:{}; border-radius:50%; width:10px; height:10px; display:inline-block"></i> Skipped (No Data)
    </div>
    '''.format(COLOR_TREE, COLOR_SKIPPED_DISTANCE, COLOR_SKIPPED_NO_DATA)

    # Add the legend to the map only once
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save the map to an HTML file
    m.save(OUTPUT_FILE)
    print(f"Map saved to {OUTPUT_FILE}")

# import folium
# import pandas as pd

# OUTPUT_FILE = 'map_visualization.html'

# def visualize_tree_to_road(csv_path, skipped_csv_path):
#     """
#     Visualizes tree points and road points on a map. Tree points are in green, road points in blue.
#     Skipped points due to distance are displayed with tree points in light grey and their corresponding 
#     road points in dark grey, connected by a grey line. Skipped points due to no condition are displayed 
#     in orange, with a similar setup.

#     Args:
#         csv_path (str): Path to the CSV file containing tree and road points data.
#         skipped_csv_path (str): Path to the CSV file containing skipped points data.
#     """
#     # Load data from CSV
#     df = pd.read_csv(csv_path)
#     skipped_df = pd.read_csv(skipped_csv_path)

#     # Create a map centered around the average location
#     map_center = [df['tree_lat'].mean(), df['tree_lon'].mean()]
#     m = folium.Map(location=map_center, zoom_start=15)

#     # Add processed points and lines to the map
#     for index, row in df.iterrows():
#         tree_location = [row['tree_lat'], row['tree_lon']]
#         road_location = [row['road_lat'], row['road_lon']]
        
#         folium.CircleMarker(location=tree_location, radius=3, color='green', fill=True, fill_color='green').add_to(m)
#         folium.CircleMarker(location=road_location, radius=3, color='blue', fill=True, fill_color='blue').add_to(m)
#         folium.PolyLine(locations=[road_location, tree_location], color='black', weight=1).add_to(m)
    
#     # Add skipped points to the map with specific reasons
#     for index, row in skipped_df.iterrows():
#         tree_location = [row['tree_lat'], row['tree_lon']]
#         road_location = [row['road_lat'], row['road_lon']]
#         reason = row['Reason']
        
#         if 'Too far' in reason:
#             # Skipped due to distance - use light grey for trees, dark grey for roads
#             tree_color = '#D3D3D3'  # Light Grey
#             road_color = '#A9A9A9'  # Dark Grey
#         elif 'No condition' in reason:
#             # Skipped due to no condition - use orange for both
#             tree_color = road_color = 'orange'
#         else:
#             # Default color for any other reason (if exists)
#             tree_color = road_color = 'pink'
        
#         folium.CircleMarker(location=tree_location, radius=3, color=tree_color, fill=True, fill_color=tree_color, popup=f'Skipped Tree: {reason}').add_to(m)
#         folium.CircleMarker(location=road_location, radius=3, color=road_color, fill=True, fill_color=road_color, popup='Skipped Road Point').add_to(m)
#         folium.PolyLine(locations=[tree_location, road_location], color=road_color, weight=1, dash_array='5, 5').add_to(m)

#     # Save the map to an HTML file
#     m.save(OUTPUT_FILE)
#     print(f"Map saved to {OUTPUT_FILE}")

# # import folium
# # import pandas as pd

# # OUTPUT_FILE = 'map_visualization.html'

# # def visualize_tree_to_road(csv_path, skipped_csv_path):
# #     """
# #     Visualizes tree points and road points on a map. Tree points are in green, 
# #     road points in blue, and skipped points are displayed with tree points in 
# #     greenish-grey and their corresponding road points in grey, connected by a grey 
# #     line, indicating that these points were skipped during processing.

# #     Args:
# #         csv_path (str): Path to the CSV file containing tree and road points data.
# #         skipped_csv_path (str): Path to the CSV file containing skipped points data.
# #     """
# #     # Load data from CSV
# #     df = pd.read_csv(csv_path)
# #     skipped_df = pd.read_csv(skipped_csv_path)

# #     # Create a map centered around the average location
# #     map_center = [df['tree_lat'].mean(), df['tree_lon'].mean()]
# #     m = folium.Map(location=map_center, zoom_start=15)

# #     # Add processed points and lines to the map
# #     for index, row in df.iterrows():
# #         tree_location = [row['tree_lat'], row['tree_lon']]
# #         road_location = [row['road_lat'], row['road_lon']]
        
# #         folium.CircleMarker(location=tree_location, radius=3, color='green', fill=True, fill_color='green').add_to(m)
# #         folium.CircleMarker(location=road_location, radius=3, color='blue', fill=True, fill_color='blue').add_to(m)
# #         folium.PolyLine(locations=[road_location, tree_location], color='black', weight=1).add_to(m)
    
# #     # Add skipped points to the map with tree points in greenish-grey
# #     for index, row in skipped_df.iterrows():
# #         tree_location = [row['tree_lat'], row['tree_lon']]
# #         road_location = [row['road_lat'], row['road_lon']]
# #         reason = row['Reason']  # Assuming there's a 'Reason' column in your skipped points CSV
        
# #         # Display skipped tree points in greenish-grey
# #         folium.CircleMarker(location=tree_location, radius=3, color='#789E9E', fill=True, fill_color='#789E9E', popup=f'Skipped Tree: {reason}').add_to(m)
# #         folium.CircleMarker(location=road_location, radius=3, color='grey', fill=True, fill_color='grey', popup='Skipped Road Point').add_to(m)
        
# #         # Connect skipped tree and road points with a grey line
# #         folium.PolyLine(locations=[tree_location, road_location], color='grey', weight=1, dash_array='5, 5').add_to(m)

# #     # Save the map to an HTML file
# #     m.save(OUTPUT_FILE)
# #     print(f"Map saved to {OUTPUT_FILE}")

# # # import folium
# # # import pandas as pd

# # # def visualize_tree_to_road(csv_path):
# # #     """
# # #     Creates an HTML file visualizing tree points in green, road points in blue, 
# # #     and arrows from each road point to the corresponding tree point on an OpenStreetMap.
    
# # #     Args:
# # #         csv_path (str): Path to the CSV file containing tree and road points data.
# # #     """
# # #     # Load data from CSV
# # #     df = pd.read_csv(csv_path)
    
# # #     # Create a map centered around the average location
# # #     map_center = [df['tree_lat'].mean(), df['tree_lon'].mean()]
# # #     m = folium.Map(location=map_center, zoom_start=15, tiles='OpenStreetMap')
    
# # #     # Add points and lines to the map
# # #     for index, row in df.iterrows():
# # #         tree_location = [row['tree_lat'], row['tree_lon']]
# # #         road_location = [row['road_lat'], row['road_lon']]
        
# # #         # Draw the tree point in green
# # #         folium.CircleMarker(
# # #             location=tree_location,
# # #             radius=3,
# # #             color='green',
# # #             fill=True,
# # #             fill_color='green',
# # #             fill_opacity=1,
# # #             popup='Tree Point'
# # #         ).add_to(m)
        
# # #         # Draw the road point in blue
# # #         folium.CircleMarker(
# # #             location=road_location,
# # #             radius=3,
# # #             color='blue',
# # #             fill=True,
# # #             fill_color='blue',
# # #             fill_opacity=1,
# # #             popup='Road Point'
# # #         ).add_to(m)
        
# # #         # Draw a line for the arrow
# # #         folium.PolyLine(locations=[road_location, tree_location], color='black', weight=1).add_to(m)
        
# # #         # Unfortunately, Folium does not support arrows directly.
# # #         # For an arrow, you might need to consider a workaround or simply use a line as done here.
    
# # #     # Save the map to an HTML file
# # #     m.save('RetrieveGSV/RoadPointsGeneration/OverpassData/camera_to_tree_visualization.html')
# # #     print("Map saved to RetrieveGSV/RoadPointsGeneration/OverpassData/camera_to_tree_visualization.html")

# # # import folium
# # # import pandas as pd

# # # OUTPUT_FILE = 'map_visualization.html'

# # # def visualize_tree_to_road(csv_path, skipped_csv_path):
# # #     """
# # #     Visualizes tree points and road points on a map. Tree points are in green, 
# # #     road points in blue, and skipped points in red with arrows from each road point 
# # #     to the corresponding tree point. Skipped points are marked with a different color 
# # #     to denote they were not processed for some reason.

# # #     Args:
# # #         csv_path (str): Path to the CSV file containing tree and road points data.
# # #         skipped_csv_path (str): Path to the CSV file containing skipped points data.
# # #     """
# # #     # Load data from CSV
# # #     df = pd.read_csv(csv_path)
# # #     skipped_df = pd.read_csv(skipped_csv_path)

# # #     # Create a map centered around the average location
# # #     map_center = [df['tree_lat'].mean(), df['tree_lon'].mean()]
# # #     m = folium.Map(location=map_center, zoom_start=15)

# # #     # Add processed points and lines to the map
# # #     for index, row in df.iterrows():
# # #         tree_location = [row['tree_lat'], row['tree_lon']]
# # #         road_location = [row['road_lat'], row['road_lon']]
        
# # #         folium.CircleMarker(location=tree_location, radius=3, color='green', fill=True, fill_color='green').add_to(m)
# # #         folium.CircleMarker(location=road_location, radius=3, color='blue', fill=True, fill_color='blue').add_to(m)
# # #         folium.PolyLine(locations=[road_location, tree_location], color='black', weight=1).add_to(m)
    
# # #     # Add skipped points to the map in red
# # #     for index, row in skipped_df.iterrows():
# # #         tree_location = [row['tree_lat'], row['tree_lon']]
# # #         reason = row['Reason']  # Assuming there's a 'Reason' column in your skipped points CSV
        
# # #         folium.CircleMarker(
# # #             location=tree_location,
# # #             radius=3,
# # #             color='red',
# # #             fill=True,
# # #             fill_color='red',
# # #             fill_opacity=1,
# # #             popup=f'Skipped: {reason}'
# # #         ).add_to(m)

# # #     # Save the map to an HTML file
# # #     m.save(OUTPUT_FILE)
# # #     print(f"Map saved to {OUTPUT_FILE}")

# # # import folium
# # # import pandas as pd

# # # OUTPUT_FILE = 'map_visualization.html'

# # # def visualize_tree_to_road(csv_path, skipped_csv_path):
# # #     """
# # #     Visualizes tree points and road points on a map. Tree points are in green, 
# # #     road points in blue, and skipped points are displayed with tree points and 
# # #     their corresponding road points both in grey, connected by a grey line, 
# # #     indicating that these points were skipped during processing.

# # #     Args:
# # #         csv_path (str): Path to the CSV file containing tree and road points data.
# # #         skipped_csv_path (str): Path to the CSV file containing skipped points data.
# # #     """
# # #     # Load data from CSV
# # #     df = pd.read_csv(csv_path)
# # #     skipped_df = pd.read_csv(skipped_csv_path)

# # #     # Create a map centered around the average location
# # #     map_center = [df['tree_lat'].mean(), df['tree_lon'].mean()]
# # #     m = folium.Map(location=map_center, zoom_start=15)

# # #     # Add processed points and lines to the map
# # #     for index, row in df.iterrows():
# # #         tree_location = [row['tree_lat'], row['tree_lon']]
# # #         road_location = [row['road_lat'], row['road_lon']]
        
# # #         folium.CircleMarker(location=tree_location, radius=3, color='green', fill=True, fill_color='green').add_to(m)
# # #         folium.CircleMarker(location=road_location, radius=3, color='blue', fill=True, fill_color='blue').add_to(m)
# # #         folium.PolyLine(locations=[road_location, tree_location], color='black', weight=1).add_to(m)
    
# # #     # Add skipped points to the map in grey
# # #     for index, row in skipped_df.iterrows():
# # #         tree_location = [row['tree_lat'], row['tree_lon']]
# # #         road_location = [row['road_lat'], row['road_lon']]
# # #         reason = row['Reason']  # Assuming there's a 'Reason' column in your skipped points CSV
        
# # #         # Display both tree and road points for skipped entries in grey
# # #         folium.CircleMarker(location=tree_location, radius=3, color='grey', fill=True, fill_color='grey', popup=f'Skipped Tree: {reason}').add_to(m)
# # #         folium.CircleMarker(location=road_location, radius=3, color='grey', fill=True, fill_color='grey', popup='Skipped Road Point').add_to(m)
        
# # #         # Connect skipped tree and road points with a grey line
# # #         folium.PolyLine(locations=[tree_location, road_location], color='grey', weight=1, dash_array='5, 5').add_to(m)

# # #     # Save the map to an HTML file
# # #     m.save(OUTPUT_FILE)
# # #     print(f"Map saved to {OUTPUT_FILE}")

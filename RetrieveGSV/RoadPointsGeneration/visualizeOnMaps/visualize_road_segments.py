import folium
import random

OUTPUT_FILE = 'RetrieveGSV/RoadPointsGeneration/visualizeOnMaps/road_segments.html'


def generate_random_color():
    """Generate random color for visualization."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def visualize_road_segments(road_data):
    print("Visualizing road segments...")
    map_center = [51.987876, 4.366472]  # Center of Delft

    # Initialize the folium map centered at the calculated center
    m = folium.Map(location=map_center, zoom_start=13, tiles='OpenStreetMap')

    # Plot each road with a different color
    for element in road_data['elements']:
        if element['type'] == 'way' and 'geometry' in element:
            line_coords = [(node['lat'], node['lon']) for node in element['geometry']]
            road_color = generate_random_color()

            folium.PolyLine(locations=line_coords, color=road_color, weight=2.5).add_to(m)
            
            # Additionally, add a marker for each point defining the way
            for coord in line_coords:
                folium.CircleMarker(location=coord, radius=3, color=road_color, fill=True).add_to(m)
    
    # Save the map to an HTML file
    m.save(OUTPUT_FILE)
    print(f"Map of overpass way data saved to {OUTPUT_FILE}")
import folium
import random


def generate_random_color():
    """Generate random color for visualization."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def visualize_roads(road_data):
    map_center = [51.987876, 4.366472]  # Center of Delft

    # Initialize the folium map centered at the calculated center
    m = folium.Map(location=map_center, zoom_start=13, tiles='OpenStreetMap')

    # Plot each road with a different color
    for element in road_data['elements']:
        if element['type'] == 'way' and 'geometry' in element:
            line_coords = [(node['lat'], node['lon']) for node in element['geometry']]
            folium.PolyLine(locations=line_coords, color=generate_random_color(), weight=2.5).add_to(m)
    
    # Save the map to an HTML file
    m.save('TreehealthDataset/roads_visualization.html')
    print("Map of overpass way data saved to 'TreehealthDataset/roads_visualization.html'")
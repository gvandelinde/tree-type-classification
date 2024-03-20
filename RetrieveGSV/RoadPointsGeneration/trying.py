import geopandas as gpd
import matplotlib.pyplot as plt


# Replace 'your_shapefile_path.shp' with the path to your .shp file
shapefile_path = 'TreehealthDataset/bomen.shp'
gdf = gpd.read_file(shapefile_path)

# Print the first few rows of the GeoDataFrame
print("Column Headers:", gdf.columns)

print(gdf.head())

gdf.plot()
plt.show()

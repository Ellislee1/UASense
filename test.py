import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

# Given bounding box in (lon, lat) pairs
poly = np.array(
    [
        [-74.35067099968856, 40.95038628235707],
        [-74.35134665517963, 40.49961892079759],
        [-73.64880132727227, 40.50053835401126],
        [-73.65109221855438, 40.94965116545227],
        [-74.35067099968856, 40.95038628235707],
    ]
)

# Convert to a Shapely Polygon
polygon = Polygon(poly)

# Determine the bounding box width and height in meters
bounds_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
bounds_gdf = bounds_gdf.to_crs("EPSG:3395")  # Project to a metric CRS

print(bounds_gdf)
minx, miny, maxx, maxy = bounds_gdf.total_bounds
width = maxx - minx
height = maxy - miny

# Define the size of the squares
square_size = 300  # in meters

# Calculate the number of squares in each dimension
num_squares_x = int(np.ceil(width / square_size))
num_squares_y = int(np.ceil(height / square_size))

# Create a list to store the bounding boxes
bounding_boxes = []

# Create a grid of points based on the number of squares
for i in range(num_squares_x):
    for j in range(num_squares_y):
        x = minx + i * square_size
        y = miny + j * square_size

        # Create a bounding box for each grid point
        box = Polygon([(x, y), (x + square_size, y), (x + square_size, y + square_size), (x, y + square_size)])

        # Append the bounding box to the list
        bounding_boxes.append(box)

# Convert the bounding boxes back to (lon, lat) coordinates
bounding_boxes_lonlat = gpd.GeoDataFrame(geometry=bounding_boxes, crs="EPSG:3395")
bounding_boxes_lonlat = bounding_boxes_lonlat.to_crs("EPSG:4326")  # Project back to WGS84

# Convert the bounding boxes to a list of (lon, lat) pairs
bounding_boxes_lonlat_list = np.array(
    [(box.exterior.xy[0].tolist(), box.exterior.xy[1].tolist()) for box in bounding_boxes_lonlat.geometry]
)

print(bounding_boxes_lonlat_list.shape)

# Now, bounding_boxes_lonlat_list contains the list of bounding boxes in (lon, lat) pairs

plt.plot(poly[:, 0], poly[:, 1], c="green")

for box in bounding_boxes_lonlat_list:
    plt.plot(box[0], box[1], c="red", alpha=0.5)

plt.show()

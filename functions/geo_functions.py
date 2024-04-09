import math
import typing
from time import perf_counter

import numba as nb
import numpy as np
from scipy.spatial.distance import cdist
from shapely import LineString, Polygon


@nb.njit
def haversine_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
    """Calculates the haversine distance between two coordinates using the Haversine formula.

    Args:
        coord1 (np.ndarray): The coordinates of the first point as a numpy array [lon1, lat1, ...].
        coord2 (np.ndarray): The coordinates of the second point as a numpy array [lon2, lat2, ...].

    Returns:
        float: The haversine distance between the two coordinates.

    Example:
        ```python
        coord1 = np.array([0, 0])
        coord2 = np.array([1, 1])

        distance = haversine_distance(coord1, coord2)
        print(distance)
        ```
    """
    lon1, lat1 = np.radians(coord1[:2])
    lon2, lat2 = np.radians(coord2[:2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


@nb.njit
def calculate_initial_compass_bearing(lon1, lat1, lon2, lat2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)

    initial_bearing = math.atan2(y, x)

    return math.degrees(initial_bearing) % 360


def calculate_vertiport_points(lon, lat, r, a, n):
    # Convert latitude and longitude from degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    points = []

    for i in range(n):
        # Calculate the angle for the current point
        angle = 2 * math.pi * (i / n)

        # Calculate the distance along the circle's circumference
        d = r  # Since r is in kilometers

        # Use Haversine formula to find the new coordinates
        new_lat = math.asin(
            math.sin(lat_rad) * math.cos(d / 6371.0) + math.cos(lat_rad) * math.sin(d / 6371.0) * math.cos(angle)
        )
        new_lon = lon_rad + math.atan2(
            math.sin(angle) * math.sin(d / 6371.0) * math.cos(lat_rad),
            math.cos(d / 6371.0) - math.sin(lat_rad) * math.sin(new_lat),
        )

        # Convert back to degrees
        new_lat = math.degrees(new_lat)
        new_lon = math.degrees(new_lon)

        points.append((new_lon, new_lat, a))

    return np.array(points)


def initilise_start_goal_vertiports(vertiports: dict, start: str = None, goal: str = None) -> typing.Tuple[str, str]:
    """Initializes the start and goal vertiports for the FMT* algorithm.

    Args:
        vertiports (dict): A dictionary of vertiports.
        start (str, optional): The name of the start vertiport. Defaults to None.
        goal (str, optional): The name of the goal vertiport. Defaults to None.

    Returns:
        Tuple[str, str]: A tuple containing the names of the start and goal vertiports.

    Example:
        ```python
        vertiports = {
            "A": (0, 0),
            "B": (1, 1),
            "C": (2, 2)
        }
        start = "A"
        goal = "B"

        result = initilise_start_goal_vertiports(vertiports, start, goal)
        print(result)
        ```
    """

    if start is None or start not in vertiports:
        print("No point set for start, choosing a random point...")
        start = None

        while start is None:
            s = np.random.choice(list(vertiports.keys()))
            if s != goal:
                start = s

    if goal is None or goal not in vertiports:
        print("No point set for goal, choosing a random point...")
        goal = None

        while goal is None:
            e = np.random.choice(list(vertiports.keys()))
            if e != start:
                goal = e

    return (start, goal)


@nb.njit
def compute_distance_matrix(index_matrix, distance_matrix, distance_matrix_full):
    # sourcery skip: use-itertools-product
    num_nodes = len(index_matrix)
    for i in range(num_nodes):
        for j in range(num_nodes):
            index_i, index_j = index_matrix[i], index_matrix[j]
            if index_i != -1 and index_j != -1:
                distance_matrix_full[i, j] = distance_matrix[index_i, index_j]
    return distance_matrix_full


# Call the function in your code


def build_distance_matrix(running_nodes: np.ndarray, nodes: np.ndarray, method: str) -> np.ndarray:
    print("Generating distance matrix...")
    s = perf_counter()

    unique_nodes = np.unique(running_nodes[:, :2], axis=0)
    distance_matrix = cdist(unique_nodes, unique_nodes, method)

    # return nodes, running_nodes, distance_matrix

    node_to_index = {tuple(node): index for index, node in enumerate(unique_nodes)}
    num_nodes = len(running_nodes)

    # Create an index matrix for quick lookups
    index_matrix = np.array([node_to_index.get(tuple(node[:2]), -1) for node in running_nodes])

    # Create the full distance matrix with vectorized operations
    distance_matrix_full = np.zeros((num_nodes, num_nodes))

    distance_matrix_full = compute_distance_matrix(index_matrix, distance_matrix, distance_matrix_full)
    print("Compiling distance matrix...")

    print(f"Distance matrix generated in {round(perf_counter() - s, 2)} seconds, with {num_nodes} nodes.")
    return nodes, running_nodes, distance_matrix_full


def evaluate_intersection(line: LineString, point1: np.ndarray, point2: np.ndarray, poly: Polygon, h: float) -> bool:
    """Evaluates the intersection between a line segment and a polygon in 2D or 3D space.

    Args:
        self: The instance of the class.
        line (LineString): The line segment represented as a Shapely LineString object.
        point1 (np.ndarray): The coordinates of the first point of the line segment as a numpy array [lon1, lat1, alt1].
        point2 (np.ndarray): The coordinates of the second point of the line segment as a numpy array [lon2, lat2, alt2].
        poly (Polygon): The polygon represented as a Shapely Polygon object.
        h (float): The threshold altitude value.

    Returns:
        bool: True if the line segment does not intersect with the polygon , False otherwise.

    Raises:
        ValueError: Raised when the type of the intersection result is unexpected.

    Example:
        ```python
        fmt = FMTStar()
        line = LineString([(0, 0), (1, 1)])
        point1 = np.array([0, 0, 0])
        point2 = np.array([1, 1, 0])
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        h = 5

        result = fmt.evaluate_intersection(line, point1, point2, poly, h)
        print(result)
        ```
    """  # noqa:E501

    point_intersections = line.intersection(poly)
    point1 = (x1, y1, z1) = point1[0], point1[1], point1[2]
    point2 = (x2, y2, z2) = point2[0], point2[1], point2[2]

    if point_intersections.geom_type == "LineString":
        return evaluate_intersection_height(np.array(point_intersections.coords), point1, point2, h)

    elif point_intersections.geom_type == "MultiLineString":
        return all(
            evaluate_intersection_height(np.array(linestring.coords), point1, point2, h)
            for linestring in point_intersections.geoms
        )
    else:
        raise ValueError(f"Unexpected type: {point_intersections.geom_type}")


def evaluate_intersection_height(
    line: np.ndarray,
    point1: typing.Tuple[float, float, float],
    point2: typing.Tuple[float, float, float],
    h,
) -> bool:
    """Evaluates the intersection height between a line segment and a set of points.

    Args:
        line (np.ndarray): The line segment represented as an array of points.
        point1 (Tuple[float, float, float]): The coordinates of the first point of the line segment as a tuple (x1, y1, z1).
        point2 (Tuple[float, float, float]): The coordinates of the second point of the line segment as a tuple (x2, y2, z2).
        h: The threshold altitude value.

    Returns:
        bool: True if the line segment does not intersect with any point below the threshold altitude, False otherwise.

    Example:
        ```python
        line = np.array([(0, 0), (1, 1), (2, 2)])
        point1 = (0, 0, 0)
        point2 = (1, 1, 0)
        h = 5

        result = evaluate_intersection_height(line, point1, point2, h)
        print(result)
        ```
    """  # noqa:E501

    if len(line) == 0:
        return True

    x1, y1, z1 = point1
    x2, y2, z2 = point2

    for i_point in line:
        x, y = i_point
        t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
        altitude = z1 + t * (z2 - z1)

        if altitude <= h + 50:
            return False

    return True

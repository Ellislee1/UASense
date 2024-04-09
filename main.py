import time

import matplotlib.pyplot as plt
import numpy as np
import simplekml

from functions.geo_functions import haversine_distance
from geojson_parser import read_file
from map_plot import Map
from methods import FMT

x = np.random.randint(0, 99999)
print(x)
# np.random.seed(121234)

IS3D = True
# PATH = "airspaces/Blade+Operation+Zone.geojson"
PATH = "airspaces/NYC+Zones.geojson"
# PATH = "airspaces/DFW+AIRSPACE.geojson"
CRUISE = 60  # (m/s)
MAX_CLIMB = 1000  # (ft/min)
MAX_DIST = 10  # (km)
MIN_DIST = 0  # (km)
N_NODES = 250

# Load the environment data
environment = read_file(PATH)
m = Map(environment)
# m.publish()
# raise Exception("Stop here")


def save_3d_path_to_kml(path, output_file):
    kml = simplekml.Kml()

    linestring = kml.newlinestring(name="3D Path", altitudemode=simplekml.AltitudeMode.absolute)

    linestring.style.linestyle.color = simplekml.Color.changealphaint(
        100, simplekml.Color.cyan
    )  # Set line color with 0.3 opacity
    linestring.style.linestyle.width = 5  # Set the line width to 5

    for lon, lat, alt_ft in path:
        # Convert altitude from feet to meters
        alt_meters = alt_ft * 0.3048
        linestring.coords.addcoordinates([(lon, lat, alt_meters)])

    kml.save(output_file)
    print(f"KML file saved as {output_file}")


def get_max_climb_profile(v_path, time):
    max_climb = [0]
    times = [0]
    for i in range(1, len(v_path)):
        t0 = times[-1]
        y0 = v_path[i - 1]
        y1 = v_path[i]
        t1 = t0 + (y1 - y0) / (MAX_CLIMB if y1 > y0 else -MAX_CLIMB)

        if y1 < y0:
            t1 = time[i] - (t1 - t0)
            times.append(t1)
            max_climb.append(y0)
            times.append(time[i])
            max_climb.append(y1)

        else:
            times.append(t1)
            max_climb.extend((y1, y1))
            times.append(time[i])
    return times, max_climb


def plot_vProfile(_path):
    time = [0]
    altitude = [0]
    distance = [0]  # To store the cumulative distance

    for i in range(1, len(_path)):
        # Calculate time based on distance and speed in minutes
        time.append((time[-1] + (haversine_distance(_path[i - 1, :2], _path[i, :2]) * 1000 / CRUISE) / 60))

        # Calculate cumulative distance
        distance.append(distance[-1] + haversine_distance(_path[i - 1, :2], _path[i, :2]))

        altitude.append(_path[i, -1])

    # Create a new figure with twin x and y axes
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    ax3 = ax1.twinx()

    ax1.plot(time, altitude, label="Minimum Climb Profile")

    # Plot points at the locations of the nodes
    ax1.scatter(time[1:-1], altitude[1:-1], color="blue", s=30)

    times, max_climb = get_max_climb_profile(altitude, time)
    ax1.plot(times, max_climb, label=f"Max Climb Profile ({MAX_CLIMB} ft/min)")
    ax1.scatter(times, max_climb, color="orange", s=30)

    ax1.scatter(time[0], altitude[0], color="green", label="Start", s=30)
    ax1.scatter(time[-1], altitude[-1], color="red", label="End", s=30)

    ax2.plot(distance, altitude, alpha=0)  # Plot distance on the secondary x-axis
    ax2.set_xlabel("Distance (km)")

    # Define a conversion factor for altitude from feet to meters
    feet_to_meters = 0.3048

    # Plot altitude in meters on the secondary y-axis
    ax3.plot(time, [a * feet_to_meters for a in altitude], label="Altitude (m)")
    ax3.set_ylabel("Altitude (m)")
    dist = sum(haversine_distance(_path[i - 1, :2], _path[i, :2]) for i in range(1, len(_path)))
    origional_dist = haversine_distance(_path[0, :2], _path[-1, :2])

    print(
        f"Total dist: {round(dist, 2)}km (Direct dist: {round(origional_dist, 2)}km). "
        + f"Total time en route: {int(round(time[-1], 2) // 1):02}:{int(round(time[-1] % 1 * 60, 2)):02}. "
        + f"This route is {round(((dist - origional_dist) / origional_dist) * 100, 2)}% "
        + f"({round((dist-origional_dist),2)}km) longer than the direct route."
    )

    # Add labels and legend
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Altitude (ft)")
    ax1.legend()

    plt.show()


path_find = FMT(environment, cruise_speed=CRUISE, climb_rate=MAX_CLIMB)
path_find.initilise_route(n_nodes=50, start="ESSEX COUNTY", goal="BELMONT PARK")
# path_find.initilise_route(start="Quogue / Westhampton", goal="Montauk")
path_find.populate_nodes(N=N_NODES, is3D=IS3D)
s = time.perf_counter()
path_find.find_path(min_dist=MIN_DIST, max_dist=MAX_DIST, is3D=IS3D)
print(f"Time taken: {round(time.perf_counter()-s,2)}s")
path = path_find.build_path_object()

m.add_elements(path_find.build_all_path_objects(), "multi_path")
m.add_elements(path_find.nodes, _type="nodes")
m.add_startgoal(path_find.vertiports[path_find.start], path_find.vertiports[path_find.goal])
m.add_elements(path, "path")

save_3d_path_to_kml(path, "out/path.kml")
m.publish()

plot_vProfile(path)

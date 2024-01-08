import folium
import warnings
import numpy as np


def generate_zone_info(name, zone):
    # iframe = folium.IFrame('Account#:' + str(row.loc['ACCT']) + '<br>' + 'Name: ' + row.loc['NAME'] + '<br>' + 'Terr#: ' + str(row.loc['TERR']))
    iframe = folium.IFrame(
        f"<h4><b>{name}</b></h4>\n<b>Operating Altitudes</b>: 0ft-{zone[:,-1].max()}ft (0m-{round(zone[:,-1].max()/3.281,1)}m)"
    )
    return folium.Popup(iframe, min_width=200, max_width=200)


class Map:
    def __init__(self, environment):
        self.sector = environment["sector"]
        self.tczs = environment["TCZs"]
        self.vertiports = environment["vertiports"]
        self.frzs = environment["FRZs"]

        self.center_lat, self.center_lon = (
            self.sector[:, 1].min() + self.sector[:, 1].max()
        ) / 2, (self.sector[:, 0].min() + self.sector[:, 0].max()) / 2

        self.refresh()

    def populate_base(self):
        # Add airspace bounds as a green polygon
        folium.Polygon(locations=self.sector[:, [1, 0]], color="green").add_to(
            self.map_area
        )  # Swap lat and lon

        # Add TCZs as purple lines
        for name, tcz in self.tczs.items():
            folium.Polygon(
                locations=tcz[:-1, [1, 0]],  # Swap lat and lon
                color="purple",
                fill=True,  # Fill the polygon
                fill_color="purple",  # Fill color
                fill_opacity=0.1,
                popup=generate_zone_info(name, tcz),
            ).add_to(self.map_area)

        for name, frz in self.frzs.items():
            folium.Polygon(
                locations=frz[:-1, [1, 0]],  # Swap lat and lon
                color="red",
                fill=True,  # Fill the polygon
                fill_color="frd",  # Fill color
                fill_opacity=0.1,
                popup=generate_zone_info(name, frz),
            ).add_to(self.map_area)

        # Add vertiports as blue star markers
        for name, vertiport in self.vertiports.items():
            folium.Marker(
                location=vertiport[[1, 0]],
                icon=folium.Icon(color="blue", icon="star"),
                tooltip=name,
            ).add_to(
                self.map_area
            )  # Swap lat and lon

    def add_elements(self, element, _type: str = None):
        if _type is None:
            warnings.warn("No type assigned, skipping...")
        elif _type.lower() == "nodes":
            unique_nodes = np.unique(element[:, :2], axis=0)
            
            for node in unique_nodes:
                folium.CircleMarker(
                    location=node[[1, 0]],
                    radius=3,
                    color="black",
                    fill_color="black",
                    fill_opacity=1,
                ).add_to(self.map_area)
        elif _type.lower() == "path":
            folium.PolyLine(locations=element[:, [1, 0]], weight=5).add_to(
                self.map_area
            )

            for i in range(1, len(element) - 1):
                node = element[i]
                folium.CircleMarker(
                    location=node[[1, 0]],
                    radius=5,
                    color="orange",
                    fill_color="orange",
                    fill_opacity=1,
                    tooltip=f"Node {i}, Alt: {round(node[-1],2)}ft ({round(node[-1]/3.281,1)}m))",
                ).add_to(self.map_area)

        elif _type.lower() == "multi_path":
            added = set()
            for path in element:
                folium.PolyLine(
                    locations=path[:, [1, 0]], color="black", dash_array="10", weight=2
                ).add_to(self.map_area)

                for node in path:
                    if tuple(node) in added:
                        continue

                    folium.CircleMarker(
                        location=node[[1, 0]],
                        radius=3,
                        color="black",
                        fill_color="black",
                        fill_opacity=1,
                    ).add_to(self.map_area)
                    
                    added.add(tuple(node))

        else:
            warnings.warn(f"No type {_type} exists, skipping...")

    def add_startgoal(self, start, goal):
        self.add_formatted_marker(
            [start, goal],
            [
                {
                    "radius": 10,
                    "color": "green",
                    "fill_color": "green",
                    "fill_opacity": 0.5,
                },
                {
                    "radius": 10,
                    "color": "blue",
                    "fill_color": "blue",
                    "fill_opacity": 0.5,
                },
            ],
        )

    def add_formatted_marker(self, element, _format):
        for i, node in enumerate(element):
            folium.CircleMarker(
                location=node[[1, 0]],
                radius=_format[i]["radius"],
                color=_format[i]["color"],
                fill_color=_format[i]["fill_color"],
                fill_opacity=_format[i]["fill_opacity"],
            ).add_to(self.map_area)

    def publish(self, path: str = "out/map.html"):
        print("Publishing map...")
        self.map_area.save(path)
        print(f"Done publishing file '{path}'.")

    def refresh(self):
        print("Building map...")
        self.map_area = folium.Map(
            location=[self.center_lat, self.center_lon], zoom_start=11
        )
        self.populate_base()
        print("Done building map.")

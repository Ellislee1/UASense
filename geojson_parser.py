import json

import numpy as np


def read_file(path: str) -> dict:
    sector = None
    terminal_control_zones = {}
    vertiports = {}
    frzs = {}

    with open(path) as file:
        data = json.load(file)

        for item in data.get("features", []):
            name = item["properties"]["name"]
            geom = item["geometry"]
            _type = geom["type"]

            coordinates = np.array(geom["coordinates"])

            if name.lower() == "airspace":
                if sector is not None:
                    raise ValueError("Sector area already exists. Multiple sector areas cannot exist.")
                sector = coordinates[0]
            elif geom["type"].lower() == "point":
                if name in vertiports:
                    raise IndexError(
                        f"A vertiport with name {name} already exists. Multiple vertiports with the same"
                        + "name cannot exist."
                    )
                vertiports[name] = coordinates
            elif name.startswith("TCZ_"):
                if name in terminal_control_zones:
                    raise IndexError(
                        f"A TCZ with name {name} already exists. Multiple TCZs with the same name cannot exist."
                    )
                if _type == "LineString":
                    terminal_control_zones[name] = coordinates
                else:
                    terminal_control_zones[name] = coordinates[0]
            elif name.startswith("FRZ_") or name.startswith("NFZ_"):
                if name in frzs:
                    raise IndexError(
                        f"A FRZ with name {name} already exists. Multiple FRZs with the same name cannot exist."
                    )
                frzs[name] = coordinates[0]

    if sector is None:
        raise ValueError("A sector does not exist. A sector must exist and have valid bounds.")

    if not vertiports:
        raise ValueError("No vertiports found. Vertiports must exist")

    min_bounds = sector.min(axis=0)
    min_bounds[2] = 0
    max_bounds = sector.max(axis=0)

    for tcz in terminal_control_zones:
        terminal_control_zones[tcz] = np.clip(terminal_control_zones[tcz], min_bounds, max_bounds)

    for frz in frzs:
        frzs[frz] = np.clip(frzs[frz], min_bounds, max_bounds)

    return {
        "sector": sector,
        "vertiports": vertiports,
        "TCZs": terminal_control_zones,
        "FRZs": frzs,
    }


# print(read_file('airspaces/NYC+Zones.geojson'))

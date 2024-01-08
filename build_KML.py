import json
import geopandas as gpd
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import folium
import numpy as np
import simplekml

file = 'out/output.geojson'
kml = simplekml.Kml()

json_data = json.load(open(file))

bbox = json_data['bbox']
bbox_array = np.array([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3]),(bbox[0], bbox[1])])
bbox_array_alt = np.array([(bbox[0], bbox[1],762), (bbox[2], bbox[1],762), (bbox[2], bbox[3],762), (bbox[0], bbox[3],762),(bbox[0], bbox[1],762)])
bbox = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])

features = json_data['features']

def add_polygon(kml, coords, name, color, line_width=2, fill=True, fillColor=simplekml.Color.changealphaint(100, simplekml.Color.gray), extrude=True):
    polygon = kml.newpolygon(name=name)
    polygon.altitudemode = simplekml.AltitudeMode.absolute
    polygon.outerboundaryis = coords  # Coordinates: (longitude, latitude)
    polygon.extrude = 1 if extrude else 0
    # Set the color and width of the outline
    polygon.style.polystyle.outline = 1  # Display the outline
    polygon.style.polystyle.fill = 1 if fill else 0  # Display the fill
    polygon.style.linestyle.color = color  # Set the color of the outline
    polygon.style.linestyle.width = 5  # Adjust the width of the outline as needed
    polygon.style.polystyle.color = fillColor

map_area = folium.Map(
            location=np.median(bbox_array, axis=0)[[1,0]], zoom_start=11
        )

add_polygon(kml, bbox_array_alt, "NYC", simplekml.Color.green, line_width=5, fill=False)






folium.Polygon(locations=np.array(list(zip(*bbox.exterior.xy)))[:,[1,0]], color="green").add_to(
map_area)  # Swap lat and lon



plt.plot(*bbox.exterior.xy)
point_types = set()
for feature in features:
    if feature['geometry']['type'] == 'Point':
        type = feature['properties']['display']['detailedCategory']
        coords = feature['geometry']['coordinates']
        if not bbox.contains(Point(coords)):
            continue
        
        match type:
            case 'Heliport':
                tool_tip = f'Heliport: {feature["properties"]["name"]}'
                plt.plot(coords[0], coords[1], 'bo')
                folium.Marker(location=np.array(coords)[[1,0]], tooltip=tool_tip).add_to(map_area)
                # folium.Circle(location=np.array(coords)[[1,0]], radius=float(feature['properties']['radius'])/3.281, color='blue', fill=True, fill_color='blue', fill_opacity=0.2).add_to(map_area)

                kml.newpoint(name=feature["properties"]["name"], coords=[(coords[0], coords[1], 0)])
            case _:
                pass


        
    elif feature['geometry']['type'] == 'Polygon':
        match feature['properties']['display']['detailedCategory']:
            case 'Stadium':
                coords = np.array(feature['geometry']['coordinates'][0])
                hazard_factor = feature['properties']['hazardFactor']
                name = feature['properties']['name']
                
                plt.plot(*coords.T, c='purple')
                
                folium.Polygon(locations=coords[:,[1,0]], color="purple", fill=True, popup=f"<h4><b>{name}-Stadium</b></h4>\n<ul><li>Hazard Factor:{hazard_factor}</li></ul>").add_to(
                    map_area)
                
            case 'No-Drone Zone':
                coords = np.array(feature['geometry']['coordinates'][0])
                
                alt_floor = feature['properties']['altitudeFloor']['meters']
                alt_ceiling = feature['properties']['altitudeCeiling']['meters']
                hazard_factor = feature['properties']['hazardFactor']
                name = feature['properties']['name']
                
                plt.plot(*coords.T, c='r')
                folium.Polygon(locations=coords[:,[1,0]], color="red", fill=True, popup=f"<h4><b>{name}-NDZ</b></h4>\n<ul><li>Floor:{alt_floor}m</li><li>Ceil:{alt_ceiling}m</li><li>Hazard Factor:{hazard_factor}</li></ul>").add_to(
                    map_area)
            case 'Us National Park':
                coords = np.array(feature['geometry']['coordinates'][0])
                hazard_factor = feature['properties']['hazardFactor']
                name = feature['properties']['name']
                plt.plot(*coords.T, c='magenta')
                
                
                folium.Polygon(locations=coords[:,[1,0]], color="magenta", fill=True, popup=f"<h4><b>{name}-National Park</b></h4>\n<ul><li>Hazard Factor:{hazard_factor}</li></ul>").add_to(
                    map_area)
                
                new_coords = []
                for coord in coords:
                    new_coords.append((coord[0], coord[1], 762))
                add_polygon(kml, new_coords, name, simplekml.Color.magenta,fill=True, fillColor=simplekml.Color.changealphaint(50, simplekml.Color.magenta))
                
            case 'Maximum permitted altitude':
                coords = np.array(feature['geometry']['coordinates'][0])
                alt_floor = feature['properties']['altitudeFloor']['meters']
                alt_ceiling = feature['properties']['altitudeCeiling']['meters']
                hazard_factor = feature['properties']['hazardFactor']
                name = feature['properties']['name']
                
                if alt_floor == 0 and alt_ceiling == 0:
                    c = 'red'
                    kc = simplekml.Color.red
                elif 0 < alt_ceiling<=31:
                    c='orange'
                    kc = simplekml.Color.orange
                elif 31 < alt_ceiling <= 61:
                    c='yellow'
                    kc = simplekml.Color.yellow
                elif 61 < alt_ceiling <= 92:
                    c='blue'
                    kc = simplekml.Color.blue
                else:
                    c='green'
                    kc = simplekml.Color.green
                
                plt.plot(*coords.T, c=c)
                folium.Polygon(locations=coords[:,[1,0]], color=c, fill=True, popup=f"<h4><b>{name}-Alt Restriction</b></h4>\n<ul><li>Floor:{alt_floor}m</li><li>Ceil:{alt_ceiling}m</li><li>Hazard Factor:{hazard_factor}</li></ul>").add_to(
                    map_area)
                
                new_coords = []
                for coord in coords:
                    new_coords.append((coord[0], coord[1], alt_ceiling))
                add_polygon(kml, new_coords, name, kc,fill=True, fillColor=simplekml.Color.changealphaint(70, kc))
                
            # case 'Class  D':
            #     coords = np.array(feature['geometry']['coordinates'][0])
            #     alt_floor = feature['properties']['altitudeFloor']['meters']
            #     alt_ceiling = feature['properties']['altitudeCeiling']['meters']
            #     hazard_factor = feature['properties']['hazardFactor']
            #     name = feature['properties']['name']
                
            #     c='grey'
            #     plt.plot(*coords.T, c=c)
            #     folium.Polygon(locations=coords[:,[1,0]], color=c, fill=True, popup=f"<h4><b>{name}-Class D</b></h4>\n<ul><li>Floor:{alt_floor}m</li><li>Ceil:{alt_ceiling}m</li><li>Hazard Factor:{hazard_factor}</li></ul>").add_to(
            #         map_area)
                
            #     new_coords = []
            #     for coord in coords:
            #         new_coords.append((coord[0], coord[1], alt_ceiling))
            #     # add_polygon(kml, new_coords, name, simplekml.Color.gray,fill=False, fillColor=simplekml.Color.changealphaint(70, simplekml.Color.gray))
            
            # case 'Class  B':
            #     coords = np.array(feature['geometry']['coordinates'][0])
            #     alt_floor = feature['properties']['altitudeFloor']['meters']
            #     alt_ceiling = feature['properties']['altitudeCeiling']['meters']
            #     hazard_factor = feature['properties']['hazardFactor']
            #     name = feature['properties']['name']
                
            #     if alt_floor > 1500/3.281:
            #         continue
            #     c='black'
            #     plt.plot(*coords.T, c=c)
            #     folium.Polygon(locations=coords[:,[1,0]], color=c, fill=True, popup=f"<h4><b>{name}-Class D</b></h4>\n<ul><li>Floor:{alt_floor}m</li><li>Ceil:{alt_ceiling}m</li><li>Hazard Factor:{hazard_factor}</li></ul>").add_to(
            #         map_area)
                
            #     # new_coords = []
            #     # for coord in coords:
            #     #     new_coords.append((coord[0], coord[1], alt_ceiling))
            #     # add_polygon(kml, new_coords, name, simplekml.Color.aqua,fill=False, fillColor=simplekml.Color.changealphaint(0, simplekml.Color.aqua), extrude = alt_floor==0)

            #     # if alt_floor > 0:
            #     #     new_coords = []
            #     #     for coord in coords:
            #     #         new_coords.append((coord[0], coord[1], alt_floor))
            #     #     add_polygon(kml, new_coords, name, simplekml.Color.aqua,fill=True, fillColor=simplekml.Color.changealphaint(70, simplekml.Color.aqua), extrude=0)
                
            case _:
                pass

kml.save("out/new_airspace.kml")
map_area.save('out/test_map.html')
plt.show()

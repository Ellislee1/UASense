import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import json
import numpy as np

poly = [[-74.35067099968856,40.95038628235707], [-74.35134665517963,40.49961892079759], [-73.64880132727227,40.50053835401126], [-73.65109221855438,40.94965116545227], [-74.35067099968856,40.95038628235707]]

# query_main = 'https://api.altitudeangel.com/v2/mapdata/geojson?'
query_main = 'https://api.altitudeangel.com/v2/mapdata/geojson/groundhazards?'
APIKEY = 'F7FdTYANnvrOikxc6R_Eb4PZpNNtMe8Lm7tO5J1p0'


points = {
    'n':40.95038628,
    'e':-73.64880133,
    's':40.49961892,
    'w': -74.35134666
}

query = f'{query_main}n={points["n"]}&e={points["e"]}&s={points["s"]}&w={points["w"]}'
url = query
headers = {
    'Authorization': f'X-AA-ApiKey {APIKEY}',
}

# headers = {
#     'Host': 'api.altitudeangel.com',
#     'User-Agent': 'Fiddler',
#     'Accept': '*/*',
#     'Authorization': f'bearer {APIKEY}'
# }

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()

    output_file = 'out/out2.geojson'
    with open(output_file, 'w') as f:
        f.write(json.dumps(data))

    print(f"GeoJSON data exported to {output_file}")
    
else:
    print(f"Request failed with status code: {response.status_code}")
    print(response.text)  # You can print the response content for debugging purposes


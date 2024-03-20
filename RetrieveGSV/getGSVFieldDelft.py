import requests
import numpy as np
import time
import shutil

start_time = time.time()

import pandas as pd
import csv
import urllib.request, os
import urllib.parse
import numpy as np
# import streetview
import math
import toml


KEY = toml.load('config.toml').get('apikey')
key = "&key=" + KEY


ROADPOINTS_FILENAME = 'RetrieveGSV/RoadPointsGeneration/roadPoints/roadPointsDelft.csv'

crops = pd.read_csv(ROADPOINTS_FILENAME)
print(crops.columns)

def checkInGrowing(date):
    MONTHS = '05, 06, 07, 08, 09'
    if date[-2:] in MONTHS:
        # print(date[-2:])
        return True
    else:
        return False

def getStreet(lat,lon,saveLocation, bearing, meta, conditie):
    # Ensure the subfolder for the current condition exists
    conditie_path = os.path.join(saveLocation, conditie)
    if not os.path.exists(conditie_path):
        os.makedirs(conditie_path)
        
    fi = meta + ".jpg"
    MyUrl = "https://maps.googleapis.com/maps/api/streetview?size=640x640&location="+str(lat)+","+str(lon)+"&fov=90&heading="+str(bearing)+"&pitch=0" + key
    urllib.request.urlretrieve(MyUrl, os.path.join(conditie_path, fi))

def computePointOnField(fro, theta, d):
    R = 6371e3
    Ad = d/R
    theta = math.radians(theta)
    la2 =  math.asin(math.sin(fro[0]) * math.cos(Ad) + math.cos(fro[0]) * math.sin(Ad) * math.cos(theta))
    lo2 = fro[1] + math.atan2(math.sin(theta) * math.sin(Ad) * math.cos(fro[0]) , math.cos(Ad) - math.sin(fro[0]) * math.sin(la2))
    return (la2,lo2)

def getMeta(points, saveLocation, imLimit=0):
    uniqueImageIDs= []
    points = points.reset_index()  # make sure indexes pair with number of rows
    if imLimit == 0:
        imLimit = len(points)

    i = 0
    for idx, crop in points.iterrows():

        if i <= imLimit:
            # print(crop)
            tree_lat, tree_lon = crop['tree_lat'], crop['tree_lon']
            road_lat, road_lon = crop['road_lat'], crop['road_lon']
            link = "https://maps.googleapis.com/maps/api/streetview/metadata?size=640x640&location="+str(road_lat)+","+str(road_lon)+"&fov=80&heading=0&pitch=0" + key
            response = requests.get(link)
            resJson = response.json()
            # print("resJson", resJson)
            bearing = float(crop['bearing'])


            if resJson['status'] ==  'OK':
        

                if checkInGrowing(resJson['date']):
                    if resJson['pano_id'] not in uniqueImageIDs:
                        uniqueImageIDs.append(resJson['pano_id'])
                        lat_lon_str = f"{tree_lat}_{tree_lon}".replace('.', '-')
                        conditie = crop['CONDITIE']
                        meta = f"{i}_{resJson['date']}_{lat_lon_str}_{conditie}"
                        getStreet(road_lat,road_lon, saveLocation, bearing, meta, conditie)
        
        if i % 100 == 0: #print progress
            print(f"Retrieved {i} / {len(points)} GSV images")
        i+=1
    print(f"Retrieved {i} / {len(points)} GSV images. Done!")



#
# print(computeBearing(fro, to))
#
# print(computeBearing(to, fro))


if __name__ == "__main__":
    imLimit = 100
    getMeta(crops, 'RetrieveGSV/images', imLimit=1000)

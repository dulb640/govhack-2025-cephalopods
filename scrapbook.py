from dataclasses import dataclass
from math import acos, cos, nan, pi, radians, sin
import json
import time
from typing import Any, List, Iterator, Optional, Tuple, TypeVar

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Coord:
    lat: float
    lng: float

# Calculates the distance of two locations on the Earth.
def distance(a, b):
  a_lat = radians(a.lat)
  b_lat = radians(b.lat)
  a_lng = radians(a.lng)
  b_lng = radians(b.lng)
  return 6371000*acos(sin(a_lat)*sin(b_lat)+cos(a_lat)*cos(b_lat)*cos(abs(a_lng-b_lng)))

T = TypeVar('T')
def iterate_pairs(items: List[T]) -> Iterator[Tuple[T, T]]:
  for index in range(len(items)):
    yield (items[index], items[(index+1)%len(items)])

def convert_geojson_poly(poly: List[List[float]]) -> Iterator[Coord]:
  for point in poly:
    if not isinstance(point[0], float) or not isinstance(point[1], float):
      raise Exception(f'Expected floats, but got {type(point[0]).__name__} and {type(point[1]).__name__}')
    yield Coord(point[1], point[0])

@dataclass
class CompoundPoly:
  include: List[Coord]
  exclude: Optional[List[Coord]]

# Represents a government planning zone.
@dataclass
class Zone:
    code: str
    bbox: List[float]
    polys: List[CompoundPoly]

def convert_geojson_zone(zone: Any):
  try:
    polys = None
    if zone['geometry']['type'] == 'Polygon':
        compound_poly = zone['geometry']['coordinates']
        polys = [CompoundPoly(
                    list(convert_geojson_poly(compound_poly[0])),
                    list(convert_geojson_poly(compound_poly[1])) if len(compound_poly) > 1 else None)
        ]
    elif zone['geometry']['type'] == 'MultiPolygon':
        polys = [CompoundPoly(
                    list(convert_geojson_poly(compound_poly[0])),
                    list(convert_geojson_poly(compound_poly[1])) if len(compound_poly) > 1 else None)
                for compound_poly in zone['geometry']['coordinates']
        ]
    else:
      raise Exception(f'Unrecognised zone geometry {zone["geometry"]["type"]}')
    return Zone(
      zone['properties']['ZONE_CODE'],
      zone['geometry']['bbox'],
      polys
      )
  except Exception as e:
    print(zone)
    raise Exception(f'Failed to convert zone') from e

def inside_poly(poly: List[Coord], point: Coord) -> bool:
  inside = False
  for (start, end) in iterate_pairs(poly):
    if ((start.lat > point.lat) != (end.lat > point.lat)) \
        and point.lng < (start.lng-end.lng) * (point.lat-end.lat) / (start.lat-end.lat) + end.lng:
      inside = not inside
  return inside

aus_inner_poly = [
    Coord(-34.30822249066978, 116.5653883179424),
    Coord(-24.76334137039225, 114.05593046356509),
    Coord(-14.328836543147716, 131.9232703867314),
    Coord(-23.61882450772291, 150.192123566598),
    Coord(-37.51858691338602, 149.640042838635),
    Coord(-38.271093566166726, 146.37774762794456),
    Coord(-37.71736040837502, 140.65618371996433),
    Coord(-32.37973632383235, 138.24710417976212),
    Coord(-29.49600942382542, 134.18178245567094),
]

aus_outer_poly = [
    Coord(-35.95001369078268, 114.9091463422826),
    Coord(-20.925359230111567, 111.7472294457672),
    Coord(-10.308987743260781, 131.22062239573495),
    Coord(-10.457089613805607, 142.66375021169537),
    Coord(-27.11006744598046, 155.36160695484446),
    Coord(-38.152788539347284, 150.49325871735252),
    Coord(-39.67569516503084, 146.32755873981978),
    Coord(-38.27109365808601, 139.80296831843887),
]

# The outline of Australia
aus_poly = [
  Coord(-34.01300680685471, 115.12267301075781),
  Coord(-33.20785104534772, 115.91368858045824),
  Coord(-31.049590416923184, 115.518180795608),
  Coord(-26.820587949788322, 113.80431372792377),
  Coord(-26.288685501644395, 114.4400492734514),
  Coord(-22.366799365047015, 113.90701767988867),
  Coord(-18.158460899644655, 122.40464336815332),
  Coord(-16.63790369592792, 122.9031093209688),
  Coord(-17.624623741718846, 123.61441883504615),
  Coord(-16.32987052255237, 123.70620070783033),
  Coord(-13.960172666090198, 127.05623906445271),
  Coord(-14.716315827293629, 128.10514005181733),
  Coord(-15.062793786403079, 129.65871143728035),
  Coord(-14.33069436804494, 129.4396183208221),
  Coord(-12.470757887307611, 130.69442597831144),
  Coord(-12.197262463469402, 131.0579005756694),
  Coord(-12.178155870019017, 132.62167354179348),
  Coord(-11.555937490244615, 132.64305325082537),
  Coord(-12.353693468480493, 136.8665074530299),
  Coord(-14.88376208530373, 135.4168324700277),
  Coord(-17.92013530437751, 140.74216925810475),
  Coord(-11.050060003692467, 142.31018505604587),
  Coord(-11.011339286378966, 142.72972480136204),
  Coord(-14.426505131542186, 143.81415919429847),
  Coord(-14.266105627038096, 144.52607790250437),
  Coord(-14.891006770453913, 145.2545530125205),
  Coord(-16.450351281500968, 145.41794092038413),
  Coord(-16.937378061250463, 145.9090654750822),
  Coord(-19.00514389379872, 146.29341137581878),
  Coord(-20.37713466286721, 148.73862386685948),
  Coord(-22.46930947088821, 149.65249142222333),
  Coord(-22.78847654789301, 150.7392525293525),
  Coord(-25.44821217650984, 152.8633765114687),
  Coord(-28.26823004286797, 153.55495189786964),
  Coord(-32.443998056044656, 152.5422880681589),
  Coord(-32.93092953714893, 151.72161905098724),
  Coord(-33.89156888294221, 151.28735064611104),
  Coord(-35.159594780755484, 150.76083977212548),
  Coord(-35.87983121984224, 150.1469988146545),
  Coord(-37.4942483128829, 149.97663815207142),
  Coord(-37.77880995550617, 149.48365333899284),
  Coord(-37.830781588689234, 148.15555937318229),
  Coord(-39.01997388150113, 146.32047938643413),
  Coord(-37.94668625181082, 144.85535728365107),
  Coord(-38.82409363233498, 143.52977779013085),
  Coord(-37.99313066907563, 140.53985934383155),
  Coord(-35.39744701564586, 138.56971416351536),
  Coord(-32.57053094724269, 137.80999635773276),
  Coord(-34.86281562631423, 135.76052606732563),
  Coord(-32.03292225819515, 133.7463914715807),
  Coord(-31.416780569490182, 131.14921745813842),
  Coord(-32.88261144300397, 124.36750717736132),
  Coord(-33.86262001443438, 123.54428196973626),
  Coord(-33.99341915148931, 119.76094875779181),
  Coord(-35.10422318374648, 117.79922060345127),
  Coord(-34.28336356619469, 115.15439068108138),
]

# The outline of Victoria
vic_poly = [
    Coord(-33.97968861611904, 140.96285676935724),
    Coord(-38.062067198085686, 140.97063036272135),
    Coord(-38.39823127874135, 141.5356463688063),
    Coord(-38.2519071140182, 141.72216428147834),
    Coord(-38.40240762785654, 142.16980727189124),
    Coord(-38.339737060731665, 142.38297060065932),
    Coord(-38.880846164744206, 143.5515005094382),
    Coord(-38.320052051853956, 144.34881057382532),
    Coord(-38.28023421760904, 144.63511736967342),
    Coord(-38.14640029486991, 144.72934492273737),
    Coord(-38.16349917894553, 144.36330712045051),
    Coord(-37.873217464966864, 144.81867187627242),
    Coord(-37.83942318174576, 144.92673484085788),
    Coord(-37.8537855294087, 144.9692492467307),
    Coord(-37.98608515791174, 145.02710037007034),
    Coord(-37.990755930244134, 145.0791206212439),
    Coord(-38.116610648528074, 145.14886458262927),
    Coord(-38.33445828176533, 144.97730050826056),
    Coord(-38.364921789255455, 144.76257566046576),
    Coord(-38.50184893174911, 144.92119150278333),
    Coord(-38.3936811163427, 145.11973106054333),
    Coord(-38.40298315508198, 145.23302809078683),
    Coord(-38.30398196589478, 145.20821064606685),
    Coord(-38.22095376035946, 145.29561121225467),
    Coord(-38.22010604421395, 145.47041234471493),
    Coord(-38.374227531051694, 145.5793932976158),
    Coord(-38.51198135009615, 145.4326466679671),
    Coord(-38.670536371772734, 145.60313172313306),
    Coord(-38.63598774201272, 145.80059226155743),
    Coord(-38.89259654478346, 145.92683752411594),
    Coord(-38.796791762834545, 146.1534315846029),
    Coord(-39.14409748309942, 146.41994936050904),
    Coord(-38.7892227337884, 146.4749793466273),
    Coord(-38.90015458502369, 146.30017821425162),
    Coord(-38.72864157150801, 146.20090843537162),
    Coord(-38.69075223494223, 146.24946430547598),
    Coord(-38.70843642424909, 146.41455426383075),
    Coord(-38.61391842794259, 146.90134727660634),
    Coord(-37.838840347044865, 147.85753849781608),
    Coord(-37.75825744749813, 149.4486415394186),
    Coord(-37.502743448425576, 149.95078563277207),
    Coord(-36.779958030253965, 148.1250437417417),
    Coord(-36.04081351852961, 147.92276273703584),
    Coord(-36.107316154610054, 146.92561797489634),
    Coord(-35.83461593717177, 145.0077305045134),
    Coord(-36.13263569452862, 144.7491389356118),
    Coord(-35.367910207968116, 143.65796087547832),
    Coord(-34.81328958525392, 143.30141795435847),
    Coord(-34.58458140333917, 142.77835773544393),
    Coord(-34.765023628884194, 142.5530696257286),
    Coord(-34.16580644947154, 142.17301838052475),
    Coord(-34.209560210971624, 141.51674432273762),
    Coord(-34.03927927949678, 141.0034792390087),
]

# Returns whether a location is in Australia.
def inside_aus(point: Coord) -> bool:
  if inside_poly(aus_inner_poly, point):
    return True
  if not inside_poly(aus_outer_poly, point):
    return False
  return inside_poly(aus_poly, point)

# Returns whether a location is in Victoria.
def inside_vic(point: Coord) -> bool:
  if not inside_poly(aus_outer_poly, point):
    return False
  return inside_poly(vic_poly, point)

def normalise_closeness_score(min, max, x):
  if x <= min:
    return 1
  if x >= max:
    return 0
  return (max-x)/(max-min)

with open('datasets/submarine-cables.json') as f:
  submarine_cables = json.load(f)
print(f'submarine_cables {len(submarine_cables["features"])}')

with open('datasets/major-power-stations.json') as f:
  power_stations = json.load(f)
print(f'power_stations {len(power_stations["features"])}')

with open('datasets/vic-zones.json') as f:
  zones = json.load(f)
print(f'zones {len(zones["features"])}')
# print('zone 0')
# print(zones["features"][0]) 

# Zones good for datacentre construction.
zone_include_list = {
    'IN2Z', 'IN3Z', 'IN1Z',
    # Farming
    'FZ',
    # Green Wedge
    'GWZ',
}

# Zones inappropriate for datacentre construction.
zone_exclude_list = {
    # General residential
    'GRZ', 'GRZ1', 'GRZ2', 'GRZ3', 'GRZ4', 'GRZ5', 'GRZ6', 'GRZ7', 'GRZ8', 'GRZ9', 'GRZ10',
    'GRZ11', 'GRZ12', 'GRZ13', 'GRZ14', 'GRZ15', 'GRZ16', 'GRZ17', 'GRZ18',
    # Neighbourhood residential
    'NRZ1', 'NRZ2', 'NRZ3', 'NRZ4', 'NRZ5', 'NRZ6', 'NRZ7', 'NRZ8', 'NRZ9', 'NRZ10',
    'NRZ11', 'NRZ12', 'NRZ13', 'NRZ14',
    # Low density residential
    'LDRZ', 'LDRZ1', 'LDRZ2', 'LDRZ3', 'LDRZ4', 'LDRZ5', 'LDRZ6',
    # Residential growth
    'RGZ', 'RGZ1', 'RGZ2', 'RGZ3', 'RGZ4', 'RGZ5', 'RGZ6', 'RGZ7', 'RGZ8', 'RGZ9',
    # Public park
    'PPRZ', 'PCRZ'
}

include_zones = [convert_geojson_zone(zone) for zone in zones['features'] if zone['properties']['ZONE_CODE'] in zone_include_list]
exclude_zones = [convert_geojson_zone(zone) for zone in zones['features'] if zone['properties']['ZONE_CODE'] in zone_exclude_list]

print(f'include_zones {len(include_zones)}')
print(f'exclude_zones {len(exclude_zones)}')

def inside_bbox(point, bbox):
    return (point.lat > bbox[1]) != (point.lat > bbox[3]) and (point.lng > bbox[0]) != (point.lng > bbox[2])

def inside_any_zone(point):
    return any(inside_bbox(point, zone['geometry']['bbox']) for zone in zones['features'])

# Returns whether a location is within an inappropriate planning zone.
def inside_excluded_zone(point):
    return any( \
        inside_bbox(point, zone.bbox) \
        and any(inside_poly(poly.include, point) for poly in zone.polys) \
        for zone in exclude_zones)

# Returns whether a location is within a good planning zone.
def inside_included_zone(point):
    return any( \
        inside_bbox(point, zone.bbox) \
        and any(inside_poly(poly.include, point) for poly in zone.polys) \
        for zone in include_zones)

# Returns the distance from a location to the nearest power station.
def distance_to_station(lat, lng):
    return min(distance(Coord(lat, lng), Coord(station['geometry']['coordinates'][1], station['geometry']['coordinates'][0])) for station in power_stations['features'])

# Returns how many megawatts of power generation are within the a given radius of a location.
def mw_within_radius(point, radius):
  return sum(station['properties']['generationmw'] for station in power_stations['features'] if distance(point, Coord(station['geometry']['coordinates'][1], station['geometry']['coordinates'][0])) <= radius and station['properties']['generationmw'] is not None)

# Names of international cables that have terminals in Australia.
relevant_cable_ids = [
    'australia-singapore-cable-asc',
    'indigo-west',
    'north-west-cable-system',
    'japan-guam-australia-south-jga-s',
    'tasman-global-access-tga-cable',
    # Bass strait is not really an international cable.
    # 'bass-strait-1',
    # 'bass-strait-2',
    'honomoana',
    'sydney-melbourne-adelaide-perth-smap',
    'tabua',
    'tasman-ring-network',
    'australia-connect-interlink',
]
relevant_cables = [cable for cable in submarine_cables['features'] if cable['properties']['id'] in relevant_cable_ids]

# Returns the distance to the nearest submarine cable.
def distance_to_cable(lat, lng):
  return min(distance(Coord(lat, lng), Coord(point[1], point[0])) for cable in relevant_cables for line in cable['geometry']['coordinates'] for point in line)

# Generates a rating for a location.
# If the point is outside victoria, it is currently excluded, because I don't
# have zoning data. If the point is in a residential neighbourhood, or in a
# park or conservation area, it scores 0. Points are rated higher for being
# close to subsea cables, and power stations. The point receives the maximum
# score if it is within 30 kilometres of both a cable and a power station.
def rating(lat, lng):
  if not inside_vic(Coord(lat, lng)):
    return nan
  
#   if inside_included_zone(Coord(lat, lng)):
#     return 1.0
  if inside_excluded_zone(Coord(lat, lng)):
    return 0.0
  subsea_comp = normalise_closeness_score(30, 200, distance_to_cable(lat, lng)/1000)
  mw_comp = normalise_closeness_score(30, 100, distance_to_station(lat, lng)/1000)
  return subsea_comp + mw_comp

# Code below this point generates the plot and saves it to a file.
print('plotting')

x = np.linspace(140, 151.5, 150)
y = np.linspace(-33.5, -39.5, 150)

# Create a 2D grid of x and y values
lngs, lats = np.meshgrid(x, y)

print('rasterising')
t0 = time.perf_counter()
# Calculate the value for each point based on the rating function.
Z = np.vectorize(rating)(lats, lngs)
t1 = time.perf_counter()
print(f'Finished rasterising in {t1-t0} cpu seconds.')

print('plotting')
plt.figure(figsize=(12, 9))
plt.pcolormesh(lngs, lats, Z, shading='auto')
plt.colorbar(label='score')
plt.title('Location rating')
plt.xlabel('longitude')
plt.ylabel('lattitude')
plt.axis('tight')
plt.savefig('aus_rating.png')
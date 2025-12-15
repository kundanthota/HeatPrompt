import requests
import argparse
import json
import sys
import os

def build_wfs_url(bbox):
    return (
        "https://www.energieatlas.rlp.de/geoserver/earp/ows"
        "?service=WFS&version=1.1.0&request=GetFeature"
        "&typeName=wb_rlp_agg_gmndesch10_4326"
        "&outputFormat=application/json"
        "&srsName=EPSG:4326"
        f"&bbox={bbox},EPSG:4326"
    )

def fetch_data(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        sys.exit(1)
    return response.json()

def extract_features(data):
    coords_dict = {}
    props_dict = {}

    for feature in data.get("features", []):
        fid = feature.get("properties", {}).get("id_agg")
        if not fid:
            continue

        geometry = feature.get("geometry", {}).get("coordinates")
        props = feature.get("properties", {})

        coords_dict[fid] = geometry
        props_dict[fid] = {
            "wb_gs": props.get("wb_gs"),
            "geb_n": props.get("geb_n"),
            "Shape_Leng": props.get("Shape_Leng"),
            "flaeche": props.get("flaeche"),
            "Shape_Area": props.get("Shape_Area")
        }

    return coords_dict, props_dict

def main():
    parser = argparse.ArgumentParser(description="Fetch WAD features by BBOX.")
    parser.add_argument('--bbox', required=True, help='Bounding box in format: minLon,minLat,maxLon,maxLat')

    args = parser.parse_args()
    bbox = args.bbox

    os.makedirs('data/atlas_data', exist_ok=True)

    print(f"[INFO] Fetching data for bbox: {bbox}")
    url = build_wfs_url(bbox)
    data = fetch_data(url)

    coords, features = extract_features(data)

    with open("data/atlas_data/geometry_by_id.json", "w") as f:
        json.dump(coords, f, indent=2)

    with open("data/atlas_data/features_by_id.json", "w") as f:
        json.dump(features, f, indent=2)

    print("[INFO] Saved geometry_by_id.json and features_by_id.json")

if __name__ == "__main__":
    main()

import os
import requests
import json
import subprocess
from typing import Tuple, List, Optional
from pydantic import BaseModel

class GeocodeResult(BaseModel):
    lat: float
    lon: float
    formatted_address: str
    place_id: str

class ParcelData(BaseModel):
    building_footprint: Optional[List[Tuple[float, float]]]
    parcel_bounds: Optional[List[Tuple[float, float]]]
    bounding_box: Tuple[float, float, float, float]

def geocode_address(address: str) -> GeocodeResult:
    # ... [Keep your existing geocode_address function exactly as is] ...
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY environment variable is not set.")

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "OK" or not data.get("results"):
        error_msg = data.get("error_message", "No additional error message provided by Google.")
        raise ValueError(f"Geocoding failed for {address}. Status: {data.get('status')} - {error_msg}")

    result = data["results"][0]
    location = result["geometry"]["location"]
    
    return GeocodeResult(
        lat=location["lat"],
        lon=location["lng"],
        formatted_address=result["formatted_address"],
        place_id=result["place_id"]
    )

def fetch_osm_parcel_data(lat: float, lon: float) -> ParcelData:
    """
    Fetches exact building footprints from the Overture Maps Foundation.
    Selects the building closest to the target geocoded coordinates.
    """
    offset = 0.0004
    bbox = (lon - offset, lat - offset, lon + offset, lat + offset)
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    
    building_footprint = None
    
    try:
        print(f"[GIS API] Fetching building footprints from Overture Maps for bounding box: {bbox_str}", flush=True)
        
        result = subprocess.run(
            ["overturemaps", "download", "--bbox", bbox_str, "-f", "geojson", "--type", "building"],
            capture_output=True, 
            text=True, 
            check=True
        )
        
        geojson_data = json.loads(result.stdout)
        
        if geojson_data.get("features"):
            best_footprint = None
            min_dist = float('inf')
            
            for feature in geojson_data["features"]:
                geometry = feature["geometry"]
                if geometry["type"] == "Polygon":
                    raw_coords = geometry["coordinates"][0]
                    
                    # Calculate simple centroid of the polygon
                    avg_lon = sum(c[0] for c in raw_coords) / len(raw_coords)
                    avg_lat = sum(c[1] for c in raw_coords) / len(raw_coords)
                    
                    # Squared Euclidean distance is sufficient for a micro-local bounding box
                    dist = (avg_lon - lon)**2 + (avg_lat - lat)**2
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_footprint = [(p_lat, p_lon) for p_lon, p_lat in raw_coords]
            
            if best_footprint:
                building_footprint = best_footprint
                print(f"[GIS API] ✅ Successfully extracted target footprint from Overture.", flush=True)

    except FileNotFoundError:
        print("[WARN] 'overturemaps' CLI tool not found. Did you run 'pip install overturemaps'?", flush=True)
    except Exception as e:
        print(f"[WARN] Overture Maps extraction bypassed: {e}. Falling back to visual satellite pipeline.", flush=True)

    default_bounding_box = (lat - offset, lon - offset, lat + offset, lon + offset)

    return ParcelData(
        building_footprint=building_footprint,
        parcel_bounds=None,
        bounding_box=default_bounding_box
    )

def get_site_context(address: str) -> dict:
    location = geocode_address(address)
    parcel = fetch_osm_parcel_data(location.lat, location.lon)

    return {
        "address": location.formatted_address,
        "coordinates": (location.lat, location.lon),
        "place_id": location.place_id,
        "parcel_data": parcel.model_dump()
    }
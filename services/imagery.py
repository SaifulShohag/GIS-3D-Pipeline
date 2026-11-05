import os
import math
import requests
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel

class ImageryAssets(BaseModel):
    street_view_image_paths: List[str]
    metadata_status: str

def calculate_heading(cam_lat: float, cam_lon: float, target_lat: float, target_lon: float) -> float:
    """Calculates the compass bearing from the street camera to the house."""
    dLon = math.radians(target_lon - cam_lon)
    lat1, lat2 = math.radians(cam_lat), math.radians(target_lat)
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def get_pano_metadata(identifier: str, is_pano: bool = False) -> dict:
    """Fetches panorama links and exact street coordinates."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"key": api_key, "pano" if is_pano else "location": identifier}
    response = requests.get(url, params=params)
    return response.json()

def download_pano_image(identifier: str, heading: float, save_path: Path, zoom: int = 55) -> Optional[str]:
    """Downloads a zoomed-in street view image pointing exactly at the computed heading."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "600x640",
        "scale": 4,
        "fov": zoom,
        "heading": heading,
        "pitch": 0,
        "return_error_code": "true",
        "key": api_key
    }
    
    # Allow querying by either pano ID or lat,lon coordinates
    if "," in identifier:
        params["location"] = identifier
    else:
        params["pano"] = identifier
        
    response = requests.get(url, params=params)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return str(save_path)
    return None

def calculate_offset(lat: float, lon: float, distance_m: float = 15.0, bearing_deg: float = 0.0) -> tuple[float, float]: 
    """Calculates a new GPS coordinate by moving a set distance along a compass bearing."""
    R = 6378137.0 # Earth radius in meters
    lat_rad, lon_rad = math.radians(lat), math.radians(lon)
    bearing_rad = math.radians(bearing_deg)
    
    new_lat = math.asin(math.sin(lat_rad) * math.cos(distance_m / R) + 
                        math.cos(lat_rad) * math.sin(distance_m / R) * math.cos(bearing_rad))
    
    new_lon = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_m / R) * math.cos(lat_rad),
                                   math.cos(distance_m / R) - math.sin(lat_rad) * math.sin(new_lat))
    
    return math.degrees(new_lat), math.degrees(new_lon)

def fetch_multi_angle_street_view(house_lat: float, house_lon: float, base_dir: Path) -> List[str]:
    """
    1. Finds the street in front of the house and takes a center photo.
    2. Calculates coordinates 15 meters left/right along the street.
    3. Takes angled photos from those new positions focused back on the house.
    """
    saved_paths = []
    
    # 1. Get Center Image
    center_meta = get_pano_metadata(f"{house_lat},{house_lon}", is_pano=False)
    if center_meta.get("status") != "OK":
        return saved_paths

    cam_lat, cam_lon = center_meta["location"]["lat"], center_meta["location"]["lng"]
    center_heading = calculate_heading(cam_lat, cam_lon, house_lat, house_lon)
    
    center_path = download_pano_image(center_meta["pano_id"], center_heading, base_dir / "street_center.png")
    if center_path:
        saved_paths.append(center_path)


    left_bearing = (center_heading - 60) % 360
    right_bearing = (center_heading + 90) % 360

    # Execute Left Step
    left_lat, left_lon = calculate_offset(cam_lat, cam_lon, 17.0, left_bearing)
    left_heading = calculate_heading(left_lat, left_lon, house_lat, house_lon)
    left_path = download_pano_image(f"{left_lat},{left_lon}", left_heading, base_dir / "street_left.png", zoom=48)
    if left_path:
        saved_paths.append(left_path)

    # Execute Right Step
    right_lat, right_lon = calculate_offset(cam_lat, cam_lon, 12.0, right_bearing)
    right_heading = calculate_heading(right_lat, right_lon, house_lat, house_lon)
    right_path = download_pano_image(f"{right_lat},{right_lon}", right_heading, base_dir / "street_right.png", zoom=48)
    if right_path:
        saved_paths.append(right_path)

    return saved_paths

def download_site_imagery(lat: float, lon: float, site_id: str) -> ImageryAssets:
    """
    Orchestrates the downloading of multi-angle street view assets for DUSt3R.
    """
    base_dir = Path("static/assets/imagery") / site_id
    base_dir.mkdir(parents=True, exist_ok=True)

    # Only fetch the 3x Street View angles
    saved_sv_paths = fetch_multi_angle_street_view(lat, lon, base_dir)

    return ImageryAssets(
        street_view_image_paths=saved_sv_paths,
        metadata_status="success" if saved_sv_paths else "failed_no_streetview"
    )
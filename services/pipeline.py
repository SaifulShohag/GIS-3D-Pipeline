import os
import time
import math
import hashlib
import asyncio
from typing import Tuple, List

from models.site_model import SiteModelPayload
from services.geocoder import get_site_context
from services.imagery import download_site_imagery
from services.reconstruction import generate_3d_mesh

def generate_site_id(address: str) -> str:
    return hashlib.md5(address.strip().lower().encode('utf-8')).hexdigest()[:8]

def compute_meters_per_degree(lat: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
    return m_per_deg_lat, m_per_deg_lon

def project_gis_to_local(
    footprint_coords: List[Tuple[float, float]],
    center_lat: float,
    center_lon: float,
    m_lat: float,
    m_lon: float
) -> List[Tuple[float, float]]:
    local_pts = []
    for lat, lon in footprint_coords:
        x_m = (lon - center_lon) * m_lon
        z_m = -(lat - center_lat) * m_lat
        local_pts.append((x_m, z_m))
    return local_pts

async def generate_site_payload(address: str) -> SiteModelPayload:
    print(f"\n[Pipeline] 🚀 Starting generation for address: {address}", flush=True)
    start_time = time.time()
    site_id = generate_site_id(address)
    
    expected_mesh_path = f"static/assets/models/{site_id}.glb"

    print("[Pipeline] 📍 Step 1: Geocoding & Fetching OSM Parcel Data...", flush=True)
    t1 = time.time()
    site_context = await asyncio.to_thread(get_site_context, address)
    lat, lon = site_context["coordinates"]
    print(f"[Pipeline] ✅ Step 1 Complete ({time.time() - t1:.2f}s): Resolved to {lat}, {lon}", flush=True)

    print("[Pipeline] 📐 Formatting Spatial GIS Data...", flush=True)
    bbox = site_context["parcel_data"]["bounding_box"]
    gis_footprint = site_context["parcel_data"]["building_footprint"]
    
    min_lat, min_lon, max_lat, max_lon = bbox
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    
    m_per_lat, m_per_lon = compute_meters_per_degree(center_lat)
    lot_height_m = (max_lat - min_lat) * m_per_lat
    lot_width_m = (max_lon - min_lon) * m_per_lon

    if gis_footprint:
        local_footprint = project_gis_to_local(gis_footprint, center_lat, center_lon, m_per_lat, m_per_lon)
    else:
        local_footprint = [[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]

    # --- CACHE CHECK ---
    if os.path.exists(expected_mesh_path):
        print(f"[Pipeline] ⚡ CACHE HIT! Found existing 3D model for {site_id}. Skipping inference.", flush=True)
        return SiteModelPayload(
            address=site_context["address"],
            site_id=site_id,
            lot_dimensions=(lot_width_m, lot_height_m),
            footprint_polygon=local_footprint,
            mesh_url=f"/{expected_mesh_path}",
            site_metadata={"architectural_style": "Loaded from cache"}
        )

    print(f"[Pipeline] 📸 Step 2: Downloading Multi-Angle Street View for {site_id}...", flush=True)
    t2 = time.time()
    imagery_assets = await asyncio.to_thread(download_site_imagery, lat, lon, site_id)
    print(f"[Pipeline] ✅ Step 2 Complete ({time.time() - t2:.2f}s)", flush=True)

    # --- Step 3: DUSt3R Point Cloud Reconstruction ---
    print("[Pipeline] 🧊 Step 3: Generating 3D Point Cloud via DUSt3R...", flush=True)
    t3 = time.time()
    mesh_url = await asyncio.to_thread(
        generate_3d_mesh, 
        imagery_assets.street_view_image_paths, 
        site_id
    )
    print(f"[Pipeline] ✅ Step 3 Complete ({time.time() - t3:.2f}s): Point cloud generated.", flush=True)

    final_payload = SiteModelPayload(
        address=site_context["address"],
        site_id=site_id,
        lot_dimensions=(lot_width_m, lot_height_m),
        footprint_polygon=local_footprint,
        mesh_url=mesh_url,
        site_metadata={"architectural_style": "Generated via DUSt3R Point Cloud"}
    )
    
    print(f"[Pipeline] 🎉 Pipeline Finished in {time.time() - start_time:.2f} seconds! Returning payload.", flush=True)
    return final_payload
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field

class SiteModelPayload(BaseModel):
    address: str
    site_id: str
    lot_dimensions: Tuple[float, float]
    footprint_polygon: List[Tuple[float, float]] = Field(..., description="Ordered (x, z) perimeter points in meters to scale the 3D mesh.")
    mesh_url: str = Field(..., description="Relative path to the generated .glb or .obj file (e.g., '/static/assets/models/site_123.glb')")
    site_metadata: Optional[dict] = None
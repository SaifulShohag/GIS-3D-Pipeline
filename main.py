from dotenv import load_dotenv
load_dotenv()  # Loads .env into os.environ before service imports run

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from models.site_model import SiteModelPayload
from services.pipeline import generate_site_payload

app = FastAPI(title="Freshprint Pipeline")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

# Handlers for both GET (browser URL testing) and POST (frontend/API client)
@app.api_route("/api/generate-site", methods=["GET", "POST"], response_model=SiteModelPayload)
async def create_site_model(
    address: str = Query(
        default="505 Ridge Dr, Florissant, MO 63033",
        description="Target property address to process"
    )
):
    try:
        payload = await generate_site_payload(address)
        return payload
    except Exception as e:
        # Re-raise with full details so pipeline errors are visible in browser/tools
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")
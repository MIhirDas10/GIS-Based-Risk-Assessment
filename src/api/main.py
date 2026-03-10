from fastapi import FastAPI
app = FastAPI(
    title = "Dengue Risk Prediction API",
    description = "GIS-based vector-borne disease risk prediction for Bangladesh",
    version = "0.1.0"
)

@app.get("/api/v1/health")
def health():
    return {
        "status": "online",
        "model_version": None,
        "data_freshness": None,
        "message": "Pipelines yet to be run"
    }
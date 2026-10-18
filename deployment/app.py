from pathlib import Path
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from inference import CarFeaturePredictor 
from schemas import CarMileageInput, CarPowerInput, CarPriceInput 

# Define paths to model artifact directories
BASE_DIR = Path(__file__).resolve().parents[0]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Initialize predictors 
mileage_predictor = CarFeaturePredictor(ARTIFACTS_DIR / "mileage")
power_predictor = CarFeaturePredictor(ARTIFACTS_DIR / "power") 
price_predictor = CarFeaturePredictor(ARTIFACTS_DIR / "price")

app = FastAPI(
    title="Car Analytics API",
    description="Multi-model inference API providing predictions for vehicle metrics like mileage, power, and price.",
    version="1.0.0",
)
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
 
@app.get("/", status_code=status.HTTP_200_OK, tags=["General"])
def home():
    """Welcome endpoint providing basic API information."""
    return {
        "message": "Welcome to the Car Analytics API",
        "docs_url": "/docs",
        "health_check": "/health",
    }
 
@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health Check"])
def health_check():
    """Returns server health status."""
    return {"status": "healthy", "service": "Car Analytics API"}
 
@app.post("/predict/mileage", status_code=status.HTTP_200_OK, tags=["Mileage"])
def predict_mileage_endpoint(data: CarMileageInput):
    """Accepts car specifications and returns predicted mileage (kmpl)."""
    try:
        input_dict = data.model_dump()
        predicted_val = mileage_predictor.predict(input_dict)
        return {
            "status": "success",
            "predicted_mileage": round(predicted_val, 2),
            "unit": "kmpl",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/predict/power", status_code=status.HTTP_200_OK, tags=["Power"])
def predict_power_endpoint(data: CarPowerInput):
    """Accepts car specifications and returns predicted engine power (bhp)."""
    try:
        input_dict = data.model_dump()
        predicted_val = power_predictor.predict(input_dict)
        return {
            "status": "success",
            "predicted_power": round(predicted_val, 2),
            "unit": "bhp",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/predict/price", status_code=status.HTTP_200_OK, tags=["Price"])
def predict_price_endpoint(data: CarPriceInput):
    """Accepts car specifications and returns predicted resale price (INR)."""
    try:
        input_dict = data.model_dump(by_alias=True)
        predicted_val = price_predictor.predict(input_dict)
        return {
            "status": "success",
            "predicted_price": round(predicted_val, 2),
            "unit": "INR",
        }
    except Exception as e:
        import traceback
        traceback.print_exc() 
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    
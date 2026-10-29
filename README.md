# Used Car Features Prediction

An end-to-end machine learning project for second-hand cars. It scrapes listings, cleans and encodes the data, trains models, and serves three predictions through a FastAPI service with an interactive web dashboard:

| Prediction | Endpoint | Unit |
|---|---|---|
| Mileage | `POST /predict/mileage` | kmpl |
| Engine power | `POST /predict/power` | bhp |
| Resale price | `POST /predict/price` | INR |

---

## Quick start with Docker

No Python setup is needed. With Docker installed, run:

```bash
docker run -p 8000:8000 sumangouda/car-analytics:latest
```

Then open the interactive dashboard in your browser:

**http://localhost:8000/dashboard**

Other pages served by the same container:

- API docs (Swagger UI): http://localhost:8000/docs
- Health check: http://localhost:8000/health

The image contains only the inference service: the API, the dashboard and the trained models. Scraping and training are not part of it.

### Build the image yourself

```bash
cd deployment
docker build -t car-analytics .
docker run --rm -p 8000:8000 car-analytics
```

The Docker build uses `deployment/requirements.txt`, a slim list of pinned packages that only the API needs. The root `requirements.txt` covers the whole project, including training and scraping.

---

## Features

- **Data collection:** a CarDekho scraper that stores listings in a database.
- **Reproducible pipeline:** fetch, clean, encode, train and evaluate scripts, with an EDA report (correlation heatmap, distributions, outliers) saved for each target.
- **Experiment tracking:** training runs are logged and registered with MLflow.
- **One inference class for all models:** `CarFeaturePredictor` loads a model and its encoding metadata from an artifact folder, so adding a new target needs new artifacts, not new code.
- **Validated API:** Pydantic schemas check types and allowed categories before a request reaches the model.
- **Dashboard:** a dark-themed web UI with a form for each prediction, served by the same FastAPI app.
- **Containerized:** a Docker image runs the full service with a single command.
- **Tests:** pytest suites for the cleaning, preprocessing, model helpers and training pipeline.

---

## How it works

```
CarDekho ──► web_scraping ──► database ──► predictions/fetch_raw_data
                                                   │
                                      process_raw_data (clean + encode)
                                                   │
                                  training + evaluation (MLflow tracking)
                                                   │
                                   deployment/artifacts/<target>/
                                      model.pkl + encoding_metadata.json
                                                   │
                         FastAPI + CarFeaturePredictor ──► JSON API / dashboard
                                                   │
                                          Docker image (deployment/)
```

**Encoding is saved with the model.** Each artifact folder holds an `encoding_metadata.json` that records how every categorical column was transformed during training. At inference time the same transformations are replayed from that file, so training and serving cannot drift apart. Supported encodings are:

- **one-hot** (with a dropped baseline category), for example Fuel, Drive Type, Turbo Charger
- **frequency**, for high-cardinality columns
- **target**, used for the price model's `brand` and `model`
- **label / ordinal**, for example Transmission (Manual = 0, Automatic = 1)

Categories the model never saw fall back to a stored default instead of raising an error. For the price model, the input `car_name` is split into `brand` (first word) and `model` (the rest) before encoding.

---

## Project structure

```
.
├── config/                  # Project settings
├── web_scraping/            # CarDekho scraper and database handlers
│   ├── core/cardekho_scraper.py
│   ├── utils/db_handler.py
│   ├── main.py
│   └── save_to_db.py
├── predictions/             # Data and training pipeline
│   ├── fetch_raw_data.py
│   ├── process_raw_data.py
│   ├── activate_train_pipeline.py
│   └── core/                # training, evaluation, pipeline
├── utils/                   # Cleaners, preprocessing, EDA, model helpers, MLflow registry
├── deployment/              # The API and its Docker setup
│   ├── app.py               # FastAPI routes and dashboard routes
│   ├── inference.py         # CarFeaturePredictor
│   ├── schemas.py           # Pydantic request models
│   ├── artifacts/           # mileage/, power/, price/
│   ├── templates/           # Dashboard HTML
│   ├── static/              # Dashboard CSS
│   ├── Dockerfile
│   ├── .dockerignore
│   └── requirements.txt     # API-only dependencies
├── tests/                   # pytest suites
└── requirements.txt         # Full project dependencies
```

---

## Run locally without Docker

### Prerequisites

- Python 3.14 (the version the pinned dependencies were developed and tested with)
- `pip`

### Install

```bash
git clone https://github.com/SumanGouda/predict-used-car-features.git
cd predict-used-car-features

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### Start the API

The trained models are included in `deployment/artifacts/`, so you can start the API directly. Run it from inside `deployment/`, because the app imports its modules from that folder:

```bash
cd deployment
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000/dashboard for the dashboard, or http://127.0.0.1:8000/docs for the API docs.

---

## API usage

Request keys must match the aliases below, including the spaces.

**`POST /predict/power`**
```json
{
  "Engine": 796,
  "No. of Cylinders": 3,
  "Transmission": "Manual",
  "Emission Norm Compliance": 4,
  "Valves Per Cylinder": 4,
  "Engine Type": "I3",
  "Fuel": "Petrol",
  "Drive Type": "FWD"
}
```
Response:

```json
{
  "status": "success",
  "predicted_power": 69.86,
  "unit": "bhp"
}
```

**`POST /predict/mileage`**
```json
{
  "Engine": 796,
  "Kerb Weight": 750,
  "Transmission": "Manual",
  "Power": 47,
  "No. of Cylinders": 3,
  "Registration Year": 2019,
  "Fuel": "Petrol"
}
```
Response:

```json
{
  "status": "success",
  "predicted_mileage": 24.6,
  "unit": "kmpl"
}
```

**`POST /predict/price`**

```json
{
  "Mileage": 21.2,
  "Engine": 1197,
  "Kerb Weight": 890,
  "Power": 82,
  "Registration Year": 2019,
  "Kms Driven": 45000,
  "Seats": 5,
  "Transmission": "Manual",
  "Drive Type": "FWD",
  "car_name": "Maruti Suzuki Swift",
  "Turbo Charger": "No",
  "Ownership": 1,
  "Fuel": "Petrol",
  "No. of Cylinders": 4
}
```

Response:

```json
{
  "status": "success",
  "predicted_price": 471164.75,
  "unit": "INR"
}
```

Allowed categories for the price model: `Transmission` is `Automatic` or `Manual`, `Drive Type` is `FWD`, `RWD` or `AWD`, `Turbo Charger` is `No`, `Yes` or `Twin`, and `Fuel` is `CNG`, `Diesel`, `LPG` or `Petrol`.

The request fields for `/predict/mileage` and `/predict/power` are listed on the `/docs` page.

---

## Reproducing the pipeline

The raw and processed datasets are not committed to the repository, so rebuild them first:

```bash
# 1. Scrape listings and store them in the database
python web_scraping/main.py
python web_scraping/save_to_db.py

# 2. Fetch, process and train
python predictions/fetch_raw_data.py
python predictions/process_raw_data.py
python predictions/activate_train_pipeline.py
```

Training runs are tracked in MLflow. To browse them:

```bash
mlflow ui
```

Copy the final `model.pkl` and `encoding_metadata.json` for each target into `deployment/artifacts/<target>/` to update the API, then rebuild the Docker image.

---

## Testing

```bash
pytest
```

---

## Tech stack

Python, pandas, XGBoost, scikit-learn, MLflow, FastAPI, Pydantic, Jinja2, Docker, pytest.
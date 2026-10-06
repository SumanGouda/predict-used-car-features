import pytest
from pathlib import Path
import numpy as np 
import pandas as pd

@pytest.fixture
def temp_mlruns_dir(tmp_path): 
    tracking_dir = tmp_path / "test_mlruns"
    return tracking_dir.as_uri()

@pytest.fixture
def sample_dataset(tmp_path): 
    np.random.seed(42)
    n_samples = 100

    data = {
        "Feature_1": np.random.rand(n_samples),
        "Feature_2": np.random.rand(n_samples),
        "Mileage": np.random.rand(n_samples) * 50,
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "dummy_car_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


import sqlite3
import pytest
from pathlib import Path

# Replace with your actual import path
from utils.model_helper import get_and_validate_features


@pytest.fixture
def mock_db(tmp_path: Path) -> Path:
    """Creates a temporary SQLite database with standard tables and columns."""
    db_path = tmp_path / "test_car_database.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create dummy tables
    cursor.execute(
        "CREATE TABLE specs (Mileage REAL, Engine REAL, Power REAL, 'Transmission Type' TEXT);"
    )
    cursor.execute(
        "CREATE TABLE details ('Registration Year' INTEGER, Price REAL);"
    )
    conn.commit()
    conn.close()
    return db_path


def test_get_and_validate_features_success(tmp_path: Path, mock_db: Path):
    """Tests happy path: valid feature names matching DB (including case-insensitivity)."""
    features_file = tmp_path / "features.txt"
    features_file.write_text(
        "# This is a comment\n"
        "Mileage\n"
        "engine\n"  # Lowercase test against 'Engine' in DB
        "Transmission Type\n",
        encoding="utf-8",
    )

    features = get_and_validate_features(features_file, mock_db)

    # Asserts that exact original strings from file are returned
    assert features == ["Mileage", "engine", "Transmission Type"]


def test_missing_features_file_raises_error(tmp_path: Path, mock_db: Path):
    """Tests error handling when the features file does not exist."""
    non_existent_file = tmp_path / "non_existent.txt"

    with pytest.raises(FileNotFoundError, match="Features configuration file not found"):
        get_and_validate_features(non_existent_file, mock_db)


def test_missing_db_file_raises_error(tmp_path: Path):
    """Tests error handling when the database file does not exist."""
    features_file = tmp_path / "features.txt"
    features_file.write_text("Mileage\nEngine\n", encoding="utf-8")
    non_existent_db = tmp_path / "non_existent.db"

    with pytest.raises(FileNotFoundError, match="Database file not found"):
        get_and_validate_features(features_file, non_existent_db)


def test_empty_features_file_raises_error(tmp_path: Path, mock_db: Path):
    """Tests error handling when features file is empty or only contains comments."""
    features_file = tmp_path / "features.txt"
    features_file.write_text("# Only comments\n\n  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="No valid features found"):
        get_and_validate_features(features_file, mock_db)


def test_invalid_feature_not_in_db_raises_error(tmp_path: Path, mock_db: Path):
    """Tests error handling when a feature in the text file does not exist in any DB table."""
    features_file = tmp_path / "features.txt"
    features_file.write_text("Mileage\nNonExistentFeature\n", encoding="utf-8")

    with pytest.raises(KeyError, match="Validation Failed"):
        get_and_validate_features(features_file, mock_db)


def test_empty_database_raises_error(tmp_path: Path):
    """Tests error handling when DB has no user-created tables."""
    empty_db = tmp_path / "empty.db"
    conn = sqlite3.connect(empty_db)
    conn.close()

    features_file = tmp_path / "features.txt"
    features_file.write_text("Mileage\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain any tables"):
        get_and_validate_features(features_file, empty_db)

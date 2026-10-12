from pydantic import BaseModel, Field
from typing import Literal

class CarPowerInput(BaseModel):
    Engine: float
    No_of_Cylinders: int = Field(..., alias="No. of Cylinders")
    Drive_Type: str = Field(..., alias="Drive Type")
    Transmission_Type: str = Field(..., alias="Transmission Type")
    Fuel: str
    Emission_Norms: float = Field(..., alias="Emission Norms")
    valves_per_cylinder: int = Field(..., alias="valves per cylinder")
    power: float = Field(..., alias="Power")
    Engine_shape: str = Field(..., alias="Engine shape")

    model_config = {"populate_by_name": True}


class CarMileageInput(BaseModel):
    Engine: float
    Kerb_Weight: float = Field(..., alias="Kerb Weight")
    Fuel: str
    Transmission_Type: str = Field(..., alias="Transmission Type")
    Power: float
    No_of_Cylinders: int = Field(..., alias="No. of Cylinders")
    Registration_Year: int = Field(..., alias="Registration Year")

    model_config = {"populate_by_name": True}


class CarPriceInput(BaseModel):
    Mileage: float
    Engine: float
    Kerb_Weight: float = Field(..., alias="Kerb Weight")
    Power: float
    Registration_Year: int = Field(..., alias="Registration Year")
    Kms_Driven: float = Field(..., alias="Kms Driven")
    Seats: int
    Transmission: Literal["Automatic", "Manual"]
    Drive_Type: Literal["FWD", "RWD", "AWD"] = Field(..., alias="Drive Type")
    car_name: str
    Turbo_Charger: Literal["No", "Yes", "Twin"] = Field(..., alias="Turbo Charger")
    Ownership: int
    Fuel: Literal["CNG", "Diesel", "LPG", "Petrol"]
    No_of_Cylinders: float = Field(..., alias="No. of Cylinders")

    model_config = {"populate_by_name": True}
    
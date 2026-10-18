from pydantic import BaseModel, Field
from typing import Literal


class CarPowerInput(BaseModel):
    Engine: float
    No_of_Cylinders: int = Field(..., alias="No. of Cylinders")
    Transmission: Literal["Automatic", "Manual"]
    Emission_Norm_Compliance: float = Field(..., alias="Emission Norm Compliance")
    Valves_Per_Cylinder: int = Field(..., alias="Valves Per Cylinder")
    Engine_Type: Literal[
        "Unknown", "Inline", "I2", "I3", "I4", "I6", "V", "V6", "V8", "V12"
    ] = Field(..., alias="Engine Type")
    Fuel: Literal["CNG", "Diesel", "LPG", "Petrol"]
    Drive_Type: Literal["AWD", "FWD", "RWD"] = Field(..., alias="Drive Type")

    model_config = {"populate_by_name": True}


class CarMileageInput(BaseModel):
    Engine: float
    Kerb_Weight: float = Field(..., alias="Kerb Weight")
    Transmission: Literal["Automatic", "Manual"]
    Power: float
    No_of_Cylinders: int = Field(..., alias="No. of Cylinders")
    Registration_Year: int = Field(..., alias="Registration Year")
    Fuel: Literal["CNG", "Diesel", "LPG", "Petrol"]

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

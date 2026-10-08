from pydantic import BaseModel, Field


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
    
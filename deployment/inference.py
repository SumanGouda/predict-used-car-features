import json
from pathlib import Path
import joblib 
import pandas as pd

class CarFeaturePredictor:
    def __init__(self, artifact_dir):
        self.artifact_dir = Path(artifact_dir)
        self.model = joblib.load(artifact_dir / "model.pkl")
        encoding_path = self.artifact_dir / "encoding_metadata.json"
        self.encoding_metadata = {}
        
        if encoding_path.exists():  
            with open(encoding_path, "r", encoding="utf-8") as f:
                self.encoding_metadata = json.load(f)

        missing_path = self.artifact_dir / "fill_missing.json"
        self.fill_missing_metadata = {}
        if missing_path.exists():
            with open(missing_path, "r", encoding="utf-8") as f:
                self.fill_missing_metadata = json.load(f)

    def preprocess_input(self, input_data: dict) -> pd.DataFrame:
        df = pd.DataFrame([input_data])

        for col_name, meta in self.fill_missing_metadata.items():
            if col_name in df.columns:
                fill_val = meta.get("fill_value")
                indicator_col = meta.get("indicator_column")

                if indicator_col:
                    df[indicator_col] = df[col_name].isnull().astype(int)
                if fill_val is not None:
                    df[col_name] = df[col_name].fillna(fill_val)
 
        if "car_name" in df.columns:
            parts = str(df["car_name"].iloc[0]).strip().split(" ", 1)
            car_brand = parts[0]
            car_model = parts[1] if len(parts) > 1 else "Unknown"
            
            if "brand" in self.encoding_metadata and "brand" not in df.columns:
                df["brand"] = car_brand
            if "model" in self.encoding_metadata and "model" not in df.columns:
                df["model"] = car_model

        for col_name, info in self.encoding_metadata.items():
            if col_name not in df.columns:
                continue
            encoding_type = info.get("encoding_type")
            meta = info.get("metadata", {})
            val = df[col_name].iloc[0]
    
            if encoding_type == "one_hot":
                encoded_cols = meta.get("encoded_columns", [])
                for enc_col in encoded_cols:
                    category_name = enc_col[len(col_name) + 1:]
                    df[enc_col] = 1 if str(val) == category_name else 0
                df.drop(columns=[col_name], inplace=True)

            elif encoding_type in ("frequency", "target"):
                enc_map = meta.get("frequency_map") or meta.get("encoding_map", {})
                default_val = meta.get("unseen_default_value", 0)
                target_col = meta.get("encoded_column", col_name)

                df[target_col] = df[col_name].astype(str).map(enc_map).fillna(default_val)
                if target_col != col_name and col_name in df.columns:
                    df.drop(columns=[col_name], inplace=True)
    
            elif encoding_type in ("label", "ordinal"):
                label_map = meta.get("label_map", {})
                default_val = meta.get("unseen_default_value", -1)
                target_col = meta.get("encoded_column", col_name)
    
                df[target_col] = df[col_name].map(label_map).fillna(default_val)
                if target_col != col_name and col_name in df.columns:
                    df.drop(columns=[col_name], inplace=True)

        if self.model and hasattr(self.model, "feature_names_in_"):
            expected_cols = list(self.model.feature_names_in_)
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0
            df = df[expected_cols]

        return df
    
    def predict(self, input_data: dict) -> float:
        """Executes full inference pipeline for any target feature."""
        processed_df = self.preprocess_input(input_data)
        prediction = self.model.predict(processed_df)
        return float(prediction[0])
    
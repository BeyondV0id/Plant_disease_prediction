from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "Soil_Moisture",
    "Ambient_Temperature",
    "Soil_Temperature",
    "Humidity",
    "Light_Intensity",
    "Soil_pH",
    "Nitrogen_Level",
    "Phosphorus_Level",
    "Potassium_Level",
    "Chlorophyll_Content",
    "Electrochemical_Signal",
]

DISPLAY_NAMES = {
    "Soil_Moisture": {"English": "Soil Moisture", "Hindi": "मिट्टी की नमी"},
    "Ambient_Temperature": {"English": "Ambient Temperature", "Hindi": "वातावरण तापमान"},
    "Soil_Temperature": {"English": "Soil Temperature", "Hindi": "मिट्टी का तापमान"},
    "Humidity": {"English": "Humidity", "Hindi": "नमी"},
    "Light_Intensity": {"English": "Light Intensity", "Hindi": "प्रकाश तीव्रता"},
    "Soil_pH": {"English": "Soil pH", "Hindi": "मिट्टी pH"},
    "Nitrogen_Level": {"English": "Nitrogen Level", "Hindi": "नाइट्रोजन स्तर"},
    "Phosphorus_Level": {"English": "Phosphorus Level", "Hindi": "फॉस्फोरस स्तर"},
    "Potassium_Level": {"English": "Potassium Level", "Hindi": "पोटैशियम स्तर"},
    "Chlorophyll_Content": {"English": "Chlorophyll Content", "Hindi": "क्लोरोफिल स्तर"},
    "Electrochemical_Signal": {"English": "Electrochemical Signal", "Hindi": "इलेक्ट्रोकेमिकल सिग्नल"},
}

UNITS = {
    "Soil_Moisture": "%",
    "Ambient_Temperature": "C",
    "Soil_Temperature": "C",
    "Humidity": "%",
    "Light_Intensity": "lux",
    "Soil_pH": "pH",
    "Nitrogen_Level": "mg/kg",
    "Phosphorus_Level": "mg/kg",
    "Potassium_Level": "mg/kg",
    "Chlorophyll_Content": "SPAD",
    "Electrochemical_Signal": "mV",
}

TRANSLATIONS = {
    "Healthy": {"English": "Healthy", "Hindi": "स्वस्थ"},
    "Moderate Stress": {"English": "Moderate Stress", "Hindi": "मध्यम तनाव"},
    "High Stress": {"English": "High Stress", "Hindi": "उच्च तनाव"},
    "Stable": {"English": "Stable", "Hindi": "स्थिर"},
    "Watch": {"English": "Watch", "Hindi": "निगरानी"},
    "High": {"English": "High", "Hindi": "उच्च"},
    "Critical": {"English": "Critical", "Hindi": "गंभीर"},
    "low": {"English": "low", "Hindi": "कम"},
    "high": {"English": "high", "Hindi": "ज्यादा"},
    "balanced": {"English": "balanced", "Hindi": "संतुलित"},
    "Sensor reading is outside the training range; verify the device and interpret carefully.": {
        "English": "Sensor reading is outside the training range; verify the device and interpret carefully.",
        "Hindi": "सेंसर रीडिंग ट्रेनिंग रेंज से बाहर है; डिवाइस जांचें और सावधानी से समझें।",
    },
}

ACTION_LIBRARY = {
    "inspect_within_24h": {
        "English": "Inspect this plant within 24 hours and confirm the field conditions before the next irrigation cycle.",
        "Hindi": "इस पौधे की 24 घंटे के भीतर जांच करें और अगली सिंचाई से पहले खेत की स्थिति की पुष्टि करें।",
    },
    "tune_environment": {
        "English": "Adjust the stressed sensors today and review the next reading instead of waiting for visible damage.",
        "Hindi": "आज ही तनाव वाले सेंसर मान सुधारें और दिखने वाले नुकसान का इंतजार करने के बजाय अगली रीडिंग देखें।",
    },
    "maintain_schedule": {
        "English": "Keep the current schedule, but continue routine monitoring to catch drift early.",
        "Hindi": "मौजूदा शेड्यूल बनाए रखें, लेकिन बदलाव जल्दी पकड़ने के लिए नियमित निगरानी जारी रखें।",
    },
    "increase_irrigation": {
        "English": "Soil moisture is low. Increase irrigation frequency slightly and inspect drip lines for blockage.",
        "Hindi": "मिट्टी की नमी कम है। सिंचाई की आवृत्ति थोड़ी बढ़ाएं और ड्रिप लाइन में रुकावट जांचें।",
    },
    "reduce_irrigation": {
        "English": "Soil moisture is high. Reduce irrigation and check drainage to avoid root-zone stress.",
        "Hindi": "मिट्टी की नमी अधिक है। सिंचाई कम करें और जल निकास जांचें ताकि जड़ों पर तनाव न बढ़े।",
    },
    "cool_canopy": {
        "English": "Temperature is high. Use shade, irrigate during cooler hours, and improve airflow around the crop.",
        "Hindi": "तापमान अधिक है। छाया दें, ठंडे समय में सिंचाई करें, और फसल के आसपास वायु प्रवाह बढ़ाएं।",
    },
    "protect_from_cold": {
        "English": "Temperature is low. Protect the crop from cold stress and avoid late-evening watering.",
        "Hindi": "तापमान कम है। फसल को ठंड से बचाएं और देर शाम सिंचाई से बचें।",
    },
    "improve_airflow": {
        "English": "Humidity is high. Improve airflow and reduce long leaf-wetness periods to lower disease pressure.",
        "Hindi": "नमी अधिक है। वायु प्रवाह बढ़ाएं और पत्तियों पर लंबे समय तक पानी रहने से बचें।",
    },
    "preserve_humidity": {
        "English": "Humidity is low. Use mulching or light misting so the plant does not dehydrate under heat.",
        "Hindi": "नमी कम है। मल्चिंग या हल्की मिस्टिंग करें ताकि गर्मी में पौधा निर्जलित न हो।",
    },
    "reduce_harsh_light": {
        "English": "Light intensity is high. Add shade during peak noon hours to reduce photo-stress.",
        "Hindi": "प्रकाश तीव्रता अधिक है। दोपहर के चरम समय में छाया दें ताकि फोटो-तनाव कम हो।",
    },
    "improve_light_exposure": {
        "English": "Light intensity is low. Improve exposure by trimming obstruction or moving the plant to better light.",
        "Hindi": "प्रकाश तीव्रता कम है। रुकावट हटाकर या पौधे को बेहतर रोशनी वाली जगह रखकर एक्सपोजर बढ़ाएं।",
    },
    "lower_ph": {
        "English": "Soil pH is high. Lower it gradually with sulfur-based amendments or acidic organic matter.",
        "Hindi": "मिट्टी pH अधिक है। सल्फर आधारित संशोधन या अम्लीय जैविक पदार्थ से इसे धीरे-धीरे कम करें।",
    },
    "raise_ph": {
        "English": "Soil pH is low. Apply lime or dolomite carefully and recheck after the next watering cycle.",
        "Hindi": "मिट्टी pH कम है। चूना या डोलोमाइट सावधानी से दें और अगली सिंचाई के बाद दोबारा जांचें।",
    },
    "boost_nitrogen": {
        "English": "Nitrogen is low. Apply a split nitrogen dose instead of one heavy correction.",
        "Hindi": "नाइट्रोजन कम है। एक बार में भारी मात्रा देने के बजाय भागों में नाइट्रोजन दें।",
    },
    "reduce_nitrogen": {
        "English": "Nitrogen is high. Reduce nitrogen input to avoid soft growth and secondary stress.",
        "Hindi": "नाइट्रोजन अधिक है। नरम वृद्धि और द्वितीयक तनाव से बचने के लिए नाइट्रोजन कम करें।",
    },
    "boost_phosphorus": {
        "English": "Phosphorus is low. Correct it with a phosphorus-rich input and check root activity.",
        "Hindi": "फॉस्फोरस कम है। फॉस्फोरस-समृद्ध इनपुट दें और जड़ों की सक्रियता जांचें।",
    },
    "reduce_phosphorus": {
        "English": "Phosphorus is high. Pause phosphorus-heavy fertilizer until the next soil review.",
        "Hindi": "फॉस्फोरस अधिक है। अगली मिट्टी समीक्षा तक फॉस्फोरस-भारी खाद रोकें।",
    },
    "boost_potassium": {
        "English": "Potassium is low. Add potassium support to improve stress tolerance and leaf strength.",
        "Hindi": "पोटैशियम कम है। तनाव सहनशीलता और पत्तियों की मजबूती बढ़ाने के लिए पोटैशियम दें।",
    },
    "reduce_potassium": {
        "English": "Potassium is high. Review the fertilizer mix so it does not suppress other nutrients.",
        "Hindi": "पोटैशियम अधिक है। उर्वरक मिश्रण की समीक्षा करें ताकि अन्य पोषक तत्व दबें नहीं।",
    },
    "inspect_leaf_health": {
        "English": "Chlorophyll is low. Inspect leaves for nutrient deficiency, chlorosis, or delayed recovery.",
        "Hindi": "क्लोरोफिल कम है। पत्तियों में पोषक कमी, क्लोरोसिस, या धीमी रिकवरी की जांच करें।",
    },
    "repeat_sensor_check": {
        "English": "Electrochemical stress signal is elevated. Recheck the sensor and inspect for hidden physiological stress.",
        "Hindi": "इलेक्ट्रोकेमिकल तनाव सिग्नल बढ़ा हुआ है। सेंसर दोबारा जांचें और छिपे शारीरिक तनाव की जांच करें।",
    },
}

SENSOR_ACTIONS = {
    "Soil_Moisture": {"low": "increase_irrigation", "high": "reduce_irrigation"},
    "Ambient_Temperature": {"low": "protect_from_cold", "high": "cool_canopy"},
    "Soil_Temperature": {"low": "protect_from_cold", "high": "cool_canopy"},
    "Humidity": {"low": "preserve_humidity", "high": "improve_airflow"},
    "Light_Intensity": {"low": "improve_light_exposure", "high": "reduce_harsh_light"},
    "Soil_pH": {"low": "raise_ph", "high": "lower_ph"},
    "Nitrogen_Level": {"low": "boost_nitrogen", "high": "reduce_nitrogen"},
    "Phosphorus_Level": {"low": "boost_phosphorus", "high": "reduce_phosphorus"},
    "Potassium_Level": {"low": "boost_potassium", "high": "reduce_potassium"},
    "Chlorophyll_Content": {"low": "inspect_leaf_health", "high": None},
    "Electrochemical_Signal": {"low": None, "high": "repeat_sensor_check"},
}

STATUS_PRIORITY = {"Healthy": 0, "Moderate Stress": 1, "High Stress": 2}


def localize(text: str, language: str) -> str:
    return TRANSLATIONS.get(text, {}).get(language, text)


def display_name(feature: str, language: str) -> str:
    return DISPLAY_NAMES.get(feature, {}).get(language, feature)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class FeatureProfile:
    median: float
    lower: float
    upper: float
    std: float
    train_min: float
    train_max: float


class SmartPlantAdvisor:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent)
        assets = load_assets(self.base_dir)
        self.model = assets["model"]
        self.scaler = assets["scaler"]
        self.label_encoder = assets["label_encoder"]
        self.feature_profiles = assets["feature_profiles"]
        self.feature_weights = assets["feature_weights"]
        self.examples = assets["examples"]

    def _prepare_frame(self, data: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, dict):
            frame = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            frame = data.to_frame().T
        else:
            frame = data.copy()

        missing = [feature for feature in FEATURE_COLUMNS if feature not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        ordered = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        if ordered.isna().any().any():
            raise ValueError("All feature columns must contain numeric values.")

        return ordered

    def get_example_scenarios(self) -> dict[str, dict[str, float]]:
        return self.examples

    def diagnose(self, record: dict[str, Any] | pd.Series | pd.DataFrame, language: str = "English") -> dict[str, Any]:
        frame = self._prepare_frame(record)
        row = frame.iloc[0]
        scaled = self.scaler.transform(frame)
        probabilities = self.model.predict_proba(scaled)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_status = str(self.label_encoder.inverse_transform([predicted_index])[0])
        probability_map = {
            str(label): float(probabilities[idx])
            for idx, label in enumerate(self.label_encoder.classes_)
        }

        issues = self._rank_sensor_issues(row)
        risk_score = self._compute_risk_score(predicted_status, probability_map, issues)
        alert_level = self._alert_level(risk_score)
        validation_notes = self._validation_notes(row, language)
        recommendations = self._recommendations(predicted_status, issues, language)

        return {
            "predicted_status": predicted_status,
            "predicted_status_localized": localize(predicted_status, language),
            "confidence": float(probability_map[predicted_status]),
            "probabilities": probability_map,
            "risk_score": float(risk_score),
            "alert_level": alert_level,
            "alert_level_localized": localize(alert_level, language),
            "drivers": [self._localize_issue(issue, language) for issue in issues[:5]],
            "recommendations": recommendations,
            "summary": self._summary(predicted_status, probability_map[predicted_status], issues, language),
            "validation_notes": validation_notes,
            "raw_values": {feature: float(row[feature]) for feature in FEATURE_COLUMNS},
        }

    def predict_dataframe(self, frame: pd.DataFrame, language: str = "English") -> pd.DataFrame:
        features = self._prepare_frame(frame)
        diagnoses = [self.diagnose(features.iloc[idx], language=language) for idx in range(len(features))]

        result = frame.copy()
        result["predicted_status"] = [item["predicted_status"] for item in diagnoses]
        result["confidence"] = [round(item["confidence"], 4) for item in diagnoses]
        result["risk_score"] = [round(item["risk_score"], 2) for item in diagnoses]
        result["alert_level"] = [item["alert_level"] for item in diagnoses]
        result["primary_driver"] = [
            item["drivers"][0]["feature_label"] if item["drivers"] else "Balanced"
            for item in diagnoses
        ]
        result["driver_direction"] = [
            item["drivers"][0]["direction_label"] if item["drivers"] else localize("balanced", language)
            for item in diagnoses
        ]
        result["action_summary"] = [
            " | ".join(item["recommendations"][:2])
            for item in diagnoses
        ]
        result["priority_rank"] = (
            result["risk_score"].rank(ascending=False, method="dense").astype(int)
        )

        return result.sort_values(["risk_score", "confidence"], ascending=[False, False]).reset_index(drop=True)

    def append_monitoring_log(
        self,
        diagnosis: dict[str, Any],
        destination: str | Path | None = None,
    ) -> Path:
        log_path = Path(destination or self.base_dir / "outputs" / "monitoring" / "prediction_log.csv")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "Logged_At": pd.Timestamp.utcnow().isoformat(),
            **diagnosis["raw_values"],
            "Predicted_Status": diagnosis["predicted_status"],
            "Confidence": round(diagnosis["confidence"], 4),
            "Risk_Score": round(diagnosis["risk_score"], 2),
            "Alert_Level": diagnosis["alert_level"],
            "Summary": diagnosis["summary"],
        }
        frame = pd.DataFrame([record])

        if log_path.exists():
            frame.to_csv(log_path, mode="a", header=False, index=False)
        else:
            frame.to_csv(log_path, index=False)

        return log_path

    def _rank_sensor_issues(self, row: pd.Series) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        for feature in FEATURE_COLUMNS:
            profile = self.feature_profiles[feature]
            value = float(row[feature])
            direction = "balanced"
            delta = 0.0

            if value < profile.lower:
                direction = "low"
                delta = (profile.lower - value) / profile.std
            elif value > profile.upper:
                direction = "high"
                delta = (value - profile.upper) / profile.std

            severity = clamp(delta * 22.0, 0.0, 100.0)
            weighted_score = severity * (0.7 + self.feature_weights[feature])

            issues.append(
                {
                    "feature": feature,
                    "value": value,
                    "direction": direction,
                    "severity": round(severity, 2),
                    "weighted_score": round(weighted_score, 2),
                    "healthy_low": round(profile.lower, 2),
                    "healthy_high": round(profile.upper, 2),
                    "unit": UNITS.get(feature, ""),
                }
            )

        issues.sort(key=lambda item: item["weighted_score"], reverse=True)
        return [issue for issue in issues if issue["direction"] != "balanced"]

    def _compute_risk_score(
        self,
        predicted_status: str,
        probability_map: dict[str, float],
        issues: list[dict[str, Any]],
    ) -> float:
        model_risk = 100.0 * (
            probability_map.get("Moderate Stress", 0.0) * 0.55
            + probability_map.get("High Stress", 0.0)
        )
        sensor_pressure = np.mean([issue["severity"] for issue in issues[:4]]) if issues else 0.0
        risk_score = (model_risk * 0.72) + (sensor_pressure * 0.28)

        confidence = probability_map.get(predicted_status, 0.0)
        if predicted_status == "High Stress":
            risk_score = max(risk_score, 70.0 + (confidence * 25.0))
        elif predicted_status == "Moderate Stress":
            risk_score = max(risk_score, 42.0 + (confidence * 18.0))
        else:
            risk_score = min(risk_score, 45.0)

        return round(clamp(risk_score, 0.0, 100.0), 2)

    def _alert_level(self, risk_score: float) -> str:
        if risk_score >= 80:
            return "Critical"
        if risk_score >= 60:
            return "High"
        if risk_score >= 40:
            return "Watch"
        return "Stable"

    def _recommendations(
        self,
        predicted_status: str,
        issues: list[dict[str, Any]],
        language: str,
    ) -> list[str]:
        action_ids: list[str] = []
        if predicted_status == "High Stress":
            action_ids.append("inspect_within_24h")
        elif predicted_status == "Moderate Stress":
            action_ids.append("tune_environment")
        else:
            action_ids.append("maintain_schedule")

        for issue in issues[:3]:
            action_id = SENSOR_ACTIONS.get(issue["feature"], {}).get(issue["direction"])
            if action_id and action_id not in action_ids:
                action_ids.append(action_id)

        return [ACTION_LIBRARY[action_id][language] for action_id in action_ids]

    def _validation_notes(self, row: pd.Series, language: str) -> list[str]:
        notes: list[str] = []
        message = localize(
            "Sensor reading is outside the training range; verify the device and interpret carefully.",
            language,
        )
        for feature in FEATURE_COLUMNS:
            value = float(row[feature])
            profile = self.feature_profiles[feature]
            if value < profile.train_min or value > profile.train_max:
                notes.append(f"{display_name(feature, language)}: {message}")
        return notes

    def _summary(
        self,
        predicted_status: str,
        confidence: float,
        issues: list[dict[str, Any]],
        language: str,
    ) -> str:
        if issues:
            drivers = ", ".join(display_name(issue["feature"], language) for issue in issues[:3])
        else:
            drivers = {
                "English": "all major sensors staying near the healthy band",
                "Hindi": "मुख्य सेंसर मान स्वस्थ सीमा के पास बने हुए हैं",
            }[language]

        if language == "Hindi":
            return (
                f"{localize(predicted_status, language)} का अनुमान {confidence * 100:.1f}% विश्वास के साथ लगाया गया। "
                f"मुख्य दबाव बिंदु: {drivers}।"
            )

        return (
            f"{predicted_status} predicted with {confidence * 100:.1f}% confidence. "
            f"Main pressure points: {drivers}."
        )

    def _localize_issue(self, issue: dict[str, Any], language: str) -> dict[str, Any]:
        localized = issue.copy()
        localized["feature_label"] = display_name(issue["feature"], language)
        localized["direction_label"] = localize(issue["direction"], language)
        localized["healthy_band"] = f"{issue['healthy_low']} - {issue['healthy_high']} {issue['unit']}".strip()
        return localized


@lru_cache(maxsize=1)
def load_assets(base_dir: Path) -> dict[str, Any]:
    data_dir = base_dir / "data"
    model_dir = base_dir / "models"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(model_dir / "hybrid_voting_model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
        label_encoder = joblib.load(model_dir / "label_encoder.pkl")
        random_forest = joblib.load(model_dir / "random_forest_model.pkl")

    raw_data = pd.read_csv(data_dir / "plant_health_data.csv")
    profiles = build_feature_profiles(raw_data)
    weights = normalize_weights(dict(zip(FEATURE_COLUMNS, random_forest.feature_importances_)))
    examples = build_examples(raw_data)

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_profiles": profiles,
        "feature_weights": weights,
        "examples": examples,
    }


def build_feature_profiles(raw_data: pd.DataFrame) -> dict[str, FeatureProfile]:
    healthy = raw_data.loc[raw_data["Plant_Health_Status"] == "Healthy", FEATURE_COLUMNS]
    reference = healthy if not healthy.empty else raw_data[FEATURE_COLUMNS]
    profiles: dict[str, FeatureProfile] = {}

    for feature in FEATURE_COLUMNS:
        series = reference[feature].dropna()
        global_series = raw_data[feature].dropna()
        lower, median, upper = np.quantile(series, [0.25, 0.5, 0.75])
        std = float(series.std())
        if std <= 0:
            std = max(float((upper - lower) / 1.35), 1e-3)

        profiles[feature] = FeatureProfile(
            median=float(median),
            lower=float(lower),
            upper=float(upper),
            std=float(std),
            train_min=float(global_series.quantile(0.01)),
            train_max=float(global_series.quantile(0.99)),
        )

    return profiles


def normalize_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    max_weight = max(raw_weights.values()) if raw_weights else 1.0
    if max_weight <= 0:
        return {feature: 0.0 for feature in FEATURE_COLUMNS}
    return {feature: float(weight / max_weight) for feature, weight in raw_weights.items()}


def build_examples(raw_data: pd.DataFrame) -> dict[str, dict[str, float]]:
    examples: dict[str, dict[str, float]] = {}
    for status in ["Healthy", "Moderate Stress", "High Stress"]:
        subset = raw_data.loc[raw_data["Plant_Health_Status"] == status, FEATURE_COLUMNS]
        if subset.empty:
            continue
        examples[f"{status} baseline"] = {
            feature: round(float(subset[feature].median()), 2) for feature in FEATURE_COLUMNS
        }
    return examples

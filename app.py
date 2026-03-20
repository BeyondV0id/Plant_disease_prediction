from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from smart_advisor import FEATURE_COLUMNS, SmartPlantAdvisor, UNITS, display_name, localize


BASE_DIR = Path(__file__).resolve().parent

UI_COPY = {
    "English": {
        "title": "Smart Crop Stress Advisor",
        "subtitle": "A sensor-driven early warning dashboard built on your hybrid plant health model.",
        "language": "Language",
        "scenario": "Example scenario",
        "load_scenario": "Load scenario values",
        "single_tab": "Single Plant Check",
        "batch_tab": "Batch Triage",
        "story_tab": "Project Story",
        "predict": "Run diagnosis",
        "status": "Predicted status",
        "confidence": "Confidence",
        "risk": "Risk score",
        "alert": "Alert level",
        "probabilities": "Class probabilities",
        "drivers": "Top stress drivers",
        "actions": "Recommended actions",
        "notes": "Validation notes",
        "summary": "Decision summary",
        "batch_header": "Upload a CSV for farm-wide prioritization",
        "batch_help": "Required columns: "
        + ", ".join(FEATURE_COLUMNS),
        "download": "Download triage CSV",
        "top_rows": "Highest-risk records",
        "trend": "Risk trend",
        "story_header": "Why this version is stronger than a basic notebook project",
        "resume_header": "Resume bullets you can use",
        "flow_header": "Architecture flow",
        "upload_error": "The CSV could not be processed. Check that all required feature columns are present and numeric.",
        "rows": "Rows scored",
        "critical_rows": "Critical alerts",
        "high_stress": "High stress rows",
        "avg_risk": "Average risk",
        "plant_filter": "Plant ID",
        "monitoring_log": "Prediction logged to outputs/monitoring/prediction_log.csv",
    },
    "Hindi": {
        "title": "स्मार्ट क्रॉप स्ट्रेस एडवाइजर",
        "subtitle": "आपके हाइब्रिड प्लांट हेल्थ मॉडल पर आधारित सेंसर-ड्रिवन अर्ली वार्निंग डैशबोर्ड।",
        "language": "भाषा",
        "scenario": "उदाहरण स्थिति",
        "load_scenario": "उदाहरण मान लोड करें",
        "single_tab": "एक पौधे की जांच",
        "batch_tab": "बैच ट्रायेज",
        "story_tab": "प्रोजेक्ट स्टोरी",
        "predict": "डायग्नोसिस चलाएं",
        "status": "अनुमानित स्थिति",
        "confidence": "विश्वास",
        "risk": "रिस्क स्कोर",
        "alert": "अलर्ट स्तर",
        "probabilities": "क्लास संभावनाएं",
        "drivers": "मुख्य तनाव कारण",
        "actions": "सुझाए गए कदम",
        "notes": "सत्यापन नोट्स",
        "summary": "निर्णय सारांश",
        "batch_header": "पूरे खेत की प्राथमिकता के लिए CSV अपलोड करें",
        "batch_help": "आवश्यक कॉलम: " + ", ".join(FEATURE_COLUMNS),
        "download": "ट्रायेज CSV डाउनलोड करें",
        "top_rows": "सबसे अधिक जोखिम वाले रिकॉर्ड",
        "trend": "रिस्क ट्रेंड",
        "story_header": "यह संस्करण बेसिक नोटबुक प्रोजेक्ट से अधिक मजबूत क्यों है",
        "resume_header": "रिज्यूमे के लिए उपयोगी बुलेट्स",
        "flow_header": "आर्किटेक्चर फ्लो",
        "upload_error": "CSV प्रोसेस नहीं हो सका। सभी आवश्यक फीचर कॉलम और न्यूमेरिक मान जांचें।",
        "rows": "स्कोर की गई पंक्तियां",
        "critical_rows": "गंभीर अलर्ट",
        "high_stress": "उच्च तनाव पंक्तियां",
        "avg_risk": "औसत जोखिम",
        "plant_filter": "प्लांट ID",
        "monitoring_log": "प्रेडिक्शन outputs/monitoring/prediction_log.csv में लॉग हो गया।",
    },
}


@st.cache_resource
def get_advisor() -> SmartPlantAdvisor:
    return SmartPlantAdvisor(BASE_DIR)


def copy(language: str, key: str) -> str:
    return UI_COPY[language][key]


def load_example_into_state(example_values: dict[str, float]) -> None:
    for feature in FEATURE_COLUMNS:
        st.session_state[feature] = float(example_values[feature])


def initialize_state(advisor: SmartPlantAdvisor) -> None:
    examples = advisor.get_example_scenarios()
    default_example = next(iter(examples.values()))
    for feature in FEATURE_COLUMNS:
        st.session_state.setdefault(feature, float(default_example[feature]))


def probability_frame(probabilities: dict[str, float], language: str) -> pd.DataFrame:
    rows = []
    for status, probability in probabilities.items():
        rows.append(
            {
                "Status": localize(status, language),
                "Probability": round(probability, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("Probability", ascending=False)


def driver_frame(drivers: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for driver in drivers:
        rows.append(
            {
                "Sensor": driver["feature_label"],
                "Current": round(float(driver["value"]), 2),
                "Direction": driver["direction_label"],
                "Healthy Band": driver["healthy_band"],
                "Severity": float(driver["severity"]),
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="Smart Crop Stress Advisor", layout="wide")
st.markdown(
    """
    <style>
    .hero {
        padding: 1.1rem 1.3rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #143109 0%, #2f5233 45%, #d6f5d6 140%);
        color: #f7fff7;
        margin-bottom: 1rem;
    }
    .story-card {
        padding: 1rem 1.1rem;
        border-radius: 16px;
        background: #f4f7f1;
        border: 1px solid #dfe9d7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

advisor = get_advisor()
initialize_state(advisor)

language = st.sidebar.selectbox(copy("English", "language"), ["English", "Hindi"])
examples = advisor.get_example_scenarios()
selected_scenario = st.sidebar.selectbox(copy(language, "scenario"), list(examples.keys()))
if st.sidebar.button(copy(language, "load_scenario")):
    load_example_into_state(examples[selected_scenario])

st.markdown(
    f"""
    <div class="hero">
        <h1 style="margin:0;">{copy(language, "title")}</h1>
        <p style="margin:0.4rem 0 0 0;">{copy(language, "subtitle")}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

single_tab, batch_tab, story_tab = st.tabs(
    [
        copy(language, "single_tab"),
        copy(language, "batch_tab"),
        copy(language, "story_tab"),
    ]
)

with single_tab:
    with st.form("single_prediction_form"):
        input_columns = st.columns(2)
        for idx, feature in enumerate(FEATURE_COLUMNS):
            with input_columns[idx % 2]:
                st.number_input(
                    f"{display_name(feature, language)} ({UNITS[feature]})",
                    key=feature,
                    step=0.1,
                    format="%.2f",
                )
        submitted = st.form_submit_button(copy(language, "predict"), use_container_width=True)

    if submitted:
        values = {feature: float(st.session_state[feature]) for feature in FEATURE_COLUMNS}
        diagnosis = advisor.diagnose(values, language=language)
        advisor.append_monitoring_log(diagnosis)

        metric_columns = st.columns(4)
        metric_columns[0].metric(copy(language, "status"), diagnosis["predicted_status_localized"])
        metric_columns[1].metric(copy(language, "confidence"), f"{diagnosis['confidence'] * 100:.1f}%")
        metric_columns[2].metric(copy(language, "risk"), f"{diagnosis['risk_score']:.1f}/100")
        metric_columns[3].metric(copy(language, "alert"), diagnosis["alert_level_localized"])

        st.subheader(copy(language, "summary"))
        st.write(diagnosis["summary"])
        st.caption(copy(language, "monitoring_log"))

        probability_columns = st.columns([1.1, 1.4])
        with probability_columns[0]:
            st.subheader(copy(language, "probabilities"))
            st.bar_chart(probability_frame(diagnosis["probabilities"], language).set_index("Status"))
        with probability_columns[1]:
            st.subheader(copy(language, "drivers"))
            if diagnosis["drivers"]:
                st.dataframe(driver_frame(diagnosis["drivers"]), use_container_width=True, hide_index=True)
            else:
                st.success(localize("Healthy", language))

        st.subheader(copy(language, "actions"))
        for recommendation in diagnosis["recommendations"]:
            st.info(recommendation)

        if diagnosis["validation_notes"]:
            st.subheader(copy(language, "notes"))
            for note in diagnosis["validation_notes"]:
                st.warning(note)

with batch_tab:
    st.subheader(copy(language, "batch_header"))
    st.caption(copy(language, "batch_help"))
    uploaded_file = st.file_uploader("CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_input = pd.read_csv(uploaded_file)
            triage = advisor.predict_dataframe(batch_input, language=language)
        except Exception:
            st.error(copy(language, "upload_error"))
        else:
            metrics = st.columns(4)
            metrics[0].metric(copy(language, "rows"), len(triage))
            metrics[1].metric(copy(language, "critical_rows"), int((triage["alert_level"] == "Critical").sum()))
            metrics[2].metric(copy(language, "high_stress"), int((triage["predicted_status"] == "High Stress").sum()))
            metrics[3].metric(copy(language, "avg_risk"), f"{triage['risk_score'].mean():.1f}")

            st.download_button(
                copy(language, "download"),
                data=triage.to_csv(index=False).encode("utf-8"),
                file_name="plant_stress_triage.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.subheader(copy(language, "top_rows"))
            st.dataframe(triage.head(20), use_container_width=True, hide_index=True)

            if "Timestamp" in triage.columns:
                trend_frame = triage.copy()
                trend_frame["Timestamp"] = pd.to_datetime(trend_frame["Timestamp"], errors="coerce")
                trend_frame = trend_frame.dropna(subset=["Timestamp"])
                if not trend_frame.empty:
                    st.subheader(copy(language, "trend"))
                    if "Plant_ID" in trend_frame.columns:
                        plant_choice = st.selectbox(
                            copy(language, "plant_filter"),
                            sorted(trend_frame["Plant_ID"].dropna().unique().tolist()),
                        )
                        trend_frame = trend_frame.loc[trend_frame["Plant_ID"] == plant_choice]
                    trend_frame = trend_frame.sort_values("Timestamp")
                    st.line_chart(trend_frame.set_index("Timestamp")["risk_score"])

with story_tab:
    st.subheader(copy(language, "story_header"))
    st.markdown(
        """
        <div class="story-card">
            <p><strong>Not just classification:</strong> the project now converts raw sensor readings into an alert level, a 0-100 risk score, and field actions.</p>
            <p><strong>Product layer:</strong> a Streamlit dashboard supports one-plant diagnosis and farm-wide batch triage from CSV uploads.</p>
            <p><strong>Farmer-facing thinking:</strong> the output explains which sensors are driving stress and what to fix first.</p>
            <p><strong>Operational value:</strong> every diagnosis can be logged for monitoring, which makes the project closer to a real deployment workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(copy(language, "resume_header"))
    st.markdown(
        """
        - Built an IoT crop stress early-warning system using a hybrid VotingClassifier over soil, nutrient, temperature, and physiological sensor data.
        - Added a risk engine that converts model probabilities and healthy-band deviation into a 0-100 triage score with actionable field recommendations.
        - Developed a Streamlit dashboard for single-plant diagnosis, batch CSV prioritization, multilingual guidance, and monitoring log generation.
        """
    )

    st.subheader(copy(language, "flow_header"))
    st.code(
        "Sensor readings -> scaler -> hybrid voting model -> risk scoring -> stress driver analysis -> field recommendations -> monitoring log",
        language="text",
    )

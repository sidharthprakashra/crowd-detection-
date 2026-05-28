import streamlit as st
from predict import predict_crowd

st.set_page_config(
    page_title="Crowd Forecast AI",
    layout="centered"
)

st.title("🔥 Crowd Density Prediction System")

# ======================================================
# INPUTS
# ======================================================

day_type = st.selectbox(
    "Select Day Type",
    [
        "normal_days",
        "exam_special_days"
    ]
)

location = st.selectbox(
    "Select Location",
    [
        "canteen",
        "gate1",
        "gate2",
        "gate3",
        "ground",
        "open_auditorium"
    ]
)

time_slot = st.selectbox(
    "Select Time",
    [
        "morning",
        "afternoon",
        "evening"
    ]
)

# ======================================================
# PREDICT
# ======================================================

if st.button("Predict Crowd"):

    count, level = predict_crowd(
        day_type,
        location,
        time_slot
    )

    st.success(
        f"Estimated Crowd Count: {count}"
    )

    st.metric(
        "Crowd Level",
        level
    )

    # COLOR INDICATOR
    if level == "LOW":
        st.info("Low crowd expected")

    elif level == "MEDIUM":
        st.warning("Medium crowd expected")

    else:
        st.error("High crowd expected")
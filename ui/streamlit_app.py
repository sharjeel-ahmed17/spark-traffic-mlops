import streamlit as st
import requests
import os
from dotenv import load_dotenv
load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Traffic Vehicle Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Traffic Vehicle Prediction")
st.markdown("Predict vehicle count based on traffic junction and time parameters")

st.divider()

with st.form("prediction_form"):
    st.subheader("Enter Prediction Parameters")

    col1, col2 = st.columns(2)

    with col1:
        junction = st.selectbox(
            "Junction",
            options=[1, 2, 3, 4],
            help="Select the traffic junction (1-4)"
        )
        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(1, 8)),
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x - 1],
            help="Select the day of the week (1=Monday, 7=Sunday)"
        )

    with col2:
        hour = st.slider("Hour", 0, 23, 12, help="Select the hour of the day (0-23)")
        month = st.slider("Month", 1, 12, 6, help="Select the month (1-12)")

    year = st.number_input("Year", min_value=2020, max_value=2030, value=2024, step=1)

    st.divider()

    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

if submitted:
    with st.spinner("Making prediction..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                params={
                    "junction": float(junction),
                    "day_of_week": float(day_of_week),
                    "hour": float(hour),
                    "month": float(month),
                    "year": float(year)
                }
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    st.success("✅ Prediction successful!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Vehicles", data["predicted_vehicles"])
                    with col2:
                        st.metric("Model", data["model_name"])

                    st.info(f"**Input Parameters:** Junction {int(junction)}, {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][int(day_of_week) - 1]}, {int(hour)}:00, Month {int(month)}, Year {int(year)}")
                else:
                    st.error(f"❌ Prediction failed: {data.get('error', 'Unknown error')}")
            else:
                st.error(f"❌ API Error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the server is running at " + API_URL)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.divider()
st.caption("Built with Streamlit | Traffic Vehicle Prediction API")
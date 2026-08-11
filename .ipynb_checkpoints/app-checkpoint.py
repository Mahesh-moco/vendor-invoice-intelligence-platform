import streamlit as st

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("📦 Vendor Invoice Intelligence Portal")

st.markdown("""
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning to:

- 🚚 Forecast freight costs accurately
- 🚨 Detect risky or abnormal vendor invoices
- ⚡ Reduce financial leakage and manual workload
""")

st.markdown("---")

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("🔍 Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    (
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    )
)

st.sidebar.markdown("""
---

### Business Impact

- 📉 Improved cost forecasting
- 🚨 Reduced invoice fraud & anomalies
- ⚙️ Faster finance operations
""")

# ============================================================
# Freight Cost Prediction
# ============================================================

if selected_model == "Freight Cost Prediction":

    st.subheader("🚛 Freight Cost Prediction")

    st.write(
        "Predict freight cost using Invoice Dollars."
    )

    with st.form("freight_form"):

        dollars = st.number_input(
            "💰 Invoice Dollars",
            min_value=1.0,
            value=18500.0
        )

        submit_freight = st.form_submit_button(
            "🎯 Predict Freight Cost"
        )

    if submit_freight:

        input_data = {
            "Dollars": [dollars]
        }

        prediction = predict_freight_cost(input_data)

        st.success("Prediction completed successfully!")

        st.metric(
            "Estimated Freight Cost",
            f"${prediction['Predicted_Freight'].iloc[0]:,.2f}"
        )

        st.dataframe(prediction)

# ============================================================
# Invoice Flag Prediction
# ============================================================

else:

    st.subheader("🚨 Invoice Manual Approval Prediction")

    st.write(
        "Predict whether an invoice requires manual approval."
    )

    with st.form("invoice_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )

            freight = st.number_input(
                "Freight",
                min_value=0.0,
                value=1.73
            )

        with col2:

            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=352.95
            )

            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=162
            )

        with col3:

            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=2476.0
            )

            avg_receiving_delay = st.number_input(
                "Average Receiving Delay",
                min_value=0.0,
                value=5.0
            )

        submit_flag = st.form_submit_button(
            "🧠 Evaluate Invoice Risk"
        )

    if submit_flag:

        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars],
            "avg_receiving_delay": [avg_receiving_delay]
        }

        prediction = predict_invoice_flag(input_data)

        flag = prediction["Predicted_Flag"].iloc[0]

        if flag == 1:
            st.error("🚨 Invoice requires MANUAL APPROVAL")
        else:
            st.success("✅ Invoice is SAFE for Auto Approval")

        st.dataframe(prediction)
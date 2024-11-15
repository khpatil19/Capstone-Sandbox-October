import pandas as pd
import streamlit as st
import pickle

st.set_page_config(
    page_title="Capstone-demo",
    page_icon="🏗️",
)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select a Page",
    ["Ad Performance Prediction", "Additional Insights"]
)

# Apply custom CSS for further branding
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

if page == "Ad Performance Prediction":
    st.title("🏗️ Capstone Model Demonstration")
    st.subheader("Prototype of the project MVP")

    with st.form("form1", clear_on_submit=False):
        st.subheader("Enter the following details of your Google Ad")

        search_keyword_input = st.selectbox(
            'Search keyword',
            ('Labelmaster',)
        )

        campaign_input = st.selectbox(
            'Campaign',
            ('Branded', 'IATA')
        )

        ad_group_input = st.selectbox(
            'Ad Group',
            ('Hazard Placards',)
        )

        keyword_max_cpc_input = st.slider('Enter the max CPC', 1, 10, 3)

        ad_type_input = st.selectbox(
            'Ad type',
            ('Responsive search ad',)
        )

        ad_name_input = st.selectbox(
            'Ad Name',
            ('Spring Safety Ad',)
        )

        submit = st.form_submit_button("Get predicted performance")

        # Load the model from the pickle file
        with open("random_forest_regressor_2.pkl", "rb") as file:
            rf_pipeline = pickle.load(file)

        if submit:
            # Constructing the dictionary with the collected inputs
            sample_input = {
                'Search keyword': [search_keyword_input],
                'Campaign': [campaign_input],
                'Ad group': [ad_group_input],
                'Keyword max CPC': [keyword_max_cpc_input],
                'Ad type': [ad_type_input],
                'Ad name': [ad_name_input]
            }

            # Convert the dictionary to a DataFrame
            input_data = pd.DataFrame(sample_input)

            # Use the trained pipeline to make a prediction
            predicted_values = rf_pipeline.predict(input_data)

            # Display predictions in a pivoted tabular format using HTML
            st.markdown(f"""
                <h2 style="color:#FF362A;">Expected Ad Performance</h2>
                <table style="width:100%; background-color:#FFCA3A; color:#4E5056; text-align:center;">
                    <tr>
                        <th>Metric</th>
                        <th>Conversions</th>
                        <th>Clicks</th>
                        <th>Impressions</th>
                    </tr>
                    <tr>
                        <td>Value</td>
                        <td>{predicted_values[0][0]:.2f}</td>
                        <td>{predicted_values[0][1]:.2f}</td>
                        <td>{predicted_values[0][2]:.2f}</td>
                    </tr>
                </table>
            """, unsafe_allow_html=True)

elif page == "Additional Insights":
    st.title("📊 Additional Insights")
    st.subheader("Explore insights about campaign data and performance metrics.")
    
    st.markdown("""
    <p style="color:#4E5056;">
    This section will provide additional insights, trends, or data visualizations to help 
    users better understand campaign performance and make data-driven decisions.
    </p>
    """, unsafe_allow_html=True)
    # Placeholder for future data or visualizations
    st.write("Coming soon...")
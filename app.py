import pandas as pd
import streamlit as st
import pickle

st.set_page_config(
    page_title="Capstone-demo",
    page_icon="🏗️",
)

st.title("🏗️ Capstone model demonstration")
st.subheader("Prototype of the project MVP")

# Load the model from the pickle file
with open("path/to/your/model/random_forest_regressor_2.pkl", "rb") as file:
    rf_pipeline = pickle.load(file)

with st.form("form1", clear_on_submit=False):

    st.subheader("Enter the following details of your Google Ad")

    search_keyword_input = st.selectbox(
        'Search keyword',
        ('Labelmaster'))

    campaign_input = st.selectbox(
        'Campaign',
        ('Branded', 'IATA'))

    ad_group_input = st.selectbox(
        'Ad Group',
        ('Hazard Placards'))

    keyword_max_cpc_input = st.slider('Enter the max CPC', 1, 10, 3)

    ad_type_input = st.selectbox(
        'Ad type',
        ('Responsive search ad'))

    ad_name_input = st.selectbox(
        'Ad Name',
        ('Spring Safety Ad'))

    submit = st.form_submit_button("Get predicted performance")

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

        st.write(f"Predicted Conversions: {predicted_values[0][0]:.2f}")
        st.write(f"Predicted Clicks: {predicted_values[0][1]:.2f}")
        st.write(f"Predicted Impressions: {predicted_values[0][2]:.2f}")


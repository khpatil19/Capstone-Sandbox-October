import pandas as pd
import streamlit as st
import joblib
from joblib import load

st.set_page_config(
    page_title="Capstone-demo",
    page_icon="🏗️",
)

st.title("🏗️ Capstone model demonstration")
st.subheader("Prototype of the project MVP")

with st.form("form1", clear_on_submit= False): 

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

    # Loading the model trained
    # @st.cache(allow_output_mutation=True)
    # import joblib
    # def load_model():
    #     return joblib.load('khpatil19/Capstone-Sandbox-October/random_forest_regressor.joblib')

    # rf_pipeline = load_model()

    rf_pipeline = load(".random_forest_regressor_2.joblib")

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

    # Assuming rf_pipeline is the loaded model pipeline that includes preprocessing and regression model
    # Use the trained pipeline to make a prediction
    predicted_values = rf_pipeline.predict(input_data)

    # Output the predicted number of conversions, clicks, and impressions
    # print(f"Predicted Conversions: {predicted_values[0][0]:.2f}")
    # print(f"Predicted Clicks: {predicted_values[0][1]:.2f}")
    # print(f"Predicted Impressions: {predicted_values[0][2]:.2f}")



    st.write(
        "Predicted Conversions: {predicted_values[0][0]:.2f}"
        "Predicted Clicks: {predicted_values[0][1]:.2f}"
        "Predicted Impressions: {predicted_values[0][2]:.2f}"
    )

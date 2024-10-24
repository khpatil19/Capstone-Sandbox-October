import pandas as pd
import streamlit as st

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


    # Loading the model trained
    rf_pipeline = load('khpatil19/Capstone-Sandbox-October/random_forest_regressor.joblib')

    # Getting some sample input
    sample_input = {
        'Search keyword': ['hazardous materials management'], 
        'Campaign': ['Branded'],  
        'Ad group': ['Hazard Placards'],  
        'Keyword max CPC': [4],  
        'Ad type': ['Responsive search ad'], 
        'Ad name': ['Spring Safety Ad']  
    }

    input_data = pd.DataFrame(sample_input)

    # Use the trained pipeline to make a prediction
    predicted_conversions = rf_pipeline.predict(input_data)

    # # Output the predicted number of conversions
    # print(f"Predicted Conversions: {predicted_conversions[0]}")



    st.write(
        "Predicted Conversions: {predicted_conversions[0]}"
    )

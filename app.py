import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools
import random
import streamlit as st
import pickle

st.set_page_config(
    page_title="Labelmaster Ad Tool",
    page_icon="lm-favicon.ico",
)

st.image(
    "Cover_Image.png",
    use_column_width=True,
)

# Add an image to the sidebar
st.sidebar.image(
    "Red_Logo.png",
    use_column_width=True
)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select a Page",
    ["Ad Performance Prediction", "Optimized Bidding Suggestion", "Additional Insights"]
)

# Apply custom CSS
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #EE2833; /* Replace with your desired color */
        color: white; /* Optional: Change text color */
    }

    /* Style the selectbox dropdown */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #F03434; /* Change the background color */
        color: #FFFFFF; /* Change the text color */
        border-radius: 9.5px; /* Optional: Add rounded corners */
        padding: 5px; /* Optional: Add padding for better spacing */
    }

    /* Style the options inside the dropdown */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #FFFFFF; /* Options background */
    }

    /* Hover effect for options */
    .stSelectbox div[data-baseweb="select"] > div:hover {
        background-color: #FFCA3A; /* Highlight color */
    }
            
    .custom-form {
        background-color: #F03434; /* Light gray background */
        padding: 20px;
        border-radius: 20px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); /* Optional shadow for depth */
    }
    
    .big-font {
        font-size:300px !important;
    }
            
    </style>
""", unsafe_allow_html=True)

if page == "Ad Performance Prediction":
    st.title("Ad Performance Prediction")
    st.subheader("Random Forest Regressor Model Trained on Advertising data (2021-2024)")

    with st.form("form1", clear_on_submit=False):
        st.subheader("Enter the following details of your Google Ad")

        # Populate select boxes with all possible options
        search_keyword_input = st.selectbox(
            'Search keyword',
            (
                "Labelmaster", "hazard labels", "hazmat labels", "dot placards", "manual iata dangerous goods regulations",
                "latest edition of iata dgr", "return recalled batteries", "recall ready batteries", "quality control jobs"
                # Add the full list of search keywords here
            )
        )

        campaign_input = st.selectbox(
            'Campaign',
            (
                "Labels", "Training 2024", "Placards", "Branded", "IATA USA", "Lithium Battery Shipping", "ERG",
                "Software 2024", "Labels - West Coast Shipping", "Packaging - Hazmat Packaging",
                # Add the full list of campaigns here
            )
        )

        ad_group_input = st.selectbox(
            'Ad Group',
            (
                "Hazmat Training", "Hazmat Labels", "IATA DGR", "ERG", "Labels West Coast Shipping",
                "Packo-centric", "Supply Chain Coordinator", "Non-RCRA Regulated Waste Labels",
                # Add the full list of ad groups here
            )
        )

        ad_type_input = st.selectbox(
            'Ad type',
            ("Expanded text ad", "Responsive search ad", "Text ad")
        )

        ad_name_input = st.selectbox(
            'Ad Name',
            ("Spring Safety Ad", "Buy DOT Hazmat Labels Now", "California Waste Labels", "Inspection Labels",
             "DOT Hazardous Labels", "Order DOT Hazmat Labels")
        )

        keyword_max_cpc_input = st.slider('Enter the max CPC', 1, 10, 3)

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
                        <td>{predicted_values[0][0] * 100:.2f}</td>
                        <td>{predicted_values[0][1] * 100:.2f}</td>
                        <td>{predicted_values[0][2] * 100:.2f}</td>
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

elif page == "Optimized Bidding Suggestion":
    st.title("🚀 Optimized Bidding Suggestion")
    st.subheader("Find tailored recommendations for optimizing your campaigns.")
    st.markdown("""
    <p style="color:#4E5056;">
    Upload your CSV file to optimize advertising campaigns using Genetic Algorithm. Based on your campaign data, the program will provide you with actionable insights on:
                
    - High performance keywords
                
    - Action to take (Remove keyword/increase bid)
                
    - Optimized Bid
    </p>
    """, unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        # Load the dataset
        data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data")
        st.write(data.head(10))

        # Preprocessing
        st.write("Processing the uploaded data...")
        data['Day'] = pd.to_datetime(data['Day'])
        data['Impr.'] = pd.to_numeric(data['Impr.'], errors='coerce').replace(0, np.nan)
        data['Clicks'] = pd.to_numeric(data['Clicks'], errors='coerce').replace(0, np.nan)
        data['Conversions'] = pd.to_numeric(data['Conversions'], errors='coerce').replace(0, np.nan)
        data['Cost'] = pd.to_numeric(data['Cost'], errors='coerce')

        # Calculate performance metrics
        data['CTR'] = data['Clicks'] / data['Impr.']
        data['Cost_per_Conversion'] = data['Cost'] / data['Conversions']
        data['Conversion_Rate'] = data['Conversions'] / data['Clicks']

        # Filter non-zero conversions
        non_zero_conversions = data[data['Conversions'] != 0]


        # If there is invalid data, replace it first (e.g., replace with 0 or an appropriate default value)
        data['Clicks'] = pd.to_numeric(data['Clicks'], errors='coerce').fillna(0)
        data['Impr.'] = pd.to_numeric(data['Impr.'], errors='coerce').fillna(0)
        data['Conversions'] = pd.to_numeric(data['Conversions'], errors='coerce').fillna(0)

        # Recalculate
        data['CTR'] = np.where(data['Impr.'] == 0, 0, data['Clicks'] / data['Impr.'])
        data['Cost_per_Conversion'] = np.where(data['Conversions'] == 0, np.nan, data['Cost'] / data['Conversions'])
        data['Conversion_Rate'] = np.where(data['Clicks'] == 0, 0, data['Conversions'] / data['Clicks'])

        non_zero_conversions = data[data['Conversions'] != 0]

        # Calculate the average performance for each keyword while keeping the 'Ad Type', 'Ad Group', and 'Campaign'
        keyword_performance = non_zero_conversions.groupby(['Search keyword', 'Ad type', 'Ad group', 'Campaign'], as_index=False)[['CTR', 'Cost_per_Conversion', 'Conversion_Rate']].mean()
        # keyword_performance = non_zero_conversions.groupby('Search keyword', as_index=False)[['CTR', 'Cost_per_Conversion', 'Conversion_Rate']].mean()

        # Sort by conversion rate in descending order
        keyword_performance = keyword_performance.sort_values(by='Conversion_Rate', ascending=False)

        # Output the performance of the top 20 keywords
        st.markdown("####  The top 20 high performing keywords are -")
        st.write(keyword_performance.head(10))

        ctr_mean = non_zero_conversions['CTR'].mean()
        cpc_mean = non_zero_conversions['Cost_per_Conversion'].mean()
        cr_mean = non_zero_conversions['Conversion_Rate'].mean()


        high_performance_keywords = keyword_performance[
            (keyword_performance['CTR'] > ctr_mean) & 
            (keyword_performance['Conversion_Rate'] > cr_mean) &
            (keyword_performance['Cost_per_Conversion'] < cpc_mean)
        ]

        low_performance_keywords = keyword_performance[
            (keyword_performance['CTR'] < ctr_mean) | 
            (keyword_performance['Conversion_Rate'] < cr_mean) |
            (keyword_performance['Cost_per_Conversion'] > cpc_mean)
        ]


        # Calculate statistical information for high-performance keywords
        high_performance_stats = high_performance_keywords.groupby(['Search keyword', 'Ad type', 'Ad group', 'Campaign']).agg({
            'CTR': 'mean',  # Calculate the average CTR
            'Cost_per_Conversion': 'mean',  # Calculate the average cost per conversion
            'Conversion_Rate': 'mean'  # Calculate the average conversion rate
        }).reset_index()

        # Calculate statistical information for low-performance keywords
        low_performance_stats = low_performance_keywords.groupby(['Search keyword', 'Ad type', 'Ad group', 'Campaign']).agg({
            'CTR': 'mean',  # Calculate the average CTR
            'Cost_per_Conversion': 'mean',  # Calculate the average cost per conversion
            'Conversion_Rate': 'mean'  # Calculate the average conversion rate
        }).reset_index()

        st.write("High performance keywords -")
        st.write(high_performance_stats.head(10))

        st.write("Low performance keywords -")
        st.write(low_performance_stats.head(10))

        def suggest_optimization(row):
            # If the CTR and Conversion Rate are above thresholds, suggest increasing the bid.
            if row['CTR'] > ctr_mean and row['Conversion_Rate'] > cr_mean:
                return 'Increase bid'
            # If either the CTR or Conversion Rate is below thresholds, suggest pausing or removing the keyword.
            elif row['CTR'] < ctr_mean or row['Conversion_Rate'] < cr_mean:
                return 'Pause or remove keyword'
            # Otherwise, suggest maintaining the current bid.
            else:
                return 'Maintain current bid'

        # Apply the optimization suggestions
        non_zero_conversions['Optimization_Suggestion'] = non_zero_conversions.apply(suggest_optimization, axis=1)

        # Group by search keyword, Ad Type, Ad Group, and Campaign and calculate the most common optimization suggestion for each group
        grouped_optimization = non_zero_conversions.groupby(['Search keyword', 'Ad type', 'Ad group', 'Campaign'], as_index=False)[
            ['CTR', 'Conversion_Rate', 'Cost_per_Conversion', 'Optimization_Suggestion']
        ].agg(lambda x: x.mode()[0])  # Get the most common suggestion for each group

        # Output the optimization suggestions for the first 30 keywords
        st.markdown("First 30 keywords optimization suggestions -")
        st.write(grouped_optimization.head(30))

        st.write("Based on performance, we recommend the following actions -")
        st.write(grouped_optimization['Optimization_Suggestion'].value_counts())


        # Filter out keywords with a suggestion to increase bids
        new_dataset_high_performance = non_zero_conversions[non_zero_conversions['Optimization_Suggestion'] == 'Increase bid']

        # Group by keyword, Ad Type, Ad Group, and Campaign and calculate average performance metrics for each group
        grouped_high_performance = new_dataset_high_performance.groupby(['Search keyword', 'Ad type', 'Ad group', 'Campaign']).agg({
            'CTR': 'mean',                  # Calculate average CTR
            'Cost_per_Conversion': 'mean',  # Calculate average cost per conversion
            'Conversion_Rate': 'mean',      # Calculate average conversion rate
            'Optimization_Suggestion': 'first'  # Get the optimization suggestion (only keep the first value)
        }).reset_index()

        # Output the statistics for the top 30 high-performance keywords
        st.write(grouped_high_performance[['Search keyword', 'Ad type', 'Ad group', 'Campaign', 'CTR', 'Conversion_Rate', 'Cost_per_Conversion', 'Optimization_Suggestion']].head(15))

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='CTR', y='Conversion_Rate', hue='Optimization_Suggestion', data=non_zero_conversions)
        plt.title('CTR vs Conversion Rate')
        plt.xlabel('CTR')
        plt.ylabel('Conversion Rate')
        st.pyplot(plt)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Cost', y='Conversions', hue='Optimization_Suggestion', data=non_zero_conversions, palette={'Increase bid': 'orange', 'Pause or remove keyword': 'blue', 'Maintain current bid': 'green'})
        plt.title('Cost vs Conversions')
        plt.xlabel('Cost')
        plt.ylabel('Conversions')
        plt.legend(title='Optimization Suggestion')
        st.pyplot(plt)


        # Create fitness and individual classes
        try:
            del creator.FitnessMax
            del creator.Individual
        except AttributeError:
            pass 

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Group the dataset including 'Ad Type', 'Ad Group', and 'Campaign' along with aggregating Clicks and Conversions
        new_dataset_high_performance_grouped = (
            new_dataset_high_performance.groupby(['Search keyword', 'Ad type', 'Ad group', 'Campaign'], as_index=False)
            .agg({'Clicks': 'sum', 'Conversions': 'sum'})  
        )

        # Evaluation function
        def evaluate(individual):
            cpcs = np.array(individual)
            total_cost = np.sum(cpcs * new_dataset_high_performance_grouped['Clicks'])
            total_conversions = np.sum(new_dataset_high_performance_grouped['Conversions'])
            total_conv_value = total_conversions * 100  # Assuming each conversion is worth 100
            
            # Ensure the total conversion value is not less than the total cost
            if total_conv_value < total_cost or total_cost == 0:
                roi = -1  # Set as -1 to denote it's not feasible
            else:
                roi = (total_conv_value - total_cost) / total_cost
            
            return (roi,)

        # Initialize the toolbox
        toolbox = base.Toolbox()
        toolbox.register("CPC", random.uniform, 0.1, 5.0)  # CPC range
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.CPC, n=len(new_dataset_high_performance_grouped))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)  # Gaussian mutation
        toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection

        # Run the genetic algorithm
        def run_genetic_algorithm():
            population = toolbox.population(n=50)
            for gen in range(50):
                fitnesses = list(map(toolbox.evaluate, population))
                for ind, fit in zip(population, fitnesses):
                    ind.fitness.values = fit
                
                offspring = toolbox.select(population, len(population))
                offspring = list(map(toolbox.clone, offspring))
                
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.5:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                for mutant in offspring:
                    if random.random() < 0.2:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                population[:] = offspring

            best_index = np.argmax([ind.fitness.values[0] for ind in population if ind.fitness.valid])
            return population[best_index]

        # Run the genetic algorithm and get the best individual
        best_individual = run_genetic_algorithm()

        # Assumed budget
        budget = 11000000  
        optimized_strategy = []

        # Generate optimized advertising strategies
        for index, row in new_dataset_high_performance_grouped.iterrows():
            cost = best_individual[index] * row['Clicks']
            # Ensure the generated cost is positive and within budget
            if cost > 0 and cost <= budget:  
                optimized_strategy.append({
                    'Keyword': row['Search keyword'],
                    'Ad Type': row['Ad type'],
                    'Ad Group': row['Ad group'],
                    'Campaign': row['Campaign'],
                    'Cost': cost
                })

        # Create DataFrame for the optimized results
        optimized_df = pd.DataFrame(optimized_strategy)

        st.markdown("#### Optimized Ad Strategy:")
        st.write(optimized_df.head(15))

        st.markdown("#### The total budget for this optimized strategy is: $ {:.2f}".format(optimized_df['Cost'].sum()))


















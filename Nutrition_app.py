import streamlit as st
from datetime import datetime as dt
from streamlit_option_menu import option_menu
import mysql.connector as db
import pandas as pd
import re 
import requests
import mysql.connector as db
import pymysql
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://cdn.wallpapersafari.com/88/75/cLUQqJ.jpg");
  background-size: cover;
}
</style>
"""

#st.markdown(page_element, unsafe_allow_html=True)

# css for the button
button_style = """
    <style>
    .stButton > button {
        width: 200px;
        height: 50px;
        font-size: 20px;
    }
    </style>
    """

# Apply the custom CSS
st.markdown(button_style, unsafe_allow_html=True)



st.markdown("""
<style>
.big-font {
    font-size:75px !important;
    color: darkgreen;
    font-style: bold;
    font-family: 'Courier New', Courier, monospace;
    text-align: center;
    font-weight: bold;  
    background-color: white;
</style>

""", unsafe_allow_html=True)


##################################################################################
#   Database connectivity
###########################################################################################

def queries_fn():

    connection=db.connect(
        host="localhost",
        user="Nutri",
        password="12345678",
        database="nutrition",
        auth_plugin='mysql_native_password',
        port=3306
    )


    curr=connection.cursor()



    # to fetch the data from obesity table
    curr.execute("SELECT * FROM obesity")
    data=curr.fetchall()

    df_obesity=pd.DataFrame(data,columns=[i[0] for i in curr.description])
    #print(len(df_obesity))

    # to fetch the data from malnutrition table

    curr.execute("SELECT * FROM malnutrition")
    data1=curr.fetchall()

    df_malnutrition=pd.DataFrame(data1,columns=[i[0] for i in curr.description])
    #print(len(df_malnutrition))
    curr.close()
    connection.close()  


    ###############################################################################

    Questions = {1:"1.Top 5 regions with the highest average obesity levels in the most recent year(2022)",
                    2:"2.Top 5 countries with highest obesity estimates",
                    3:"3.Obesity trend in India over the years(Mean_estimate)",
                    4:"4.Average obesity by gender",
                    5:"5.Country count by obesity level category and age group",
                    6:"6.Top 5 countries least reliable countries(with highest CI_Width) and Top 5 most consistent countries (smallest average CI_Width)",
                    7:"7.Average obesity by age group",
                    8:"8.Top 10 Countries with consistent low obesity (low average + low CI)over the years",
                    9:"9.Countries where female obesity exceeds male by large margin (same year)",
                    10:"10.Global average obesity percentage per year",
                    11:"11.Avg. malnutrition by age group",
                    12:"12.Top 5 countries with highest malnutrition(mean_estimate)",
                    13:"13.Malnutrition trend in African region over the years",
                    14:"14.Gender-based average malnutrition",
                    15:"15.Malnutrition level-wise (average CI_Width by age group)",
                    16:"16.Yearly malnutrition change in specific countries(India, Nigeria, Brazil)",
                    17:"17.Regions with lowest malnutrition averages",
                    18:"18.Countries with increasing malnutrition",
                    19:"19.Min/Max malnutrition levels year-wise comparison",
                    20:"20.High CI_Width flags for monitoring(CI_width > 5)",
                    21:"21.Obesity vs malnutrition comparison by country(any 5 countries)",
                    22:"22.Gender-based disparity in both obesity and malnutrition",
                    23:"23.Region-wise avg estimates side-by-side(Africa and America)",
                    24:"24.Countries with obesity up & malnutrition down",
                    25:"25.Age-wise trend analysis"
                    }


    #  Selecting the queries

    option = st.selectbox(
        "",
        ("1.Top 5 regions with the highest average obesity levels in the most recent year(2022)",
        "2.Top 5 countries with highest obesity estimates",
        "3.Obesity trend in India over the years(Mean_estimate)",
        "4.Average obesity by gender",
        "5.Country count by obesity level category and age group",
        "6.Top 5 countries least reliable countries(with highest CI_Width) and Top 5 most consistent countries (smallest average CI_Width)",
        "7.Average obesity by age group",
        "8.Top 10 Countries with consistent low obesity (low average + low CI)over the years",
        "9.Countries where female obesity exceeds male by large margin (same year)",
        "10.Global average obesity percentage per year",
        "11.Avg. malnutrition by age group",
        "12.Top 5 countries with highest malnutrition(mean_estimate)",
        "13.Malnutrition trend in African region over the years",
        "14.Gender-based average malnutrition",
        "15.Malnutrition level-wise (average CI_Width by age group)",
        "16.Yearly malnutrition change in specific countries(India, Nigeria, Brazil)",
        "17.Regions with lowest malnutrition averages",
        "18.Countries with increasing malnutrition",
        "19.Min/Max malnutrition levels year-wise comparison",
        "20.High CI_Width flags for monitoring(CI_width > 5)",
        "21.Obesity vs malnutrition comparison by country(any 5 countries)",
        "22.Gender-based disparity in both obesity and malnutrition",   
        "23.Region-wise avg estimates side-by-side(Africa and America)",
        "24.Countries with obesity up & malnutrition down",
        "25.Age-wise trend analysis"
        ),
        index=None,
        placeholder="Select your query...",
    )

    st.write("You selected:", option)

    for i in Questions.keys():
        if option == Questions[i]:
            if i==1:
                #1. Top 5 regions with the highest average obesity levels in the most recent year(2022)

                ob_Avg = df_obesity.groupby('Region')['Mean_Estimate'].mean().reset_index()
                ob_Avg = ob_Avg.sort_values(by='Mean_Estimate', ascending=False).head(5)    
                ob_Avg['Year'] = 2022  # Adding the year column
                ob_Avg.rename(columns={'Mean_Estimate': 'Average_Obesity_Level'}, inplace=True)
                ob_Avg = ob_Avg.reset_index(drop=True)
                st.write("Top 5 regions with the highest average obesity levels in 2022:")
                st.dataframe(ob_Avg)
                
            elif i==2: 
                #2. Top 5 countries with highest obesity estimates

                ob_top_countries = df_obesity.sort_values(by='Mean_Estimate', ascending=False).head(5)
                ob_top_countries = ob_top_countries[['Country', 'Mean_Estimate', 'Year']].reset_index(drop=True)
                ob_top_countries.rename(columns={'Mean_Estimate': 'Obesity_Estimate'}, inplace=True)
                st.write("Top 5 countries with the highest obesity estimates:")
                st.dataframe(ob_top_countries)
            
            elif i==3:
                #3. Obesity trend in India over the years(Mean_estimate)
                df_obesity['Year'] = pd.to_datetime(df_obesity['Year'], format='%Y').dt.year
                df_obesity['Mean_Estimate'] = pd.to_numeric(df_obesity['Mean_Estimate'], errors='coerce')
                df_obesity = df_obesity.dropna(subset=['Mean_Estimate'])
                df_obesity = df_obesity[df_obesity['Year'] >= 2000]  # Filter for years >= 2000
                df_obesity = df_obesity[df_obesity['Country'] == 'India']  # Filter for India
                df_obesity = df_obesity[['Year', 'Mean_Estimate']].reset_index(drop=True)
                df_obesity.rename(columns={'Mean_Estimate': 'Obesity_Estimate'}, inplace=True)
                st.write("Obesity trend in India over the years:")
                st.dataframe(df_obesity.drop_duplicates('Year'))
                
            elif i==4:  
                # 4. Average obesity by gender 

                #ob_avg_gender = df_obesity.groupby(['Gender', 'Year'])['Mean_Estimate'].mean().reset_index()
                ob_avg_gender = df_obesity.groupby(['Gender'])['Mean_Estimate'].mean().reset_index()
                ob_avg_gender.rename(columns={'Mean_Estimate': 'Average_Obesity_Level'}, inplace=True)
                ob_avg_gender = ob_avg_gender.reset_index(drop=True)          
                st.write("Average obesity by gender:")
                st.dataframe(ob_avg_gender)
            
            elif i==5:      
                #5. Country count by obesity level category and age group

                ob_country_count = df_obesity.groupby(['Obesity_Level', 'Age_Group']).size().reset_index(name='Country_Count')

                ob_country_count = ob_country_count.reset_index(drop=True).drop_duplicates()      
                st.write("Country count by obesity level category and age group:")
                st.dataframe(ob_country_count)
                
            elif i==6:
                #6. Top 5 countries least reliable countries(with highest CI_Width) and Top 5 most consistent countries (smallest average CI_Width)

                ob_least_reliable = df_obesity.sort_values(by='CI_Width', ascending=False).head(5)
                ob_least_reliable = ob_least_reliable[['Country', 'CI_Width', 'Year']].reset_index(drop=True)               
                ob_least_reliable.rename(columns={'CI_Width': 'Highest_CI_Width'}, inplace=True)
                ob_most_consistent = df_obesity.groupby('Country')['CI_Width'].mean().reset_index() 
                ob_most_consistent = ob_most_consistent.sort_values(by='CI_Width').head(5).reset_index(drop=True)
                ob_most_consistent.rename(columns={'CI_Width': 'Lowest_Average_CI_Width'}, inplace=True)  
                
                st.write("Top 5 least reliable countries (highest CI_Width):")
                st.dataframe(ob_least_reliable)
                st.write("Top 5 most consistent countries (smallest average CI_Width):")
                st.dataframe(ob_most_consistent)
                
            elif i==7:
                #7. Average obesity by age group

                ob_avg_age_group = df_obesity.groupby('Age_Group')['Mean_Estimate'].mean().reset_index()
                ob_avg_age_group.rename(columns={'Mean_Estimate': 'Average_Obesity_Level'}, inplace=True)           
                ob_avg_age_group = ob_avg_age_group.reset_index(drop=True)
                st.write("Average obesity by age group:")
                st.dataframe(ob_avg_age_group)
            
            elif i==8:
                #8. Top 10 Countries with consistent low obesity (low average + low CI)over the years

                ob_consistent_low = df_obesity.groupby('Country').agg({'Mean_Estimate': 'mean', 'CI_Width': 'mean'}).reset_index()
                ob_consistent_low = ob_consistent_low.sort_values(by=['Mean_Estimate', 'CI_Width']).head(10)    
                ob_consistent_low.rename(columns={'Mean_Estimate': 'Average_Obesity_Level', 'CI_Width': 'Average_CI_Width'}, inplace=True)
                ob_consistent_low = ob_consistent_low.reset_index(drop=True)
                st.write("Top 10 countries with consistent low obesity (low average + low CI):")
                st.dataframe(ob_consistent_low)
                
            elif i==9:
                #9. Countries where female obesity exceeds male by large margin (same year)
                df_gender = df_obesity.pivot_table(
                    index=['Country', 'Year'],
                    columns='Gender',
                    values='Mean_Estimate'
                ).reset_index()
                # Only keep rows where both Male and Female data are present
                df_gender = df_gender.dropna(subset=['Female', 'Male'])
                df_gender['Difference'] = df_gender['Female'] - df_gender['Male']
                # Define a "large margin" (e.g., difference > 5)
                large_margin = 5
                df_large_margin = df_gender[df_gender['Difference'] > large_margin]
                df_large_margin = df_large_margin[['Country', 'Year', 'Female', 'Male', 'Difference']]
                df_large_margin = df_large_margin.sort_values(by='Difference', ascending=False).reset_index(drop=True)
                st.write("Countries where female obesity exceeds male by a large margin (same year):")
                st.dataframe(df_large_margin)

                
            elif i==10:
                #10. Global average obesity percentage per year 

                ob_global_avg = df_obesity.groupby('Year')['Mean_Estimate'].mean().reset_index()
                ob_global_avg.rename(columns={'Mean_Estimate': 'Global_Average_Obesity_Percentage'}, inplace=True)              
                ob_global_avg = ob_global_avg.reset_index(drop=True)
                st.write("Global average obesity percentage per year:")
                st.dataframe(ob_global_avg)
            
            elif i==11:
                #11. Avg. malnutrition by age group

                mal_avg_age = df_malnutrition.groupby('Age_Group')['Mean_Estimate'].mean().reset_index()
                mal_avg_age.rename(columns={'Mean_Estimate': 'Average_Malnutrition_Level'}, inplace=True)          
                mal_avg_age = mal_avg_age.reset_index(drop=True)
                st.write("Average malnutrition by age group:")
                st.dataframe(mal_avg_age)
    
            elif i==12:
                #12. Top 5 countries with highest malnutrition(mean_estimate)

                mal_top_countries = df_malnutrition.sort_values(by='Mean_Estimate', ascending=False).head(5)
                mal_top_countries = mal_top_countries[['Country', 'Mean_Estimate', 'Year']].reset_index(drop=True)      
                mal_top_countries.rename(columns={'Mean_Estimate': 'Malnutrition_Estimate'}, inplace=True)
                st.write("Top 5 countries with the highest malnutrition estimates:")
                st.dataframe(mal_top_countries)
            
            elif i==13:
                #13. Malnutrition trend in African region over the years

                mal_africa_trend = df_malnutrition[df_malnutrition['Country'] == 'Africa'][['Year', 'Mean_Estimate']]
                mal_africa_trend = mal_africa_trend.reset_index(drop=True)
                mal_africa_trend.rename(columns={'Mean_Estimate': 'Malnutrition_Estimate'}, inplace=True)
                st.write("Malnutrition trend in Africa over the years:")
                st.dataframe(mal_africa_trend)
            
            elif i==14:
                #14. Gender-based average malnutrition

                mal_avg_gender = df_obesity.groupby(['Gender'])['Mean_Estimate'].mean().reset_index()
                mal_avg_gender.rename(columns={'Mean_Estimate': 'Average_Obesity_Level'}, inplace=True)
                mal_avg_gender = mal_avg_gender.reset_index(drop=True)          
                st.write("Average obesity by gender:")
                st.dataframe(mal_avg_gender)
                
            elif i==15:
                #15. Malnutrition level-wise (average CI_Width by age group)

                mal_avg_ci_age = df_malnutrition.groupby(['Age_Group'])['CI_Width'].mean().reset_index()
                mal_avg_ci_age.rename(columns={'CI_Width': 'Average_Malnutrition_Level'}, inplace=True)
                mal_avg_ci_age = mal_avg_ci_age.reset_index(drop=True)          
                st.write("Average malnutrition level by age group and CI_Width:")
                st.dataframe(mal_avg_ci_age)
                
            elif i==16:
                #16. Yearly malnutrition change in specific countries(India, Nigeria, Brazil)
                mal_yearly_trend = df_malnutrition[df_malnutrition['Country'].isin(['India', 'Nigeria', 'Brazil'])]
                mal_yearly_trend = mal_yearly_trend.groupby(['Country', 'Year'])['Mean_Estimate'].mean().reset_index()
                mal_yearly_trend.rename(columns={'Mean_Estimate': 'Malnutrition_Estimate'}, inplace=True)
                mal_yearly_trend = mal_yearly_trend.sort_values(['Country', 'Year']).reset_index(drop=True)
                st.write("Yearly malnutrition change in specific countries (India, Nigeria, Brazil):")
                st.dataframe(mal_yearly_trend)
                
            elif i==17:
                # 17. Regions with lowest malnutrition averages
                
                mal_low_regions = df_malnutrition.groupby('Region')['Mean_Estimate'].mean().reset_index()
                mal_low_regions = mal_low_regions.sort_values(by='Mean_Estimate').head(5)       
                mal_low_regions.rename(columns={'Mean_Estimate': 'Average_Malnutrition_Level'}, inplace=True)
                mal_low_regions = mal_low_regions.reset_index(drop=True)
                st.write("Regions with lowest malnutrition averages:")
                st.dataframe(mal_low_regions)
                
            elif i==18:
                #18. Countries with increasing malnutrition 

                mal_increasing_countries = df_malnutrition.groupby('Country').agg({'Mean_Estimate': ['min', 'max']}).reset_index()
                mal_increasing_countries.columns = ['Country', 'Min_Malnutrition', 'Max_Malnutrition']
                mal_increasing_countries['Difference'] = mal_increasing_countries['Max_Malnutrition'] - mal_increasing_countries['Min_Malnutrition']
                mal_increasing_countries = mal_increasing_countries[mal_increasing_countries['Difference'] > 0]     
                mal_increasing_countries = mal_increasing_countries.reset_index(drop=True)
                st.write("Countries with increasing malnutrition levels:")
                st.dataframe(mal_increasing_countries)
                
                
            elif i==19:
                #19. Min/Max malnutrition levels year-wise comparison
                mal_min_max_yearly = df_malnutrition.groupby('Year')['Mean_Estimate'].agg(['min', 'max']).reset_index()
                mal_min_max_yearly.rename(columns={'min': 'Min_Malnutrition', 'max': 'Max_Malnutrition'}, inplace=True) 
                mal_min_max_yearly = mal_min_max_yearly.reset_index(drop=True)
                st.write("Min/Max malnutrition levels year-wise comparison:")
                st.dataframe(mal_min_max_yearly)
            
            
            elif i==20:
                #20. High CI_Width flags for monitoring(CI_width > 5)

                mal_high_ci_flags = df_malnutrition[df_malnutrition['CI_Width'] > 5][['Country', 'Year', 'CI_Width']]
                mal_high_ci_flags = mal_high_ci_flags.reset_index(drop=True)
                mal_high_ci_flags.rename(columns={'CI_Width': 'High_CI_Width_Flag'}, inplace=True)      
                st.write("High CI_Width flags for monitoring (CI_Width > 5):")
                st.dataframe(mal_high_ci_flags)
                
            elif i==21:
                # 21. Obesity vs malnutrition comparison by country(any 5 countries)
                
                common_countries = list(set(df_obesity['Country']).intersection(set(df_malnutrition['Country'])))
                selected_countries = common_countries[:5]  # Pick first 5 for demonstration

                ob_data = df_obesity[df_obesity['Country'].isin(selected_countries)]
                mal_data = df_malnutrition[df_malnutrition['Country'].isin(selected_countries)]

                # Group by country and year, take mean if multiple rows
                ob_grouped = ob_data.groupby(['Country', 'Year'])['Mean_Estimate'].mean().reset_index()
                mal_grouped = mal_data.groupby(['Country', 'Year'])['Mean_Estimate'].mean().reset_index()

                comparison = pd.merge(
                    ob_grouped,
                    mal_grouped,
                    on=['Country', 'Year'],
                    suffixes=('_Obesity', '_Malnutrition')
                )

                comparison.rename(columns={
                    'Mean_Estimate_Obesity': 'Obesity_Estimate',
                    'Mean_Estimate_Malnutrition': 'Malnutrition_Estimate'
                }, inplace=True)

                st.write("Obesity vs malnutrition comparison by country (any 5 countries):")
                st.dataframe(comparison)
                
            elif i==22: 
                # 22. Gender-based disparity in both obesity and malnutrition

                # Calculate average obesity by gender
                ob_gender = df_obesity.groupby('Gender')['Mean_Estimate'].mean().reset_index()
                ob_gender.rename(columns={'Mean_Estimate': 'Average_Obesity_Level'}, inplace=True)

                # Calculate average malnutrition by gender
                mal_gender = df_malnutrition.groupby('Gender')['Mean_Estimate'].mean().reset_index()
                mal_gender.rename(columns={'Mean_Estimate': 'Average_Malnutrition_Level'}, inplace=True)

                # Merge the two on Gender
                gender_disparity = pd.merge(ob_gender, mal_gender, on='Gender', how='inner')
                st.write("Gender-based disparity in both obesity and malnutrition:")
                st.dataframe(gender_disparity)
                
            elif i==23:
                # 23. Region-wise avg estimates side-by-side(Africa and America)

                ob_mal_region_comparison = pd.merge(
                    df_obesity.groupby('Region')['Mean_Estimate'].mean().reset_index(),
                    df_malnutrition.groupby('Region')['Mean_Estimate'].mean().reset_index(),
                    on='Region',
                    suffixes=('_Obesity', '_Malnutrition')
                )   
                ob_mal_region_comparison.rename(columns={
                    'Mean_Estimate_Obesity': 'Average_Obesity_Level',
                    'Mean_Estimate_Malnutrition': 'Average_Malnutrition_Level'
                }, inplace=True)
                st.write("Region-wise average estimates side-by-side (Africa and America):")
                st.dataframe(ob_mal_region_comparison)
                
            elif i==24: 
                # 24.  Countries with obesity up & malnutrition down   

                ob_mal_up_down = pd.merge(
                    df_obesity.groupby('Country')['Mean_Estimate'].mean().reset_index(),
                    df_malnutrition.groupby('Country')['Mean_Estimate'].mean().reset_index(),
                    on='Country',
                    suffixes=('_Obesity', '_Malnutrition')
                )
                ob_mal_up_down = ob_mal_up_down[
                    (ob_mal_up_down['Mean_Estimate_Obesity'] > ob_mal_up_down['Mean_Estimate_Obesity'].mean()) &
                    (ob_mal_up_down['Mean_Estimate_Malnutrition'] < ob_mal_up_down['Mean_Estimate_Malnutrition'].mean())
                ].reset_index(drop=True)    
                ob_mal_up_down.rename(columns={
                    'Mean_Estimate_Obesity': 'Average_Obesity_Level',
                    'Mean_Estimate_Malnutrition': 'Average_Malnutrition_Level'
                }, inplace=True)
                st.write("Countries with obesity up and malnutrition down:")
                st.dataframe(ob_mal_up_down)
            
            elif i==25: 
                # 25. Age-wise trend analysis

                ob_mal_age_trend = pd.merge(
                    df_obesity.groupby('Age_Group')['Mean_Estimate'].mean().reset_index(),
                    df_malnutrition.groupby('Age_Group')['Mean_Estimate'].mean().reset_index(),
                    on='Age_Group',
                    suffixes=('_Obesity', '_Malnutrition')
                )                                                                   
                ob_mal_age_trend.rename(columns={
                    'Mean_Estimate_Obesity': 'Average_Obesity_Level',
                    'Mean_Estimate_Malnutrition': 'Average_Malnutrition_Level'
                }, inplace=True)
                st.write("Age-wise trend analysis:")
                st.dataframe(ob_mal_age_trend)

def eda_fn():                            
    connection=db.connect(
        host="localhost",
        user="Nutri",
        password="12345678",
        database="nutrition",
        auth_plugin='mysql_native_password',
        port=3306
    )


    curr=connection.cursor()



    # to fetch the data from obesity table
    curr.execute("SELECT * FROM obesity")
    data=curr.fetchall()

    df_obesity=pd.DataFrame(data,columns=[i[0] for i in curr.description])
    #print(len(df_obesity))

    # to fetch the data from malnutrition table

    curr.execute("SELECT * FROM malnutrition")
    data1=curr.fetchall()

    df_malnutrition=pd.DataFrame(data1,columns=[i[0] for i in curr.description])
    #print(len(df_malnutrition))
    curr.close()
    connection.close()  


    ###############################################################################

    Questions = {1:"1.Obesity mean estimate variation for 100 samples",
                    2:"2.Malnutrition mean estimate variation for 100 samples",
                    3:"3.Displaying the mean Obesity confidence interval range",
                    4:"4.Displaying the mean malnutrition confidence interval range",
                    5:"5.Obesity vs region-wise count",
                    6:"6.Malnutrition vs region-wise count",
                    7:"7.Region wise obesity count split across age group",
                    8:"8.Region wise malnutrition count split across age group",
                    9:"9.Country count per region - Stacked by levels of obesity",
                    10:"10.Country count per region - Stacked by levels of malnutrition",
                    11:"11.yearly trend of mean estimate of obesity by levels",
                    12:"12.yearly trend of mean estimate of malnutrition by levels",
                    13:"13.Obesity vs malnutrition comparison by confidence interval width",
                    14:"14.Box Plot of obesity Mean Estimate by Region",
                    15:"15.Box Plot of malnutrition Mean Estimate by Region",
                    16:"16.3D Scatter plot of obesity Mean Estimate by Age Group and Gender",
                    17:"17.3D Scatter plot of malnutrition Mean Estimate by Age Group and Gender",
                    18:"18.Scatter plot of obesity Mean Estimate by CI_Width and Region",
                    19:"19.Scatter plot of malnutrition Mean Estimate by CI_Width and Region"
                    
                    }


    #  Selecting the queries

    # Use session state to reset selectbox after rerun
    if "eda_option" not in st.session_state:
        st.session_state.eda_option = None

    option = st.selectbox(
        "",
        (
            "1.Obesity mean estimate variation for 100 samples",
            "2.Malnutrition mean estimate variation for 100 samples",
            "3.Displaying the mean Obesity confidence interval range",
            "4.Displaying the mean malnutrition confidence interval range",
            "5.Obesity vs region-wise count",
            "6.Malnutrition vs region-wise count",
            "7.Region wise obesity count split across age group",
            "8.Region wise malnutrition count split across age group",
            "9.Country count per region - Stacked by levels of obesity",
            "10.Country count per region - Stacked by levels of malnutrition",
            "11.yearly trend of mean estimate of obesity by levels",
            "12.yearly trend of mean estimate of malnutrition by levels",
            "13.Obesity vs malnutrition comparison by confidence interval width",
            "14.Box Plot of obesity Mean Estimate by Region",
            "15.Box Plot of malnutrition Mean Estimate by Region",
            "16.3D Scatter plot of obesity Mean Estimate by Age Group and Gender",
            "17.3D Scatter plot of malnutrition Mean Estimate by Age Group and Gender",
            "18.Scatter plot of obesity Mean Estimate by CI_Width and Region",
            "19.Scatter plot of malnutrition Mean Estimate by CI_Width and Region"
        ),
        index=None if st.session_state.eda_option is None else (
            list(Questions.values()).index(st.session_state.eda_option)
            if st.session_state.eda_option in list(Questions.values()) else None
        ),
        placeholder="Select your query...",
        key="eda_option"
    )

    # Reset to default (None) after rerun
    if st.session_state.eda_option is not None and st.session_state.eda_option != option:
        st.session_state.eda_option = None

    st.write("You selected:", option)

    for i in Questions.keys():
        if option == Questions[i]:
            if i==1:
                #1.Obesity mean estimate variation for 100 samples

                plt.figure(figsize=(15, 4))
                data2 = df_obesity.tail(100)
                sns.lineplot(data=data2, x=data2.index, y="Mean_Estimate", marker='o', color='red')
                plt.xlabel("Sample Index")
                plt.ylabel("Obesity Mean Estimate")
                plt.title("Obesity Mean Estimate Variation for Last 100 Samples")
                st.pyplot(plt.gcf())
                plt.clf()
                
            elif i==2: 
               #2.Malnutrition mean estimate variation for 100 samples
                plt.figure(figsize =(15,4))


                data2=df_malnutrition.tail(100)
                sns.lineplot(data=data2, x= data2.index, y="Mean_Estimate", marker = 'o', color='red')
                plt.xlabel("Sample Index")
                plt.ylabel("Malnutritiony Mean Estimate")
                plt.title("Malnutrition Mean Estimate Variation for Last 100 Samples")
                st.pyplot(plt.gcf())
                plt.clf()
            
            elif i==3:
               #3.Displaying the mean Obesity confidence interval range
                plt.figure(figsize=(15,6))

                data1=df_obesity

                sns.lineplot(data=data1, x=data1.index, y= "CI_Width", marker='o', color='green', label="CI_Width")

                mean_y=data1['CI_Width'].mean()
                plt.axhline(y=mean_y, color= 'red', linestyle='--', label= f'Mean = {mean_y:0.2f} ')

                plt.title('Displaying the mean Obesity confidence interval range')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                st.pyplot(plt.gcf())
                plt.clf()
                
            elif i==4:  
                #4.Displaying the mean malnutrition confidence interval range
                plt.figure(figsize=(15,6))

                data2=df_malnutrition

                sns.lineplot(data=data2, x=data2.index, y= "CI_Width", marker='o', color='green', label="CI_Width")

                mean_y=data2['CI_Width'].mean()
                plt.axhline(y=mean_y, color= 'red', linestyle='--', label= f'Mean = {mean_y:0.2f} ')

                plt.title('Displaying the mean malnutrition confidence interval range')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                st.pyplot(plt.gcf())
                plt.clf()
            
            elif i==5:      
                #5.Obesity vs region-wise count
               
                data1=df_obesity
                plt.figure(figsize=(15,6))
                sns.histplot(data=data1, x="Region", label ="Region", color="green")
                plt.title("Obesity vs region-wise count")

                plt.xlabel("Region")
                plt.ylabel("Count")
                plt.legend()
                st.pyplot(plt.gcf())
                plt.clf()
                
                
            elif i==6:
                #6.Malnutrition vs region-wise count
                data2=df_malnutrition
                plt.figure(figsize=(15,6))
                sns.histplot(data=data2, x="Region", label ="Region", color="green")

                plt.legend()
                plt.xlabel("Region")
                plt.ylabel("Count")
                plt.title("Malnutrition vs region-wise count" )
                st.pyplot(plt.gcf())
                plt.clf()
                
            elif i==7:
                #7.Region wise obesity count split across age group
                st_bar = df_obesity.groupby(['Region', 'Age_Group']).size().unstack(fill_value=0)
                ax= st_bar.plot(kind="bar", stacked= True, figsize=(17,6), color = ['#40E0D0', '#9FE2BF'])

                ax.set_title("Region wise obesity count split across age group")
                ax.set_xlabel('Region')
                ax.set_ylabel('Number of Obese ppl')
                #ax.set_xticklabels(('Male', 'Female'))

                for rect in ax.patches:
                    y_value = rect.get_height()
                    x_value = rect.get_x() + rect.get_width() / 2
                    #space = 1
                    label = "{:.0f}".format(y_value)
                    ax.annotate(label, (x_value, y_value), xytext=(0, -13), textcoords="offset points", ha='center', va='bottom')

                st.pyplot(plt.gcf())
                plt.clf()
                
            elif i==8:
                #8.Region wise malnutrition count split across age group
                st_bar1 = df_malnutrition.groupby(['Region', 'Age_Group']).size().unstack(fill_value=0)
                ax= st_bar1.plot(kind="bar", stacked= True, figsize=(17,6), color = ['#40E0D0', '#9FE2BF'])

                ax.set_title("Region wise malnutrition count split across age group")
                ax.set_xlabel('Region')
                ax.set_ylabel('Number of Obese ppl')
                #ax.set_xticklabels(('Male', 'Female'))

                for rect in ax.patches:
                    y_value = rect.get_height()
                    x_value = rect.get_x() + rect.get_width() / 2
                    #space = 1
                    label = "{:.0f}".format(y_value)
                    ax.annotate(label, (x_value, y_value), xytext=(0, -13), textcoords="offset points", ha='center', va='bottom')

                st.pyplot(plt.gcf())
                plt.clf()
                
            elif i==9:
                #9.Country count per region - Stacked by levels of obesity

                country_level_counts = df_obesity.groupby(['Region', 'Obesity_Level'])['Country'].nunique().reset_index()
                country_level_counts = country_level_counts.rename(columns={'Country': 'Country_Count'})
                pivot_df = country_level_counts.pivot(index="Region", columns="Obesity_Level", values="Country_Count").fillna(0)

                ax= pivot_df.plot(kind="bar", stacked= True, figsize=(15,7))

                for idx, region in enumerate(pivot_df.index):
                    y_offset = 0
                    for level in pivot_df.columns:
                        value= pivot_df.loc[region, level]
                        if value >0:
                            ax.text(idx, y_offset + value /2, int(value), ha="center", va="center", fontsize = 10)
                            y_offset += value
                            
                plt.title(" Country count per region - Stacked by levels of obesity")
                plt.ylabel("Unique Country count")
                plt.xlabel("Region")
                plt.xticks(rotation=50)
                plt.legend(title="Levels")
                plt.tight_layout()
                
                st.pyplot(plt.gcf())
                plt.clf()
                
            elif i==10:
                #10.Country count per region - Stacked by levels of malnutrition
                country_level_counts1 = df_malnutrition.groupby(['Region', 'Malnutrition_Level'])['Country'].nunique().reset_index()
                country_level_counts1 = country_level_counts1.rename(columns={'Country': 'Country_Count'})
                pivot_df = country_level_counts1.pivot(index="Region", columns="Malnutrition_Level", values="Country_Count").fillna(0)
                #pivot_df= country_level_counts1.pivot(index="Region", columns="Malnutrition_Level", values="Country").fillna(0)

                ax= pivot_df.plot(kind="bar", stacked= True, figsize=(15,7))

                for idx, region in enumerate(pivot_df.index):
                    y_offset = 0
                    for level in pivot_df.columns:
                        value= pivot_df.loc[region, level]
                        if value >0:
                            ax.text(idx, y_offset + value /2, int(value), ha="center", va="center", fontsize = 10)
                            y_offset += value
                            
                plt.title(" Country count per region - Stacked by levels of malnutrition")
                plt.ylabel("Unique Country count")
                plt.xlabel("Region")
                plt.xticks(rotation=50)
                plt.legend(title="Levels")
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.clf()
                
            elif i==11:
                 #11.yearly trend of mean estimate of obesity by levels
                data1=df_obesity
                g=sns.FacetGrid(data1, col="Obesity_Level", col_wrap=3, height= 4, sharey= False )
                g.map_dataframe(sns.lineplot, x="Year", y="Mean_Estimate", marker='o')

                g.set_titles("{col_name}")
                g.set_axis_labels("Year","Mean_Estimate")
                plt.title("yearly trend of mean estimate of obesity by levels")
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.clf()
                
            elif i==12:
                 #12.yearly trend of mean estimate of malnutrition by levels
                data2=df_malnutrition
                plt.figure(figsize=(15,6))
                g1=sns.FacetGrid(data2, col="Malnutrition_Level", col_wrap=3, height= 4, sharey= False )
                g1.map_dataframe(sns.lineplot, x="Year", y="Mean_Estimate", marker='o')

                g1.set_titles("{col_name}")
                g1.set_axis_labels("Year","Mean_Estimate")
                plt.title("yearly trend of mean estimate of malnutrition by levels")
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.clf()
                
            elif i==13:
                #13.Obesity vs malnutrition comparison by confidence interval width
                plt.figure(figsize= (12, 5) )
                data1=df_obesity
                data2=df_malnutrition
                sns.kdeplot(data1['CI_Width'], color="blue")
                sns.kdeplot(data2['CI_Width'], color="red")
                plt.title("Obesity vs malnutrition comparison by confidence interval width")
                plt.legend(labels=["Obesity", "Malnutrition"], loc=2, bbox_to_anchor=(1, 1))
                st.pyplot(plt.gcf())
                plt.clf()
                
            elif i==14:
                #14.Box Plot of obesity Mean Estimate by Region
                plt.figure(figsize=(15,6))
                data1=df_obesity
                sns.set_style("darkgrid")
                sns.boxplot(data=data1, x="Region", y="Mean_Estimate", palette="deep")
                plt.title("Box Plot of obesity Mean Estimate by Region")

                plt.xlabel("Region")
                plt.ylabel("Mean_Estimate")

                st.pyplot(plt.gcf())
                plt.clf()
                
            elif i==15:
                 #15.Box Plot of malnutrition Mean Estimate by Region
                data2=df_malnutrition
                plt.figure(figsize=(15,6))
                sns.boxplot(data=data2, x="Region", y="Mean_Estimate", palette="deep")
                plt.title("Box Plot of malnutrition Mean Estimate by Region")

                plt.xlabel("Region")
                plt.ylabel("Mean_Estimate")

                st.pyplot(plt.gcf())
                plt.clf()
                
            elif i==16:
                #16.3D Scatter plot of obesity Mean Estimate by Age Group and Gender for 100 samples
                data1 = df_obesity.head(100)
                fig = px.scatter_3d(
                    data_frame=data1,
                    x="Age_Group",
                    y="Gender",
                    z="Mean_Estimate",
                    color="Gender",
                    symbol="Age_Group",
                    hover_data=["Country", "Year", "Region"],
                    width=1200,
                    height=600,
                    title="3D Scatter plot of obesity Mean Estimate by Age Group and Gender for 100 samples"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif i==17:
                #17.3D Scatter plot of malnutrition Mean Estimate by Age Group and Gender
                data1 = df_malnutrition.head(100)
                fig = px.scatter_3d(
                    data_frame=data1,
                    x="Age_Group",
                    y="Gender",
                    z="Mean_Estimate",
                    color="Gender",
                    symbol="Age_Group",
                    hover_data=["Country", "Year", "Region"],
                    width=1200,
                    height=600,
                    title="3D Scatter plot of Malnutrition Mean Estimate by Age Group and Gender for 100 samples"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif i==18:
                #18.Scatter plot of obesity Mean Estimate by CI_Width and Region
                data1=df_obesity
                regions=df_obesity["Region"].unique()
                n=len(regions)
                cols=3
                rows=(n+ cols - 1)//cols
                fig, axes = plt.subplots(rows, cols, figsize= (15, 4 * rows), sharex=False, sharey=False)

                for i , region in enumerate(regions):
                    r= i//cols
                    c=i%cols
                    ax=axes[r,c] if rows >1 else axes[c]
                    sns.scatterplot(data= data1[data1["Region"]==region], x ="Mean_Estimate", y="CI_Width", ax=ax)
                    ax.set_title(f'{region}')
                    ax.set_xlabel("Mean_Estimate")
                    ax.set_ylabel("CI_Width")
                    
                # to hide unused axes

                for j in range(i+1, rows * cols):
                    fig.delaxes(axes[j//cols, j%cols] if rows > 1 else axes[j % cols])
                    
                plt.title("Scatter plot of obesity Mean Estimate by CI_Width and Region")
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.clf()
                
            elif i==19:
                #19.Scatter plot of malnutrition Mean Estimate by CI_Width and Region
                data2=df_malnutrition
                regions1=df_malnutrition["Region"].unique()
                n=len(regions1)
                cols=3
                rows=(n+ cols - 1)//cols
                fig, axes = plt.subplots(rows, cols, figsize= (15, 4 * rows), sharex=False, sharey=False)

                for i , region in enumerate(regions1):
                    r= i//cols
                    c=i%cols
                    ax=axes[r,c] if rows >1 else axes[c]
                    sns.scatterplot(data= data2[data2["Region"]==region], x ="Mean_Estimate", y="CI_Width", ax=ax)
                    ax.set_title(f'{region}')
                    ax.set_xlabel("Mean_Estimate")
                    ax.set_ylabel("CI_Width")
                    
                # to hide unused axes

                for j in range(i+1, rows * cols):
                    fig.delaxes(axes[j//cols, j%cols] if rows > 1 else axes[j % cols])
                    
                plt.title("Scatter plot of malnutrition Mean Estimate by CI_Width and Region")
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.clf()
                            

# Sidebar with multiple tabs
with st.sidebar:
    selected = option_menu(
        menu_title="Choose your option", 
        options=["Queries", "EDA"], 
        icons=["folder", "calendar"],  
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )



# Content for each tab


if selected == "Queries":
    st.title("Select a Query ")
    queries_fn()
    
elif selected == "EDA":
    st.title("EDA")
    eda_fn()
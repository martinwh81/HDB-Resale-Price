import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils_functions import find_postal, find_nearest, dist_from_location, map, map_flats_year, _max_width_
import streamlit.components.v1 as components
import datetime
import joblib


st.set_page_config(layout="wide")

st.title('Prediction of Singapore HDB Resale Prices using Machine Learning')

st.text(" ")
st.text(" ")
st.text(" ")



with st.sidebar.form('User Input HDB Features'):
    flat_address = st.text_input("Flat Address or Postal Code", '110 BISHAN ST 12') # flat address
    
    town = st.selectbox('Town', list(['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                                            'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI',
                                            'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                                            'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG',
                                            'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN',
                                            'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS','PUNGGOL']),
                                index=2)
    flat_model = st.selectbox('Flat Model', list(['Model A', 'Improved', 'Premium Apartment', 'Standard',
                                                  'New Generation', 'Maisonette', 'Apartment', 'Simplified',
                                                  'Model A2', 'DBSS', 'Terrace', 'Adjoined flat', 'Multi Generation',
                                                  '2-room', 'Executive Maisonette', 'Type S1S2']), index=0)
    flat_type = st.selectbox('Flat Type', list(['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']),
                                    index=0)
    floor_area = st.slider("Floor Area (sqm)", 34,280,104) # floor area
    # storey = st.selectbox('Storey', list(['01 TO 03','04 TO 06','07 TO 09','10 TO 12','13 TO 15',
    #                                             '16 TO 18','19 TO 21','22 TO 24','25 TO 27','28 TO 30',
    #                                             '31 TO 33','34 TO 36','37 TO 39','40 TO 42','43 TO 45',
    #                                             '46 TO 48','49 TO 51']), index=3)
    storey = st.slider('Storey', 1,51,7)
    
    lease_commence_date = st.selectbox('Lease Commencement Date', list(reversed(range(1966, 2022))), index=35)
    
    submitted1 = st.form_submit_button(label = 'Submit to Predict')


@st.cache(allow_output_mutation=True)
def load_model():
    model_rf = joblib.load('rf_compressed.pkl')
    return model_rf

model_rf = load_model()






st.subheader('Application Information')
with st.expander("Expand to see details"):
    st.markdown("""
                This is a web app to demonstrate machine learning capability to predict current price of HDB Resale Price. 
                The model is build based on historical HDB Resale Prices data ([Data.gov.sg](https://data.gov.sg/)) from 2012 to 2023. 
                In addition to the **standard HDB attributes** (Town, Flat Model, Flat Type, Floor Area, Storey, Lease Commencement), important features
                related to **distance to amenities** (primary schools, supermarkets, and MRT/LRT) and **Consumer Price Index** were also used to improve the accuracy of
                the model.

                The following graph shows the most important features at predicting HDB resale price:            
                """)                

    # Plotting the Feature Importance
    feature = model_rf.feature_names_in_
    feature_mapping = {'flat_type': 'Flat Type', 'storey':'Storey', 'floor_area_sqm':'Floor Area (sqm)', 
                    'lease_commence_date':'Lease Commence Year', 'remaining_lease':'Remaining Lease (yrs)', 
                    'school_dist':'Nearest distance to Primary School', 'num_school_2km':'Num of Schools within 2km', 
                    'mrt_dist':'Nearest distance to MRT/LRT', 'num_mrt_2km':'Number of MRT within 2km',
                    'supermarket_dist':'Nearest distance to supermarket', 'num_supermarket_2km':'Number of supermarkets within 2km',
                    'dist_dhoby':'Distance to Central of Singapore', 'cpi':'Consumer Price Index', 
                    'region_East':'Region=East', 'region_North':'Region=North',
                    'region_North East':'Region=North East', 'region_West':'Region=West',
                    'model_Apartment':'Model=Apartment','model_Maisonette':'Model=Mansionette', 
                    'model_Model A':'Model=A', 'model_New Generation':'Model=New Generation',
                    'model_Special':'Model=Special'}

    feature_names = [feature_mapping[key] for key in feature]


    # Plot Feature Importance:
    fig = plt.figure(figsize=(10,10))
    feat_imp = pd.DataFrame({'Features': feature_names, 'Feature Importance': model_rf.feature_importances_}).sort_values('Feature Importance', ascending=False)
    sns.barplot(y='Features', x='Feature Importance', data=feat_imp)
    plt.title('Feature Importance', size=15)
    st.pyplot(fig)

st.subheader('How to Use')
st.markdown("""
            - Expand the *left sidebar* (top left corner),
            - Enter a HDB address / postal code and select the rest of HDB attributes
            - Click ***Submit to Predict*** button to predict
            """)

st.subheader('What is Machine Learning?')
with st.expander("Expand to see explanations"):
    st.markdown("""
                Machine learning is a branch of artificial intelligence (AI) that to enable computers to automatically *recognize patterns*, *make predictions*,
                or *make decisions* based on the information they gather and the experiences they accumulate. 
                It's all about teaching computers to learn and improve tasks by themselves through data-driven insights.

                There are **three main types** of machine learning: ***supervised learning***, ***unsupervised learning***, and ***reinforcement learning***, each with its own characteristics, advantages, and disadvantages.

                1. **Supervised Learning**:
                    - Definition: In supervised learning, the algorithm is trained on a labeled dataset, where each input example has a corresponding correct output. The goal is for the algorithm to learn a mapping from inputs to outputs.
                    - Pros:
                      1. Well-suited for tasks like classification and regression.
                      1. Can provide accurate predictions when provided with sufficient labeled data.
                    - Cons:
                      1. Requires a large amount of labeled data, which can be costly to obtain.
                      1. May perform poorly if the training data is biased or not representative of the real-world distribution.
                2. **Unsupervised Learning**:
                    - Definition: Unsupervised learning deals with unlabeled data. The algorithm tries to find patterns, structures, or groupings in the data without any predefined outputs.
                    - Pros:
                      1. Useful for discovering hidden patterns or relationships in data.
                      1. Doesn't require labeled data, making it more versatile.
                    - Cons:
                      1. Results can be subjective and may require interpretation.
                      1. Evaluation can be challenging because there are no clear correct answers.
                3. **Reinforcement Learning**:
                    - Definition: Reinforcement learning involves an agent learning to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns to maximize its cumulative reward over time.
                    - Pros:
                      1. Suitable for tasks with sequential decision-making, like robotics and game playing.
                      1. Can adapt to changing environments.
                    - Cons:
                      1. Can be computationally intensive and require a lot of training.
                      1. Learning can be slow, and initial exploration may lead to suboptimal results.    
                """)     
         


# Get flat coordinate:
coord = find_postal(flat_address)
try:
    flat_coord = pd.DataFrame({'address':[coord.get('results')[0].get('ADDRESS')],
                            'LATITUDE':[coord.get('results')[0].get('LATITUDE')], 
                            'LONGITUDE':[coord.get('results')[0].get('LONGITUDE')]})
except IndexError:
    st.error('Oops! Address is not valid! Please enter a valid address!')
    pass

@st.cache(allow_output_mutation=True)
def load_data(filepath):
    return pd.read_csv(filepath)

supermarket_coord = load_data('Data/supermarket_coordinates_clean.csv')
school_coord = load_data('Data/school_coordinates_clean.csv')
# hawker_coord = load_data('Data/hawker_coordinates_clean.csv')
# shop_coord = load_data('Data/shoppingmall_coordinates_clean.csv')
# park_coord = load_data('Data/parks_coordinates_clean.csv')
mrt_coord = load_data('Data/MRT_coordinates.csv')[['STN_NAME','Latitude','Longitude']]
cpi = pd.read_csv('Data/CPI.csv')


## Get nearest and number of amenities in 2km radius
# Supermarkets
nearest_supermarket,supermarkets_2km = find_nearest(flat_coord, supermarket_coord)
flat_supermarket = pd.DataFrame.from_dict(nearest_supermarket).T
flat_supermarket = flat_supermarket.rename(columns={0: 'flat', 1: 'supermarket', 2: 'supermarket_dist',
                                                    3: 'num_supermarket_2km'}).reset_index().drop(['index'], axis=1)
supermarkets_2km['type'] = ['Supermarket']*len(supermarkets_2km)

# Primary Schools
nearest_school,schools_2km = find_nearest(flat_coord, school_coord)
flat_school = pd.DataFrame.from_dict(nearest_school).T
flat_school = flat_school.rename(columns={0: 'flat', 1: 'school', 2: 'school_dist',
                                          3: 'num_school_2km'}).reset_index().drop('index', axis=1)
schools_2km['type'] = ['School']*len(schools_2km)

# MRT
nearest_mrt,mrt_2km = find_nearest(flat_coord, mrt_coord)
flat_mrt = pd.DataFrame.from_dict(nearest_mrt).T
flat_mrt = flat_mrt.rename(columns={0: 'flat', 1: 'mrt', 2: 'mrt_dist',
                                    3: 'num_mrt_2km'}).reset_index().drop('index', axis=1)
mrt_2km['type'] = ['MRT']*len(mrt_2km)

amenities = pd.concat([supermarkets_2km, schools_2km,mrt_2km])
amenities = amenities.rename(columns={'lat':'LATITUDE', 'lon':'LONGITUDE'})

# Distance from Dhoby Ghaut
dist_dhoby = dist_from_location(flat_coord, (1.299308, 103.845285))
flat_coord['dist_dhoby'] = [list(dist_dhoby.values())[0][1]]

## Concat all dataframes
flat_coord = pd.concat([flat_coord, flat_supermarket.drop(['flat'], axis=1), 
                        flat_school.drop(['flat'], axis=1),
                        flat_mrt.drop(['flat'], axis=1)],
                       axis=1)
# st.dataframe(flat_coord)

## ENCODING VARIABLES
# Flat Type
replace_values = {'2 ROOM':0, '3 ROOM':1, '4 ROOM':2, '5 ROOM':3, 'EXECUTIVE':4}
flat_coord['flat_type'] = replace_values.get(flat_type)

# Get Storey
flat_coord['storey'] = storey

# Floor Area
flat_coord['floor_area_sqm'] = floor_area

# Lease commence date
flat_coord['lease_commence_date'] = lease_commence_date


# Remaining lease:
flat_coord['remaining_lease'] = 99 - (datetime.datetime.now().year - lease_commence_date)

# Consumer Price Index (cpi):
flat_coord['cpi'] = cpi.iloc[-1]['cpi']

d_region = {'ANG MO KIO':'North East', 'BEDOK':'East', 'BISHAN':'Central', 'BUKIT BATOK':'West', 'BUKIT MERAH':'Central',
       'BUKIT PANJANG':'West', 'BUKIT TIMAH':'Central', 'CENTRAL AREA':'Central', 'CHOA CHU KANG':'West',
       'CLEMENTI':'West', 'GEYLANG':'Central', 'HOUGANG':'North East', 'JURONG EAST':'West', 'JURONG WEST':'West',
       'KALLANG/WHAMPOA':'Central', 'MARINE PARADE':'Central', 'PASIR RIS':'East', 'PUNGGOL':'North East',
       'QUEENSTOWN':'Central', 'SEMBAWANG':'North', 'SENGKANG':'North East', 'SERANGOON':'North East', 'TAMPINES':'East',
       'TOA PAYOH':'Central', 'WOODLANDS':'North', 'YISHUN':'North'}
region_dummy = {'region_East':[0], 'region_North':[0], 'region_North East':[0], 'region_West':[0]}
region = d_region.get(town)
if region == 'East': region_dummy['region_East'][0] += 1
elif region == 'North': region_dummy['region_North'][0] += 1
elif region == 'North East': region_dummy['region_North East'][0] += 1
elif region == 'West': region_dummy['region_West'][0] += 1
#region_dummy
flat_coord = pd.concat([flat_coord, pd.DataFrame.from_dict(region_dummy)], axis=1)

# Flat Model
replace_values = {'Model A':'model_Model A', 'Simplified':'model_Model A', 'Model A2':'model_Model A', 
                  'Standard':'Standard', 'Improved':'Standard', '2-room':'Standard',
                  'New Generation':'model_New Generation',
                  'Apartment':'model_Apartment', 'Premium Apartment':'model_Apartment',
                  'Maisonette':'model_Maisonette', 'Executive Maisonette':'model_Maisonette', 
                  'Special':'model_Special', 'Terrace':'model_Special', 'Adjoined flat':'model_Special', 
                    'Type S1S2':'model_Special', 'DBSS':'model_Special','3Gen':'model_special'}
d = {'model_Apartment':[0], 'model_Maisonette':[0], 'model_Model A':[0], 'model_New Generation':[0], 'model_Special':[0]}
if replace_values.get(flat_model) != 'Standard': d[replace_values.get(flat_model)][0] += 1


df = pd.DataFrame.from_dict(d)
flat_coord = pd.concat([flat_coord, pd.DataFrame.from_dict(d)], axis=1)
flat_coord['selected_flat'] = [1] # for height of building


# flat1 = flat_coord[['flat_type', 'storey_range', 'floor_area_sqm', 'lease_commence_date',
#        'school_dist', 'num_school_2km', 'hawker_dist', 'num_hawker_2km',
#        'park_dist', 'num_park_2km', 'mall_dist', 'num_mall_2km', 'mrt_dist',
#        'num_mrt_2km', 'supermarket_dist', 'num_supermarket_2km', 'dist_dhoby',
#        'region_East', 'region_North', 'region_North East', 'region_West',
#        'model_Apartment', 'model_Maisonette', 'model_Model A',
#        'model_New Generation', 'model_Special']]

flat1 = flat_coord[['flat_type', 'storey', 'floor_area_sqm', 'lease_commence_date',
       'remaining_lease','school_dist', 'num_school_2km', 'mrt_dist',
       'num_mrt_2km', 'supermarket_dist', 'num_supermarket_2km', 
       'dist_dhoby','cpi','region_East', 'region_North', 
       'region_North East', 'region_West',
       'model_Apartment', 'model_Maisonette', 'model_Model A',
       'model_New Generation', 'model_Special']]
# display(flat1)
flat1_predict = model_rf.predict(flat1)

st.text(" ")

## EXPANDER FOR AMENITIES INFORMATION
st.subheader('Amenities Within 2km Radius')
with st.expander("MRT/LRT Station"):
    st.subheader('Nearest MRT/LRT Station: **%s** (%0.2fkm)' % (flat_coord.iloc[0]['mrt'], flat_coord.iloc[0]['mrt_dist']))    
with st.expander("Primary School"):
    st.subheader('Nearest Primary School: **%s** (%0.2fkm)' % (flat_coord.iloc[0]['school'], flat_coord.iloc[0]['school_dist']))    
with st.expander("Supermarket/Shop"):
    st.subheader('Nearest Supermarket/Shop: **%s** (%0.2fkm)' % (flat_coord.iloc[0]['supermarket'], flat_coord.iloc[0]['supermarket_dist']))
      
st.markdown("#")

# st.header(f'The Predicted HDB Resale Price is SG${flat1_predict[0]:,.0f}')
textresult = f'<p style="font-family:Arial;color:Blue; font-size: 28px;">The Predicted HDB Resale Price is <strong>SG${flat1_predict[0]:,.0f}</strong></p>'
st.markdown(textresult,unsafe_allow_html=True)


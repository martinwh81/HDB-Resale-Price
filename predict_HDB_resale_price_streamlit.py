import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from  matplotlib.ticker import FuncFormatter
from utils_functions import find_postal, find_nearest, dist_from_location, map, map_flats_year, _max_width_
import streamlit.components.v1 as components
import datetime
import joblib
import gc

st.set_page_config(layout="wide")
tab1, tab2 = st.tabs(["📈 Manual Prediction 📈","💻  Machine Learning 💻"])


def disp_header_image():
    st.image('Comm_image_small.png')

#################################################################################################################################################
#################################################################################################################################################

with tab1:    
    disp_header_image()
    # st.image('Comm_image_small.png')
    # col1, col2 = st.columns([7.5,2.5])
    # with col1:
    #     # text_title= f'<center><p style="font-family:Arial;color:Blue; font-size: 28px;"><strong>Machine Learning for HDB Resale Price Prediction<br>(Guess the Price Game)</strong></p>'
    #     st.text(" ")
    #     text_title= f'<center><p style="font-family:Arial;color:Blue; font-size: 24px;"><strong>Machine Learning for HDB Resale Price Prediction</strong></p>'
    #     st.markdown(text_title,unsafe_allow_html=True)
    # with col2:
    # st.image('Guess_the_price_image.png')


    text_title= f'<center><p style="font-family:Arial;color:Blue; font-size: 28px;"><strong>Guess the Price - Manual Prediction of HDB Resale Price</strong></p>'
    st.markdown(text_title,unsafe_allow_html=True)
    
    # st.title('Guess the Price')
    # st.title('Prediction of Singapore HDB Resale Prices using Supervised Machine Learning')

    st.text(" ")
    st.text(" ")


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
    cpi['month'] = pd.to_datetime(cpi['month'])

    prices =load_data('Data/Data_2017_onwards.csv')
    prices['month'] = pd.to_datetime(prices['month']) # to datetime
    replace_values = {'NEW GENERATION':'New Generation', 'SIMPLIFIED':'Simplified', 'STANDARD':'Standard', 'MODEL A-MAISONETTE':'Maisonette', 'MULTI GENERATION':'Multi Generation', 'IMPROVED-MAISONETTE':'Executive Maisonette', 'Improved-Maisonette':'Executive Maisonette', 'Premium Maisonette':'Executive Maisonette', '2-ROOM':'2-room', 'MODEL A':'Model A', 'MAISONETTE':'Maisonette', 'Model A-Maisonette':'Maisonette', 'IMPROVED':'Improved', 'TERRACE':'Terrace', 'PREMIUM APARTMENT':'Premium Apartment', 'Premium Apartment Loft':'Premium Apartment', 'APARTMENT':'Apartment', 'Type S1':'Type S1S2', 'Type S2':'Type S1S2'}
    prices = prices.replace({'flat_model': replace_values})

    #################################################################################################################################################
    sns.set()
    # Function for lollipop charts
    @st.cache(allow_output_mutation=True)
    def loll_plot(df, x, y, subtitle, xlabel, xlim, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        plt.rc('axes', axisbelow=True)
        plt.grid(linestyle='--', alpha=0.4)
        plt.hlines(y=df.index, xmin=0, xmax=df[x], color=df.color, linewidth=1)
        plt.scatter(df[x], df.index, color=df.color, s=500)
        for i, txt in enumerate(df[x]):
            plt.annotate(str(round(txt)), (txt, i), color='white', fontsize=9, ha='center', va='center')
        plt.annotate(subtitle, xy=(1, 0), xycoords='axes fraction', fontsize=20,
                        xytext=(-5, 5), textcoords='offset points',
                        ha='right', va='bottom')
        plt.yticks(df.index, df[y]); plt.xticks(fontsize=12); plt.xlim(xlim)
        plt.xlabel(xlabel, fontsize=14)
        return fig, ax

    @st.cache(allow_output_mutation=True)
    def box_plot(df,df_median, x, y, xlabel,ylabel,figsize):
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.boxplot(data=df,x=x,y=y,order=df_median.index,showfliers = False)
        add_median_labels(ax)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)         
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        return fig, ax

    def add_median_labels(ax, fmt='.0f'):
        lines = ax.get_lines()
        boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
        lines_per_box = int(len(lines) / len(boxes))
        for median in lines[4:len(lines):lines_per_box]:
            x, y = (data.mean() for data in median.get_data())
            # choose value depending on horizontal or vertical plot orientation
            value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
            text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                        fontweight='bold', color='white')
            # create median-colored border around white text for contrast
            text.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground=median.get_color()),
                path_effects.Normal(),
            ])


    @st.cache(allow_output_mutation=True)
    def lineplot(df, x, y, marker, markersize, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.lineplot(df,x=x,y=y,marker=marker,markersize=markersize)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
        ax.grid(True)        
        return fig, ax
    
    @st.cache(allow_output_mutation=True)
    def load_model():
        model_rf = joblib.load('rf_compressed.pkl')
        return model_rf

    model_rf = load_model()
            

    st.subheader("HDB Median Resale Price Charts (Box Plots)\n ***(Use the following charts to manually predict HDB Resale Price in 'Guess the Price' game)***")
    
    st.markdown("""                
                **Challenge**: Can you use the charts below to predict a specific HDB resale price? Try it for yourself!
                """)
    price_game = prices.loc[prices['month'].dt.year>=2017].copy()
    price_game = price_game[['month','town','flat_model','flat_type','floor_area_sqm','storey_range','lease_commence_date','resale_price']]
    price_game['resale_price'] = price_game['resale_price'] / 1000
    price_game.head()

    ## Chart by Town:
    # price_median_town = price_game.groupby('town')['resale_price'].median().reset_index().sort_values(by='resale_price',ascending=True).reset_index(drop=True)    
    # price_median_town['color'] = ['#f8766d'] + ['#3c78d8']*(len(price_median_town)-2) + ['#00ba38']    
    # fig, ax =loll_plot(price_median_town,'resale_price','town','','Resale Price (SGD)',[50, 800],figsize=(8,10))    
    
    grouped = price_game.loc[:,['town', 'resale_price']].groupby(['town']).median().sort_values(by='resale_price',ascending = False)
    fig, ax = box_plot(price_game, grouped,'resale_price','town','Resale Price (SGD)','Town',figsize=(8,10))

    ax.set_xticklabels(f'{x:,.0f}K' for x in ax.get_xticks())
    ax.set_title('Median Resale Price by Town (Sales data from 2017-2023)',{'fontsize':18})
    st.pyplot(fig)
    

    ## Chart by Flat Model:
    # price_median_flat_model = price_game.groupby('flat_model')['resale_price'].median().reset_index().sort_values(by='resale_price',ascending=True).reset_index(drop=True)    
    # price_median_flat_model['color'] = ['#f8766d'] + ['#3c78d8']*(len(price_median_flat_model)-2) + ['#00ba38']
    # fig, ax = loll_plot(price_median_flat_model,'resale_price','flat_model','','Resale Price (SGD)',[200, 1100],figsize = (8,8))

    grouped = price_game.loc[:,['flat_model', 'resale_price']].groupby(['flat_model']).median().sort_values(by='resale_price',ascending = False)
    fig, ax = box_plot(price_game, grouped,'resale_price','flat_model','Resale Price (SGD)','Flat Model',figsize=(8,8))

    ax.set_xticklabels(f'{x:,.0f}K' for x in ax.get_xticks())
    ax.set_title('Median Resale Price by Flat Model (Sales data from 2017-2023)',{'fontsize':18})
    st.pyplot(fig)
    

    # Chart by Flat Type:
    # price_median_flat_type = price_game.groupby('flat_type')['resale_price'].median().reset_index().sort_values(by='resale_price',ascending=True).reset_index(drop=True)
    # price_median_flat_type['color'] = ['#f8766d'] + ['#3c78d8']*(len(price_median_flat_type)-2) + ['#00ba38']
    # fig, ax = loll_plot(price_median_flat_type,'resale_price','flat_type','','Resale Price (SGD)',[100, 900], figsize=(8,6))

    grouped = price_game.loc[:,['flat_type', 'resale_price']].groupby(['flat_type']).median().sort_values(by='resale_price',ascending = False)
    fig, ax = box_plot(price_game, grouped,'resale_price','flat_type','Resale Price (SGD)','Flat Type',figsize=(8,6))

    ax.set_xticklabels(f'{x:,.0f}K' for x in ax.get_xticks())
    ax.set_title('Median Resale Price by Flat Type (HDB sales data from 2017 - 2023)',{'fontsize':18})
    st.pyplot(fig)
    

    ## Chart by Floor Area:
    bins = range(20,270,20)
    price_game['floor_area_range'] = pd.cut(x=price_game['floor_area_sqm'],bins = bins)
    # price_median_floor_area = price_game.groupby('floor_area_range')['resale_price'].median().reset_index().sort_values(by='resale_price',ascending=True).reset_index(drop=True)
    # price_median_floor_area['resale_price'] = price_median_floor_area['resale_price']
    # price_median_floor_area['color'] = ['#f8766d'] + ['#3c78d8']*(len(price_median_floor_area)-2) + ['#00ba38']        
    # fig, ax = loll_plot(price_median_floor_area,'resale_price','floor_area_range','','Resale Price (SGD)',[100, 1300],figsize=(8,6))
    
    grouped = price_game.loc[:,['floor_area_range', 'resale_price']].groupby(['floor_area_range']).median().sort_values(by='resale_price',ascending = False)
    fig, ax = box_plot(price_game, grouped,'resale_price','floor_area_range','Resale Price (SGD)','Flor Area (sqm)',figsize=(8,6))

    ax.set_xticklabels(f'{x:,.0f}K' for x in ax.get_xticks())    
    ax.set_title('Median Resale Price by Floor Area (HDB sales data from 2017 - 2023)',{'fontsize':18})
    st.pyplot(fig)
    

    ## Chart by Storey:
    # price_median_storey_range = price_game.groupby('storey_range')['resale_price'].median().reset_index().sort_values(by='resale_price',ascending=True).reset_index(drop=True)    
    # price_median_storey_range['color'] = ['#f8766d'] + ['#3c78d8']*(len(price_median_storey_range)-2) + ['#00ba38']    
    # fig, ax = loll_plot(price_median_storey_range,'resale_price','storey_range','','Resale Price (SGD)',[200, 1200],figsize=(10,8))

    grouped = price_game.loc[:,['storey_range', 'resale_price']].groupby(['storey_range']).median().sort_values(by='resale_price',ascending = False)
    fig, ax = box_plot(price_game, grouped,'resale_price','storey_range','Resale Price (SGD)','Storey',figsize=(10,8))

    ax.set_xticklabels(f'{x:,.0f}K' for x in ax.get_xticks())    
    ax.set_title('Median Resale Price by Storey Range (HDB sales data from 2017 - 2023)',{'fontsize':18})
    st.pyplot(fig)
    

    ## Chart by Lease Commence Date:
    bins = range(1962,2024,4)
    price_game['lease_date_range'] = pd.cut(x=price_game['lease_commence_date'],bins = bins)

    # price_median_lease_commence_date = price_game.groupby('lease_date_range')['resale_price'].median().reset_index()
    # price_median_lease_commence_date['color'] = ['#f8766d'] + ['#3c78d8']*(len(price_median_lease_commence_date)-2) + ['#00ba38']
    # fig, ax = loll_plot(price_median_lease_commence_date,'resale_price','lease_date_range','','Resale Price (SGD)',[100, 700],figsize=(10,8))

    grouped = price_game.loc[:,['lease_date_range', 'resale_price']].groupby(['lease_date_range']).median().sort_values(by='resale_price',ascending = False)
    fig, ax = box_plot(price_game, grouped,'resale_price','lease_date_range','Resale Price (SGD)','Lease Commencement Date',figsize=(10,8))

    ax.set_xticklabels(f'{x:,.0f}K' for x in ax.get_xticks())
    ax.set_title('Median Resale Price by Lease Commence Date (HDB sales data from 2017 - 2023)',{'fontsize':18})
    st.pyplot(fig)
    

    # Chart for CPI:
    cpi_yearly_median = cpi.groupby(cpi['month'].dt.year)['cpi'].median().reset_index()
    cpi_yearly_median = cpi_yearly_median.rename(columns={'month':'Year','cpi':'Consumer Price Index'})
    cpi_yearly_median = cpi_yearly_median.loc[cpi_yearly_median['Year']>=2017].reset_index(drop=True)
    # fig, ax = plt.subplots(figsize=(7,7))
    # ax = sns.lineplot(cpi_yearly_median,x='Year',y='Consumer Price Index',marker='o',markersize=10)
    # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))    
    # ax = lineplot(cpi_yearly_median,x='Year',y='Consumer Price Index',marker='o',markersize=10, figsize=(7,7))

    fig, ax = lineplot(cpi_yearly_median,x='Year',y='Consumer Price Index',marker ='o',markersize=10, figsize=(7,7))   
    for i,val in enumerate(cpi_yearly_median['Consumer Price Index']):
        plt.annotate(f'{val:.1f}',(cpi_yearly_median.iloc[i,0]-0.2,cpi_yearly_median.iloc[i,1]+0.2))    
    ax.set_xlabel('Year',{'fontsize':10})
    ax.set_ylabel('Consumer Price Index',{'fontsize':10})
    ax.set_title('Consumer Price Index (Housing and Utilities)',{'fontsize':12})    
    st.pyplot(fig)
    
    
    # del(prices,price_median_town,price_median_flat_model,price_median_flat_type,
    #     price_median_floor_area,price_median_storey_range,price_median_lease_commence_date,
    #     cpi_yearly_median)
    del(prices,grouped, cpi_yearly_median)

    gc.collect()

#################################################################################################################################################
#################################################################################################################################################
with tab2:
    disp_header_image()
    text_title= f'<center><p style="font-family:Arial;color:Blue; font-size: 28px;"><strong>Guess the Price - Machine Learning Prediction of HDB Resale Price</strong></p>'
    st.markdown(text_title,unsafe_allow_html=True)




        
    #################################################################################################################################################

    st.subheader('What is Machine Learning?')
    with st.expander("Expand to see explanation"):
        st.markdown("""
                    **Machine learning** is a branch of artificial intelligence (AI) that  enables computers to automatically *recognize patterns*, *make predictions*,
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





    #################################################################################################################################################    
    def plot_feature_importance():
        fig = plt.figure(figsize=(6,6))
        feat_imp = pd.DataFrame({'Features': feature_names, 'Feature Importance': model_rf.feature_importances_}).sort_values('Feature Importance', ascending=False)
        sns.barplot(y='Features', x='Feature Importance', data=feat_imp)    
        plt.title('Attribute/Feature Importance', size=14)
        return fig

    st.subheader('Machine Learning for HDB Resale Price Prediction')
    with st.expander("Expand to see explanation"):
        st.markdown("""
                    As you probably have tried, using **Median Resale Price per HDB attribute** is not the best way to predict the resale price of a specific HDB. 
                    There are a few reasons why it may not work well:
                    - The charts are univariate in nature (ie. considering only 1 attribute at a time), whereas HDB resale price prediction is **multivariate in nature (affected by interactions of multiple attributes simultaneously**.
                    - Predicting a **specific HDB price** using aggregated data (median) will not be accurate as the **aggregated data has lost information on local effect of the feature**.
                    - There are **other important attributes** affecting the resale price of HDB, such as distances to amenities (eg. supermarkets, schools, MRT/LRT stations). These were not considered above.
                    - The interactions of the HDB attributes to the resale price are **complex and nonlinear** in nature, so simple averaging/median will not work well.
                                    
                    This web app has a built-in **supervised machine learning model** to predict the current price of HDB Resale Price. 
                    The model is developed based on historical HDB Resale Prices data ([Data.gov.sg](https://data.gov.sg/)) from 2012 to 2023. 
                    In addition to the **standard HDB attributes** (Town, Flat Model, Flat Type, Floor Area, Storey, Lease Commencement), important features
                    related to **distance to amenities** (Supermarkets, Primary Schools, MRT/LRT Stations) and **Consumer Price Index** were also used to improve the accuracy of
                    the model.

                    Each time a query is submitted for prediction, coordinates value will be obtained from ([onemap.gov.sg](https://www.onemap.gov.sg/))
                    and the nearest distances to different amenities are calculated. These values are then combined with the rest of HDB attributes. The combined values (a.k.a *Machine Learning Features*)
                    are then fed into machine learning model to predict the HDB Resale Price.


                    The following graph shows the most important attribute/features for predicting HDB resale price:
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
        fig = plot_feature_importance()
        st.pyplot(fig)
        st.text(" ")
        st.text(" ")
        st.markdown("""
                    **🤔 Now, can you try to predict the HDB Resale Price using Machine Learning and compare the result with your own manual prediction?**
                    """)


    #################################################################################################################################################

    options_question =  [f'Q{str(i)}' for i in range(1,21)]
    questions= [['457 ANG MO KIO AVE 10','ANG MO KIO','Adjoined flat','EXECUTIVE',176,15,1980],
                ['25 SIN MING RD','BISHAN','Improved','4 ROOM',88,5,1974],
                ['546A SEGAR RD','BUKIT PANJANG','Improved','5 ROOM',112,14,2015],
                ['445 TAMPINES ST 42','TAMPINES','Simplified','3 ROOM',64,2,1986],
                ['28 MARINE CRES','MARINE PARADE','Standard','5 ROOM',123,11,1975],
                ['527 JURONG WEST ST 52','JURONG WEST','New Generation','3 ROOM',67,11,1983],
                ['21 CHAI CHEE RD','BEDOK','Improved','3 ROOM',65,5,1972],
                ['659C PUNGGOL EAST','PUNGGOL','Model A','2 ROOM',47,2,2018],
                ['804A KEAT HONG CL','CHOA CHU KANG','Model A','4 ROOM',93,14,2017],
                ['33 GHIM MOH LINK','QUEENSTOWN','Model A','3 ROOM',68,35,2018],
                ['881 WOODLANDS ST 82','WOODLANDS','Improved','5 ROOM',123,11,1996],
                ['296A BT BATOK ST 22','BUKIT BATOK','Premium Apartment','5 ROOM',115,26,2018],
                ['1B CANTONMENT RD','CENTRAL AREA','Type S1S2','4 ROOM',93,20,2011],
                ['102 PASIR RIS ST 12','PASIR RIS','Model A','4 ROOM',104,11,1988],
                ['7 TOH YI DR','BUKIT TIMAH','Apartment','EXECUTIVE',142,5,1989],
                ['440C CLEMENTI AVE 3','CLEMENTI','Improved','5 ROOM',112,14,2018],
                ['302 SERANGOON AVE 2','SERANGOON','Improved','3 ROOM',69,8,1985],
                ['365C SEMBAWANG CRES','SEMBAWANG','Model A','4 ROOM',93,11,2019],
                ['717 CLEMENTI WEST ST 2','CLEMENTI','Improved','5 ROOM',118,23,1981],
                ['172 LOR 1 TOA PAYOH','TOA PAYOH','Improved','5 ROOM',124,20,1995]]



    options_town = sorted(list(['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                    'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI',
                    'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                    'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG',
                    'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN',
                    'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS','PUNGGOL']))
    options_flat_model = sorted(list(['Model A', 'Improved', 'Premium Apartment', 'Standard',
                                                    'New Generation', 'Maisonette', 'Apartment', 'Simplified',
                                                    'Model A2', 'DBSS', 'Terrace', 'Adjoined flat', 'Multi Generation',
                                                    '2-room', 'Executive Maisonette', 'Type S1S2']))
    options_flat_type = list(['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE'])
    options_storey = list(range(1, 52))
    options_lease_commence_date = list(reversed(range(1966, 2022)))


    st.subheader('How to Use this App to Predict HDB Resale Price (2 options)')
    st.markdown("""                
                1. **By selecting the *question number (ie. Q1-Q20) from 'Guess the Price'* game**
                """)
    # with st.sidebar:
    question_selected = st.selectbox('Question Number', options_question, index=0)           
    question_index = int(question_selected[1:])-1

    selected_flat_address = questions[question_index][0]
    selected_town_index = options_town.index(questions[question_index][1])
    selected_flat_model_index = options_flat_model.index(questions[question_index][2])
    selected_flat_type_index = options_flat_type.index(questions[question_index][3])
    selected_floor_area = questions[question_index][4]
    selected_storey = options_storey.index(questions[question_index][5])
    selected_lease_commence_date = options_lease_commence_date.index(questions[question_index][6])
    # print(selected_flat_address,selected_town_index,selected_flat_model_index,selected_flat_type_index,selected_floor_area,selected_storey,selected_lease_commence_date)
    
    st.markdown("""
                2. **By manually filling in the desired HDB attributes below and click *'Submit to Predict'***                
                """)    
    

    with st.form('User Input HDB Features'):
        col1, col2 = st.columns(2)        
    # with st.sidebar.form('User Input HDB Features'):    
        with col1:    
            flat_address = st.text_input("Flat Address or Postal Code", selected_flat_address) # flat address    
            town = st.selectbox('Town', options_town, index=selected_town_index) 
            flat_model = st.selectbox('Flat Model', options_flat_model, index=selected_flat_model_index)            
            flat_type = st.selectbox('Flat Type', options_flat_type, index=selected_flat_type_index)        
        with col2:   
            
            floor_area = st.slider("Floor Area (sqm)", 34,280,selected_floor_area) # floor area
            # storey = st.selectbox('Storey', list(['01 TO 03','04 TO 06','07 TO 09','10 TO 12','13 TO 15',
            #                                             '16 TO 18','19 TO 21','22 TO 24','25 TO 27','28 TO 30',
            #                                             '31 TO 33','34 TO 36','37 TO 39','40 TO 42','43 TO 45',
            #                                             '46 TO 48','49 TO 51']), index=3)
            # storey = st.slider('Storey', 1,51,7)
            storey = st.selectbox('Storey', options_storey,selected_storey)            
            lease_commence_date = st.selectbox('Lease Commencement Date', options_lease_commence_date, index=selected_lease_commence_date)        
            submitted1 = st.form_submit_button(label = 'Submit to Predict')

    #################################################################################################################################################

    # Get flat coordinate:
    coord = find_postal(flat_address)
    try:
        flat_coord = pd.DataFrame({'address':[coord.get('results')[0].get('ADDRESS')],
                                'LATITUDE':[coord.get('results')[0].get('LATITUDE')], 
                                'LONGITUDE':[coord.get('results')[0].get('LONGITUDE')]})
    except IndexError:
        st.error('Oops! Address is not valid! Please enter a valid address!')
        pass


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

    ###############################################################################################################################

    df_input = pd.DataFrame({'Flat Address':[flat_address],
                            'Town': [town],
                            'Flat Model': [flat_model],
                            'Flat Type':[flat_type],
                            'Floor Area': [floor_area],
                            'Storey': [storey],
                            'Lease Commencement Date': [lease_commence_date]
                            },index = ['Attribute Value'])

    # textinput = f'<p style="font-family:Arial;color:Blue; font-size: 28px;">You have selected the following HDB attribute:</p>'
    # st.markdown(textinput, unsafe_allow_html=True)
    st.subheader('You have selected the following HDB attribute')
    st.dataframe(df_input)

    # Nearest Amenities:
    st.subheader('Nearest Amenities')
    with st.expander("Expand to see details"):
        st.markdown('**Nearest MRT/LRT Station**: *%s (%0.2fkm)*' % (flat_coord.iloc[0]['mrt'], flat_coord.iloc[0]['mrt_dist']))    
        st.markdown('**Nearest Primary School**: *%s (%0.2fkm)*' % (flat_coord.iloc[0]['school'], flat_coord.iloc[0]['school_dist']))        
        st.markdown('**Nearest Supermarket/Shop**: *%s (%0.2fkm)*' % (flat_coord.iloc[0]['supermarket'], flat_coord.iloc[0]['supermarket_dist']))




    # st.header(f'The Predicted HDB Resale Price is SG${flat1_predict[0]:,.0f}')
    textresult = f'<p style="font-family:Arial;color:Blue; font-size: 28px;">The Predicted HDB Resale Price is <strong>SG${flat1_predict[0]:,.0f}</strong></p>'
    st.markdown(textresult,unsafe_allow_html=True)

    textdisclaimer = f'<p style="font-family:Arial;color:Red; font-size: 14px;"><em>(Disclaimer: This app is purely for educational purpose only. Please do not use the prediction for buying/selling decision.)</em></p>'
    st.markdown(textdisclaimer,unsafe_allow_html=True)



###############################################################################################################################
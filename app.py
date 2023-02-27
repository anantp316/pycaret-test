import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
#from pycaret.regression import setup, compare_models, pull, save_model, load_model
#from pycaret.classification import setup, compare_models, pull, save_model, load_model
#from pycaret.clustering import setup, pull, save_model, load_model
import pycaret
from pycaret import regression,classification,clustering
from sklearn.datasets import load_diabetes
# Page layout
## Page expands to full width
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='EzML',page_icon=":robot_face:",
    layout='wide')
#----------------------------------#
import os

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)




# Pycaret Section



#---------------------------------#
st.write("""
# EzML - Automating ML Model Building
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar:
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])

with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)


#---------------------------------#
#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')


if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Awaiting for CSV file to be uploaded.")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    df.head()
    #profile_df = df.profile_report()
    #st_profile_report(profile_df)


if choice == "Modelling":
    choice = st.radio('**Select your task**',['Regression','Classification','Clustering'])

    if choice=='Regression':
        # Regression Work Starts
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):

            pycaret.regression.setup(df, target=chosen_target, silent=True)
            setup_df = pycaret.regression.pull()
            st.dataframe(setup_df)
            best_model = pycaret.regression.compare_models()
            compare_df = pycaret.regression.pull()
            st.dataframe(compare_df)
            pycaret.regression.save_model(best_model, 'best_model')


        # Regression Work Ends


    if choice=='Classification':
        # Classification Work Starts
        # Regression Work Starts
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            pycaret.classification.setup(df, target=chosen_target, silent=True,fold_shuffle=True,fix_imbalance = True)
            setup_df = pycaret.classification.pull()
            st.dataframe(setup_df)
            best_model = pycaret.classification.compare_models()
            compare_df = pycaret.classification.pull()


        # Classification Work Ends



    if choice=='Clustering':
        # Clustering Work Starts
        # Regression Work Starts
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            pycaret.clustering.setup(df, target=chosen_target, silent=True)
            setup_df = pycaret.clustering.pull()
            st.dataframe(setup_df)
            best_model = pycaret.clustering.compare_models()
            compare_df = pycaret.clustering.pull()



if choice == "Download":
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")
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
#from explainerdashboard import ExplainerDashboard

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
    choice = st.radio("Navigation", ["Upload","Modelling", "Download"])

with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 0.1, 0.9, 0.8, 0.1)
    #seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)


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


if choice == "Modelling":
    choice = st.radio('**Select your task**',['Regression','Classification','Clustering'])

    if choice=='Regression':
        # Regression Work Starts
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            pycaret.regression.setup(df, target=chosen_target, silent=True,train_size=split_size,fold = 3)
            setup_df = pycaret.regression.pull()
            st.dataframe(setup_df)
            best_model = pycaret.regression.compare_models(turbo=True)
            compare_df = pycaret.regression.pull()
            st.dataframe(compare_df)
            pycaret.regression.save_model(best_model, 'best_model')
            best_reg = pycaret.classification.create_model(best_model)
            st.title("Analyzing the performance of your trained model on holdout set")
            st.subheader("Model Residual")
            #plot_choice = st.radio('**Available plots**',['auc','confusion_matrix','boundary','feature_all','tree'])
            pycaret.regression.plot_model(estimator=best_reg,plot='residual',display_format = 'streamlit')
            st.subheader("Model Prediction Error Plot")
            pycaret.regression.plot_model(estimator=best_reg,plot='error',display_format = 'streamlit')
            #st.subheader("Feature Importance")
            #pycaret.classification.plot_model(estimator=best_clf,plot='feature_all',display_format = 'streamlit')
            st.title("Interpreting built ML Model using Feature Importance")
            pycaret.regression.plot_model(estimator=best_reg,plot='feature',display_format = 'streamlit')
            st.title("Comprehensive Model Evaluation")
            pycaret.regression.evaluate_model(best_reg,use_train_data=True)
            #st.write(pycaret.regression.evaluate_model(best_reg,use_train_data=True))
            #pycaret.regression.dashboard(best_reg,display_format='jupyterlab')
            #pycaret.regression.dashboard(best_reg,display_format='external')
            


        # Regression Work Ends


    if choice=='Classification':
        # Classification Work Starts
        # Regression Work Starts
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            pycaret.classification.setup(df, target=chosen_target, silent=True,train_size=split_size,fold = 3)
            setup_df = pycaret.classification.pull()
            st.dataframe(setup_df)
            best_model = pycaret.classification.compare_models(turbo=True,exclude = ['knn','qda'])
            compare_df = pycaret.classification.pull()
            st.dataframe(compare_df)
            pycaret.classification.save_model(best_model, 'best_model')
            best_clf = pycaret.classification.create_model(best_model)
            st.title("Analyzing the performance of your trained model on holdout set")
            st.subheader("Model AUC")
            #plot_choice = st.radio('**Available plots**',['auc','confusion_matrix','boundary','feature_all','tree'])
            pycaret.classification.plot_model(estimator=best_clf,plot='auc',display_format = 'streamlit')
            st.subheader("Model Decision Boundary")
            pycaret.classification.plot_model(estimator=best_clf,plot='boundary',display_format = 'streamlit')
            #st.subheader("Feature Importance")
            #pycaret.classification.plot_model(estimator=best_clf,plot='feature_all',display_format = 'streamlit')
            st.title("Interpreting built ML Model using Feature Importance")
            pycaret.classification.plot_model(estimator=best_clf,plot='feature',display_format = 'streamlit')
            #st.subheader("Summary plot using SHAP values")
            #interpret_choice = st.radio('**Available plots**',['summary','correlation','reason'])
            #pycaret.classification.interpret_model(estimator=best_model,plot='summary')
            #st.subheader("Dependency Plot")
            #pycaret.classification.interpret_model(estimator=best_model,plot='correlation',display_format = 'streamlit')
            #st.subheader("Force plot using SHAP values")
            #pycaret.classification.interpret_model(estimator=best_model,plot='reason',display_format = 'streamlit')
            st.title("Comprehensive Model Evaluation")
            pycaret.classification.evaluate_model(best_reg,use_train_data=True)
            #pycaret.classification.dashboard(best_clf,display_format='jupyterlab')
            #pycaret.classification.dashboard(best_clf,display_format='external')
            
            #st.write(pycaret.classification.evaluate_model(best_clf,use_train_data=True))
        # Classification Work Ends



    if choice=='Clustering':
        # Clustering Work Starts
        # Regression Work Starts
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            pycaret.clustering.setup(df, silent=True)
            setup_df = pycaret.clustering.pull()
            st.dataframe(setup_df)
            #best_model = pycaret.clustering.compare_models()
            #compare_df = pycaret.clustering.pull()
            #st.dataframe(compare_df)
            #pycaret.clustering.save_model(best_model, 'best_model')
            best_clus = pycaret.clustering.create_model('kmeans')
            pycaret.clustering.save_model(best_clus, 'best_model')
            st.title("Analyzing the performance of your trained model on holdout set")
            st.subheader("Cluster PCA Plot")
            #plot_choice = st.radio('**Available plots**',['auc','confusion_matrix','boundary','feature_all','tree'])
            pycaret.clustering.plot_model(model=best_clus,plot='cluster',display_format = 'streamlit')
            st.subheader("Cluster TsNE Plot")
            pycaret.clustering.plot_model(model=best_clus,plot='tsne',display_format = 'streamlit')
            #st.subheader("Feature Importance")
            #pycaret.classification.plot_model(estimator=best_clf,plot='feature_all',display_format = 'streamlit')
            st.title("Interpreting built ML Model using Silhouette Plot")
            pycaret.clustering.plot_model(model=best_clus,plot='silhouette',display_format = 'streamlit')
            


if choice == "Download":
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")

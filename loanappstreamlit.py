import streamlit as st
# from flask import Flask, render_template, url_for, request
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app)


from pandas import read_csv
import pandas as pd
import shap
from shap.plots._force_matplotlib import draw_additive_plot
import pickle
import datetime
import os
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

model = "static/lgb_classifier_model2_02_7.pkl"
explainer = "static/explainer.pkl"
shap_values = "static/shap_values_reduce.pkl"
neutral_features_values_filename = "static/neutral_value.csv"
features_filename = "static/features_reduce.csv"
summary_plot = "static/summary_plot.png"


df = pd.read_csv(features_filename,low_memory=False)
columns = df.columns.values

del_features_for_pred = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index', 'level_0','Unnamed: 0']
features_for_pred = list(filter(lambda v: v not in del_features_for_pred, columns))


with open(shap_values, 'rb') as handle:
    shap_values_loaded = pickle.load(handle)
# shap_values_loaded = shap_values_loaded[:,:,1]


with open(explainer, 'rb') as handle:
    explainer_loaded = pickle.load(handle)
    
with open(model, 'rb') as handle:
    model_loaded = pickle.load(handle)


def main():
    pages = {
        "Choix_des_variables": page_home,
        "Résultats": page_results,
    }
    if "page" not in st.session_state:
        st.session_state.update({
            # Default page
            "page": "Choix_des_variables"

            
        })
    
    with st.sidebar:
        page = st.radio("Choississez vos paramètres et/ou regarder les résultats", tuple(pages.keys()))

    pages[page]()


def page_home():
    st.write("## Choix d'un utilisateur préexistant ou non.")
    # utilisateur_bool = "Non"
    utilisateur_bool = st.selectbox("Voulez vous visualiser un utilisateur préexistant?:", ['Oui', 'Non'],key="utilisateur_bool")
    if utilisateur_bool == "Oui":
        list_ids = df['SK_ID_CURR'].values
        user_id_value = st.selectbox("Numéro d'utilisateur",list_ids,key="user_id_value")
    else :
        
        creditInput = st.slider('Revenu annuel',20000,4100000,25000,500,key="creditInput")
        annuiteInput = st.slider('Annuité',1000,500000,20000,500,key="annuiteInput")
        sexe = st.selectbox('Sexe:', ['Femme', 'Homme'],key="sexe")
        goodsInput = st.slider('Revenu annuel',500,5000000,15000,500,key="goodsInput")
        studies_degree = st.selectbox("Niveau scolaire:", ['Secondaire', 'Haute études','Haute études incomplètes','Inférieur secondaire', 'Licence'],key="studies_degree")
        occupation_type = st.selectbox("Catégorie métier:", ['ouvrier', 'personnel basique','comptable','manager', 'conducteur','commercial',
                                                             'nettoyage','cuisine','service privé','médical','sécurité','haute technique',
                                                             'serveur/barmen','ouvrier peu qualifié','immobilier','secrétaire','informaticien','RH'],key="occupation_type")
        job_start = st.date_input("Date d'embauche",datetime.date(2021, 1, 1),key="job_start")
        incomeInput = st.slider('Revenu annuel',0,20000000,15000,500,key="incomeInput")
        daybirth = st.date_input("Date de naissance",datetime.date(1980, 1, 1),key="daybirth")
        st.write("### laisser à -1 si aucune voiture")
        carInput = st.slider('Age voiture',-1,100,-1,1,key="carInput")
        family_status = st.selectbox("Situation familiale:", ['marié(e)', 'célibataire','mariage civil','séparé', 'veuf(ve)', 'autre'],key="family_status")
 


def page_results():
    importante_features = ["AMT_ANNUITY","AMT_CREDIT","AMT_GOODS_PRICE","AMT_INCOME_TOTAL","CREDIT_TO_ANNUITY_RATIO",
                               "CODE_GENDER","CREDIT_TO_GOODS_RATIO","NAME_FAMILY_STATUS","NAME_EDUCATION_TYPE","OCCUPATION_TYPE",
                               "DAYS_EMPLOYED","ANNUITY_TO_INCOME_RATIO","DAYS_BIRTH","OWN_CAR_AGE","FLAG_OWN_CAR","EMPLOYED_TO_BIRTH_RATIO"]
    utilisateur_bool = st.session_state.utilisateur_bool
    if utilisateur_bool == "Oui":
        user_id_value = int(st.session_state.user_id_value)
        loan_accepted = model_loaded.predict(df[df['SK_ID_CURR']==user_id_value][features_for_pred].values[0].reshape(1,-1))[0] == 1
        if loan_accepted != (df[df['SK_ID_CURR']==user_id_value]['TARGET'].values[0] == 1):
            st.write("### contradiction prédiction réalité")
        index = df[df['SK_ID_CURR']==user_id_value].index[0]
        shap_values_selected = shap_values_loaded[index]
        pd_series_selected = df.iloc[index][features_for_pred]
        # shap_values_selected = shap_values_selected[:,:,1]
        
        # fig = plt.subplot(211)
        # st.set_option('deprecation.showPyplotGlobalUse', True)
        
        # shap.force_plot(shap_value_created[0])
        # st.pyplot(bbox_inches='tight')
        
    else:
        creditInput = st.session_state.creditInput
        annuiteInput = st.session_state.annuiteInput

        sexe = st.session_state.sexe
        sexe = 0 if sexe == 'Femme' else 1

        goodsInput = st.session_state.goodsInput

        studies_degree = st.session_state.studies_degree
        if studies_degree=='Secondaire':
            studies_degree = 0
        elif studies_degree=='Haute études':
            studies_degree = 1
        elif studies_degree=='Haute études incomplètes':
            studies_degree = 2
        elif studies_degree=='Inférieur secondaire':
            studies_degree = 3
        elif studies_degree=='Licence':        
            studies_degree = 4 

        occupation_type = st.session_state.occupation_type
        if occupation_type=='ouvrier':
            occupation_type = 0
        elif occupation_type=='personnel basique':
            occupation_type = 1
        elif occupation_type=='comptable':
            occupation_type = 2
        elif occupation_type=='manager':
            occupation_type = 3
        elif occupation_type=='conducteur':        
            occupation_type = 4 
        elif occupation_type=='commercial':
            occupation_type = 5
        elif occupation_type=='nettoyage':
            occupation_type = 6
        elif occupation_type=='cuisine':
            occupation_type = 7
        elif occupation_type=='service privé':
            occupation_type = 8
        elif occupation_type=='médical':        
            occupation_type = 9 
        elif occupation_type=='sécurité':
            occupation_type = 10
        elif occupation_type=='haute technique':
            occupation_type = 11
        elif occupation_type=='serveur/barmen':
            occupation_type = 12
        elif occupation_type=='ouvrier peu qualifié':
            occupation_type = 13
        elif occupation_type=='immobilier':        
            occupation_type = 14 
        elif occupation_type=='secrétaire':
            occupation_type = 15
        elif occupation_type=='informaticien':
            occupation_type = 16
        elif occupation_type=='RH':
            occupation_type = 17
            

        job_start = st.session_state.job_start
        incomeInput = st.session_state.incomeInput
        daybirth = st.session_state.daybirth
        carInput = st.session_state.carInput

        family_status = st.session_state.family_status
        if family_status=='marié(e)':
            family_status = 0
        elif family_status== 'célibataire':
            family_status = 1
        elif family_status== 'mariage civil':
            family_status = 2
        elif family_status== 'séparé':
            family_status = 3
        elif family_status== 'veuf(ve)':        
            family_status = 4
        elif family_status== 'autre':
            family_status = 5

        AMT_ANNUITY = int(annuiteInput)
        AMT_CREDIT = int(creditInput)
        AMT_GOODS_PRICE = int(goodsInput)
        AMT_INCOME_TOTAL = int(incomeInput)
        CREDIT_TO_ANNUITY_RATIO = AMT_CREDIT / AMT_ANNUITY
        CODE_GENDER = int(sexe)
        CREDIT_TO_GOODS_RATIO = AMT_CREDIT / AMT_GOODS_PRICE
        NAME_FAMILY_STATUS = int(family_status)
        NAME_EDUCATION_TYPE = int(studies_degree)
        OCCUPATION_TYPE = int(occupation_type)
        # st.write(str(job_start[0:4])
        DAYS_EMPLOYED = calcul_duree_jour(job_start)
        ANNUITY_TO_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL
        DAYS_BIRTH = calcul_duree_jour(daybirth)
        OWN_CAR_AGE = 0 if int(carInput) == -1 else int(carInput)
        FLAG_OWN_CAR = 0 if int(carInput) == -1 else 1
        EMPLOYED_TO_BIRTH_RATIO = DAYS_EMPLOYED / DAYS_BIRTH
        
        new_values_columns = {"AMT_ANNUITY":AMT_ANNUITY,"AMT_CREDIT":AMT_CREDIT,"AMT_GOODS_PRICE":AMT_GOODS_PRICE,
                              "AMT_INCOME_TOTAL":AMT_INCOME_TOTAL,"CREDIT_TO_ANNUITY_RATIO":CREDIT_TO_ANNUITY_RATIO,
                              "CODE_GENDER":CODE_GENDER,"CREDIT_TO_GOODS_RATIO":CREDIT_TO_GOODS_RATIO,
                              "NAME_FAMILY_STATUS":NAME_FAMILY_STATUS,"NAME_EDUCATION_TYPE":NAME_EDUCATION_TYPE,
                              "OCCUPATION_TYPE":OCCUPATION_TYPE,"DAYS_EMPLOYED":DAYS_EMPLOYED,
                              "ANNUITY_TO_INCOME_RATIO":ANNUITY_TO_INCOME_RATIO,"DAYS_BIRTH":DAYS_BIRTH,"OWN_CAR_AGE":OWN_CAR_AGE,
                              "FLAG_OWN_CAR":FLAG_OWN_CAR,"EMPLOYED_TO_BIRTH_RATIO":EMPLOYED_TO_BIRTH_RATIO}
        
        neutral_values = pd.read_csv(neutral_features_values_filename)
        
        for key,value in new_values_columns.items():
            neutral_values[key] = value
            
        loan_accepted = model_loaded.predict(neutral_values[features_for_pred].values[0].reshape(1,-1))[0] == 1
        shap_value_created = explainer_loaded(neutral_values[features_for_pred])
        shap_value_created = shap_value_created[:,:,1]
        shap_values_selected = shap_value_created[0]
        pd_series_selected = neutral_values.iloc[0][features_for_pred]
        # print(pd_series_selected)
    
    if loan_accepted:
        st.write("## Prêt approuvé")
    else:
        st.write("## Prêt désapprouvé")
    st_shap(shap.force_plot(shap_values_selected), 200)
    
    
    st.write("### Position du prêt")
    col_in_french = ["annuité","crédit","valeur bien à acheter","revenu annuel","ratio crédit/annuité",
                     "sexe","ratio crédit/valeur bien à acheter","statu familial","niveau d'études",
                     "catégorie métier","jours employé","ratio annuité/revenu annuel","age","age voiture",
                     "possède une voiture","ratio jours employé/naissance"]
    for ind,col in enumerate(importante_features):
        saved_graph= "static/" + str(col) + "_histogram.pkl"
        with open(saved_graph, 'rb') as handle:
            graph = pickle.load(handle)
        # print([pd_series_selected[col]])
        st.write("#### Position du prêt pour : " + col_in_french[ind])
        st.plotly_chart(graph.add_trace(go.Scatter(y=[1],x=[pd_series_selected[col]])))
        handle.close()
        
    st.write("### Incidence des variables sur le model de décision")
    st.image(summary_plot)

def calcul_duree_jour(in_date):
    today_date = datetime.date.today()
    date_str = str(in_date)
    input_date = datetime.date(int(date_str[0:4]),int(date_str[5:7]),int(date_str[8:]))
    return (int((input_date - today_date).days))

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    
def density_plot(shap_value):
    feature_names = shap_value.feature_names
    shap_df = pd.DataFrame(shap_value.values, columns=feature_names)
    vals = np.abs(shap_df.values).mean(0)
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    # st.write(shap_importance.iloc[0:10])
    for col in shap_importance.iloc[0:10]['col_name']:
        st.plotly_chart(px.histogram(df,col))
    

if __name__ == "__main__":
    main()

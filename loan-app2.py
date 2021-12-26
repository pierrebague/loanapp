from flask_socketio import SocketIO
import sys
print(sys.path)
from flask import Flask, render_template, url_for, request
app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app)


from pandas import read_csv
import pandas as pd
import shap
from shap.plots._force_matplotlib import draw_additive_plot
import pickle
import datetime
import os
# import joblib


# features100 = "static/features100.csv"
model = "static/lgb_classifier_model2_02_7.txt"
# model = "static/model.lzma"
explainer = "static/explainer.pkl"
# shap_values = "static/shap_values.pkl"
shap_values = "static/shap_values_reduce.pkl"
neutral_features_values_filename = "static/neutral_value.csv"
# features_filename = "static/features.csv"
features_filename = "static/features_reduce.csv"

posts = [
    {
        'author': 'Corey Schafer',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'April 21, 2018'
    }
]

# features_file = 'static'+features
df = pd.read_csv(features_filename,low_memory=False)
columns = df.columns.values

del_features_for_pred = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index', 'level_0','Unnamed: 0']
features_for_pred = list(filter(lambda v: v not in del_features_for_pred, columns))


with open(shap_values, 'rb') as handle:
    shap_values_loaded = pickle.load(handle)
shap_values_loaded = shap_values_loaded[:,:,1]




with open(explainer, 'rb') as handle:
    explainer_loaded = pickle.load(handle)
# explainer_loaded = joblib.load(explainer)
    
with open(model, 'rb') as handle:
    model_loaded = pickle.load(handle)
# model_loaded = joblib.load(model)

# explainer = shap.Explainer(model_loaded)
# shap_values_reduce = explainer(df[features_for_pred])
# shap_values_loaded = shap_values_reduce[:,:,1]
# print(shap_values_loaded.shape)




def _force_plot_html(explainer, shap_values, ind):
    # force_plot = shap.plots.force(shap_values[ind],matplotlib=False)
    force_plot = shap.force_plot(shap_values[ind])
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return shap_html
  
def _summary_plot_html():
    summary_plot = shap.summary_plot(shap_values_loaded,df[features_for_pred],max_display=30)
    shap_summary_html = f"<head>{shap.getjs()}</head><body>{summary_plot.html()}</body>"
    return shap_summary_html
    
def _waterfall_plot_html(shap_value):
    waterfall_plot = shap.plots.waterfall(shap_value)
    shap_waterfall_html = f"<head>{shap.getjs()}</head><body>{waterfall_plot.html()}</body>"
    return shap_waterfall_html

# neutral_values = pd.read_csv(neutral_features_values_filename)
# shap_value_created = explainer_loaded(neutral_values[features_for_pred])
# shap_value_created = shap_value_created[:,:,1]
shap_plots = {}
for i in range(2): #how many plots you want
    ind = i
    # if ind == 0:
        # shap_plots[i] = _waterfall_plot_html(shap_value_created[1])
    # else:
    shap_plots[i] = _force_plot_html(explainer_loaded, shap_values_loaded, ind)



@app.route("/")
@app.route("/presentation")
def home():
    
    
    return render_template('presentation.html', posts=posts)


@app.route("/visualization")
@app.route("/visualization/", methods=["GET","POST"])
def visualization():
    # user_id_value = request.form["user_id_value"]
    ids_user=df["SK_ID_CURR"].values
    return render_template('visualization.html', title='visualization',columns=columns,ids_user=ids_user,loan_accepted=None)



    


    
@app.route("/visualization/show_loan_approbal", methods=["POST"])
def show_loan_approbal():
    user_id_value = int(request.form["user_id_value"])
    loan_accepted = df[df['SK_ID_CURR']==user_id_value]['TARGET'].values == 1
    # print(loan_accepted)
    index = df[df['SK_ID_CURR']==user_id_value].index[0]
    
    f_plot = shap.force_plot(shap_values_loaded[index])    
    f_plot_html = f"<head>{shap.getjs()}</head><body>{f_plot.html()}</body>"    
    summary_plot = os.path.join('C:/Users/pierr/Documents/OC/projet7/project7/apps/loan_app2/static/', 'summary_plot.png')
    print(index)    
    # waterfall_plot = shap.waterfall_plot(shap_values_loaded[index])#,data=neutral_values[features_for_pred].values[0].reshape(1,-1))  à ajouter pour afficher la valeur de la donnée
    waterfall_plot2 = shap.plots.waterfall(shap_values_loaded[index])
    shap_waterfall_html = f"<head>{shap.getjs()}</head><body>{waterfall_plot.html()}</body>"
    return render_template('results.html', title='results',user_id=user_id_value,loan_accepted=loan_accepted,shap_waterfall_html=shap_waterfall_html,f_plot_html=f_plot_html,summary_plot=summary_plot)

@app.route("/visualization/calcul_pret", methods=["POST"])
@app.route("/visualization/calcul_pret/", methods=["POST"])
def calcul_pret():
    
    # def _summary_plot_html():
        # summary_plot = shap.summary_plot(shap_values_loaded,df[features_for_pred],max_display=30)
        # shap_summary_html = f"<head>{shap.getjs()}</head><body>{summary_plot.html()}</body>"
        # return shap_summary_html
        
    # def _waterfall_plot_html(shap_value):
        # waterfall_plot = shap.plots.waterfall(shap_value)
        # shap_waterfall_html = f"<head>{shap.getjs()}</head><body>{waterfall_plot.html()}</body>"
        # return shap_waterfall_html
    
    
    creditInput = request.form["creditInput"]
    annuiteInput = request.form["annuiteInput"]
    sexe = request.form["sexe"]
    goodsInput = request.form["goodsInput"]
    studies_degree = request.form["studies_degree"]
    occupation_type = request.form["occupation_type"]
    job_start = request.form["job_start"]
    incomeInput = request.form["incomeInput"]
    daybirth = request.form["daybirth"]
    carInput = request.form["carInput"]
    family_status = request.form["family_status"]
    
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
    
    # col = neutral_values.columns
    # print("longueurneut :"+str(len(col)))
    # print("longueurdf :"+str(len(df.columns)))
    # dif = list(filter(lambda v: v not in col, df.columns))
    # print("difference "+ ''.join(str(e) for e in dif))
    # print("difference "+ str(len(features_for_pred)))
    # print(features_for_pred)
    # print(neutral_values)
    # print(neutral_values.shape)
    # print(neutral_values[neutral_values.columns])
    # dif = list(filter(lambda v: v not in neutral_values.columns,features_for_pred))
    # print(type(neutral_values.columns))
    # print(type(features_for_pred))
    # print('differences : {}'.format(dif))
    # print(neutral_values[features_for_pred])
    # print(neutral_values[features_for_pred].values)
    
    loan_accepted = model_loaded.predict(neutral_values[features_for_pred].values[0].reshape(1,-1))[0] == 1
    shap_value_created = explainer_loaded(neutral_values[features_for_pred])
    shap_value_created = shap_value_created[:,:,1]
    
    # shap_plots_cp = {}
    # for i in range(2): #how many plots you want
        # ind = i
        
    f_plot = shap.force_plot(shap_value_created[0])    
    f_plot_html = f"<head>{shap.getjs()}</head><body>{f_plot.html()}</body>"    
        
        
    # summary_plot = shap.summary_plot(shap_values_loaded,df[features_for_pred],max_display=30)
    # shap_summary_html = f"<head>{shap.getjs()}</head><body>{summary_plot.html()}</body>"
    
    
    waterfall_plot = shap.plots.waterfall(shap_value_created[0])#,data=neutral_values[features_for_pred].values[0].reshape(1,-1))  à ajouter pour afficher la valeur de la donnée
    shap_waterfall_html = f"<head>{shap.getjs()}</head><body>{waterfall_plot.html()}</body>"
    
    # shap_plots_cp[0] = shap_summary_html
    # shap_plots_cp[0] = shap_waterfall_html
    # shap_plots_cp[0] = _waterfall_plot_html(shap_value_created[0])
    
    summary_plot = os.path.join('C:/Users/pierr/Documents/OC/projet7/project7/apps/loan_app2/static/', 'summary_plot.png')
    return render_template('results.html', title='results',user_id="0",loan_accepted=loan_accepted,shap_waterfall_html=shap_waterfall_html,f_plot_html=f_plot_html,summary_plot=summary_plot,new_values_columns=new_values_columns)


def calcul_duree_jour(in_date):
    today_date = datetime.date.today()
    input_date = datetime.date(int(in_date[0:4]),int(in_date[5:7]),int(in_date[8:]))
    return (int((input_date - today_date).days))
    

if __name__ == '__main__':
    app.run(debug=False)
    # socketio.run(app)
    # app.run(host="0.0.0.0", port=8080)
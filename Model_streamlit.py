import streamlit as st
import xgboost as xgb
import pandas as pd
import joblib
import numpy as np
import altair as alt

st.title('Uso del modelo')

# Cargar los modelos desde los archivos
xgboost_CPC = xgb.XGBRegressor()
xgboost_CPC.load_model("model_xgboost_CPC.json")

xgboost_CPM = xgb.XGBRegressor()
xgboost_CPM.load_model("model_xgboost_CPM.json")

xgboost_CTR = xgb.XGBRegressor()
xgboost_CTR.load_model("model_xgboost_CTR.json")

rf_CPV = joblib.load("model_rf_CPV.joblib")


# Cargar el scaler desde el archivo
loaded_scaler = joblib.load('scaler_model.joblib')
# Cargar el pca desde el archivo
loaded_pca = joblib.load('pca_model.joblib')

# Cargar el scaler desde el archivo
loaded_scaler_CPV = joblib.load('scaler_model_CPV.joblib')
# Cargar el pca desde el archivo
loaded_pca_CPV = joblib.load('pca_model_CPV.joblib')

def load_data(df_in):
    df = pd.read_csv(df_in+'.csv')
    df = df.drop("Unnamed: 0", axis=1)
    return df

# Cargar los datos
df = load_data('df_histo')


variables_modelo = xgboost_CPC.feature_names_in_
all_features = ['Año','Mes', 'Objective', 'Cost', 'Country', 'Media_type', 'Traffic_source', 'Client','Format_New','Platform','Strategy','Plataforma','Campaign_Type','Ecommerce','Service_Product','Semanas_Antiguedad']
categorical_features = ['Objective', 'Country', 'Media_type', 'Traffic_source', 'Client','Format_New','Platform','Strategy','Plataforma','Campaign_Type','Ecommerce','Service_Product']



with st.sidebar:

    
    Año = st.number_input('Año de la campaña',value=2023)
    Mes = st.number_input('Mes de la campaña',value=11, max_value=12)
    Cost = st.number_input('Costo de la campaña',value=300)
    Objective = st.selectbox(    'Objetivo',    (['Purchase','Fans','Reach', 'Traffic', 'Category', 'Awareness','Product', 'Consideration',
                                                  'Conversion', 'Views','Landing Page Views', 'NoObjective', 'Discovery', 'Impressions','Clicks', 'Conversions', 'Whatsapp']))
    Country = st.selectbox(    'Country',    (['USA','Mexico', 'Chile', 'Colombia', 'Perú', 'Ecuador', 'Argentina']))
    Media_type = st.selectbox(    'Media_type',    (['Search','Social', 'Unknown', 'Display']))
    Traffic_source = st.selectbox(    'Traffic_source',    (['Google','Facebook',  'Other', 'LinkedIn']))
    Client = st.selectbox(    'Client',    (['Hughesnet', 'Braun', 'Enterprise', 'QuickQuack', 'ChefJames','OldGlory', 'AOV']))
    Format_New = st.selectbox(    'Format_New',    (['Display', 'Video']))
    Platform = st.selectbox(    'Platform',    (['Google Ads','Search','Facebook&Instagram', 'Discovery', 'Facebook', 'Performance Max','NoPlatform',  'Facebook & Instagram', 'Programmatic','Google Ads Search', 'LinkedIn','Google Ads Display', 'Google Ads  PMAX']))
    Strategy = st.selectbox(    'Strategy',    (['Consideration','Awareness', 'Conversion',  'Views', 'NoStrategy']))
    Plataforma = st.selectbox(    'Plataforma',    (['Google Ads','Meta',  'External Source', 'NoPlataforma']))
    Campaign_Type = st.selectbox(    'Campaign_Type',    (['SEARCH','PAGE_LIKES', 'DISCOVERY', 'OUTCOME_LEADS', 'CONVERSIONS','LINK_CLICKS', 'PERFORMANCE_MAX',  'OUTCOME_AWARENESS',
                                                           'REACH', 'OUTCOME_SALES', 'NoType', 'DISPLAY','OUTCOME_ENGAGEMENT']))
    Ecommerce = st.selectbox(    'Ecommerce',    (['Si','No']))
    Service_Product = st.selectbox(    'Service_Product',    (['Serv','Prod']))

    new_data = pd.DataFrame({
    'Año': [Año],
    'Mes': [Mes],
    'Objective': [Objective],
    'Cost': [Cost],
    'Country': [Country],
    'Media_type': [Media_type],
    'Traffic_source': [Traffic_source],
    'Client': [Client],
    'Format_New': [Format_New],
    'Platform': [Platform],
    'Strategy': [Strategy],
    'Plataforma': [Plataforma],
    'Campaign_Type': [Campaign_Type],
    'Ecommerce': [Ecommerce],
    'Service_Product': [Service_Product],
    })
    
    
    # Preprocesamiento de variables categóricas
    X = pd.get_dummies(new_data, columns=categorical_features)
    
    # Asegurarte de que 'new_data_encoded' tenga las mismas columnas que se utilizaron durante el entrenamiento
    for col in variables_modelo:
        if col not in X.columns:
            X[col] = False  # Agregar la columna faltante con valores predeterminados si es necesario

    
    X_Scaled = loaded_scaler.transform(X[['Año','Mes','Cost']])
    X_pca = loaded_pca.transform(X_Scaled)
    X_pca = pd.DataFrame(X_pca)
    X['X_pca_0'] = X_pca[0]
    X['X_pca_1'] = X_pca[1]

    # Me aseguro de que mi nuevo dato tiene las mismas variables y el mismo orden que la data con la que fue entrenado el modelo
    X = X[variables_modelo]
    X.columns = X.columns.astype(str)

def prediccion_modelo(modelo,X):
    return modelo.predict(X)

bin_density = st.slider('Bins', min_value=250, max_value=350, step=5, value=300)

st.button("Reset", type="primary")
if st.button('Hacer predicción'):
    pred_CPC = prediccion_modelo(xgboost_CPC,X)[0]
    pred_CPM = prediccion_modelo(xgboost_CPM,X)[0] 
    pred_CTR = prediccion_modelo(xgboost_CTR,X)[0]

    #CPV
    for col in rf_CPV.feature_names_in_:
        if col not in X.columns:
            X[col] = False  # Agregar la columna faltante con valores predeterminados si es necesario
            print(col)

    
    
    X_Scaled = loaded_scaler_CPV.transform(X[['Año','Mes','Cost']])
    X_pca = loaded_pca_CPV.transform(X_Scaled)
    X_pca = pd.DataFrame(X_pca)
    X['X_pca_0'] = X_pca[0]
    X['X_pca_1'] = X_pca[1]
    
    X = X[rf_CPV.feature_names_in_]
    X.columns = [str(i) for i in X.columns]

    
    
    pred_CPV = prediccion_modelo(rf_CPV,X)[0]

    def histo(df,metrica,valor,bins=bin_density):
        chart = alt.Chart(df).mark_bar(
        opacity=0.3,
        binSpacing=0
    ).encode(
        alt.X(metrica+':Q').bin(maxbins=bin_density),
        alt.Y('count()').stack(None),            
    ).properties(
            width=1000,
            height=600
        ).interactive()

        linea_valor = alt.Chart(pd.DataFrame({'valor_linea': [valor]})).mark_rule(color='red').encode(
    x='valor_linea:Q',
    size=alt.value(2)  # Grosor de la línea
)
        return chart + linea_valor
    
    st.write('CPC')
    st.write(round(pred_CPC,3))
    st.altair_chart(histo(df,'CPC',pred_CPC), use_container_width=False, theme=None)
    
    st.write('CPM')
    st.write(round(pred_CPM,3))
    st.altair_chart(histo(df,'CPM',pred_CPM), use_container_width=False, theme=None)
    
    st.write('CTR')
    st.write(round(pred_CTR,3))
    st.altair_chart(histo(df,'CTR',pred_CTR), use_container_width=False, theme=None)
    
    st.write('CPV')
    st.write(round(pred_CPV,3))
    st.altair_chart(histo(df,'CPV',pred_CPV,bins=bin_density*5), use_container_width=False, theme=None)
    
else:
    st.write('Prepara tu predicción')
    

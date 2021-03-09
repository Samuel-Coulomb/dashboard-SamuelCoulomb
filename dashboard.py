# Dashboard prêt à dépenser Samuel Coulomb


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import json
import joblib
import requests
import pickle
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def main():
    n_rows=10000
    data = pd.read_csv("/Users/samuelcoulomb/Desktop/data.csv", nrows = 10000)
    data = data.fillna(data.median())
    train = pd.read_csv("/Users/samuelcoulomb/Desktop/train.csv", nrows = 10000)
    train.index = data['SK_ID_CURR']
    data.index = data['SK_ID_CURR']
    data['score'] = 100 - data['score']
    data['Age'] = (data['DAYS_BIRTH']/-365).astype(int)
    data["Année d'ancienneté"] = round((data['DAYS_EMPLOYED']/-365), 0)
    cols = ['Age', 'CODE_GENDER', "Année d'ancienneté", 'NAME_FAMILY_STATUS','CNT_CHILDREN', 'NAME_EDUCATION_TYPE', 'AMT_CREDIT', 'AMT_ANNUITY', 'FLAG_OWN_CAR', 'score']
    clients = data[cols]
    clients.columns = ['Age', 'Sexe', "Année d'ancienneté", 'Statut familial', "Nb d'enfants", "Type d'éducation", "Montant du crédit", 'Montant annuité', 'Voiture', 'score']
    clients["Année d'ancienneté"] = clients["Année d'ancienneté"].astype(int)
    
    
    
    
    
    # Infos du client
    st.title('Dashboard : prêt à dépenser')
    client_id = st.sidebar.selectbox('Identifiant du client :', data.index)
    st.sidebar.markdown('Information de ce client :')
    st.sidebar.table(clients.loc[[client_id], ['Age', 'Sexe', 'Statut familial', "Nb d'enfants", "Type d'éducation", "Année d'ancienneté", 'Voiture' ]].T)
    
    
    
    
    
    
    # 50 clients simlilaires
    tree = KDTree(train)
    id_client_sim = tree.query(train[train.index == client_id], k=50)[1][0]
    voisins = data.iloc[id_client_sim]
    score_voisins = voisins['score'].mean()
    
    
    score_client = data.loc[client_id]['score']
    
    st.header("Score du client")
    if score_client > 50:
        st.markdown("Le client **{}** est insolvable".format(client_id))
    else :
        st.markdown("Le client **{}** est solvable".format(client_id))
    fig = go.Figure(go.Indicator(
        mode = "gauge+delta+number",
        value = score_client,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [None, 100]},
		       'steps': [
			       {'range': [0, 20], 'color': "#2d7e43"},
			       {'range': [20, 40], 'color': "#97ba38"},
			       {'range': [40, 60], 'color': "#f0ca0d"},
			       {'range': [60, 80], 'color': "#d57b1a"},
                   {'range': [80, 100], 'color': "#c53419"}
		       ],
		       'threshold': {
			       'line': {'color': "black", 'width': 10},
			       'thickness': 0.8,
			       'value': score_client},

		       'bar': {'color': "black", 'thickness': 0.2},
		       },
        title = {'text': "Speed"},
        delta={'reference': score_voisins,
		       'increasing': {'color': 'red'},
		       'decreasing': {'color': 'green'}}))

    st.plotly_chart(fig)

    st.markdown('Score du client : **{0:.1f}%**'.format(score_client))
    st.markdown('Score des 50 clients similaires : **{0:.1f}%**'.format(score_voisins))
    
    
    
    
    
    
    
    # Concernant le crédit demandé
    
    st.header("Informations concernant le crédit demandé")
    
    y_clients = data[['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']].mean().values
    
    y_client = data.loc[client_id][['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']].values
    
    fig = go.Figure(data=[go.Bar(name='Client sélectionné', x=['Montant des revenues','Montant du crédit', 'Montant annuité'], y=y_client), go.Bar(name='Moyenne des clients', x=['Montant des revenues','Montant du crédit', 'Montant annuité'], y=y_clients)])
    
    st.plotly_chart(fig, use_countainer_width=True)
    
    
    
    
    
    
    
    # Interprétation du score
    
    st.header("Interprétation du score du client")
    
    feature = st.selectbox('Variable expliquant le score', ['EXT_SOURCE_3', 'OBS_60_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'EXT_SOURCE_2', 'OBS_30_CNT_SOCIAL_CIRCLE'])
    
    
    df = data[[feature, 'score']]
    x = [df[df['score']<50][feature].mean()]
    x.append(df[df['score']>50][feature].mean())
    x.append(voisins[feature].mean())
    x.append(data.loc[client_id][feature])


    fig2 = go.Figure(data=[go.Bar(x=x,y=['Moyenne des clients solvables', 'Moyenne des clients insolvables', 'Moyenne des 50 clients similaires ', 'Client sélectionné '], marker_color=['green', 'red', 'grey', 'blue'], orientation='h')])


    st.plotly_chart(fig2)
    
    
    
    
    
    # Interprétation de ce client
    
    #st.header("Pourquoi ce client est insolvable ?")
    
    #from lime import lime_tabular
    
    #explainer = lime_tabular.LimeTabularExplainer(training_data = np.array(train), mode = "classification", feature_names = data.drop(columns=['TARGET']).columns)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()


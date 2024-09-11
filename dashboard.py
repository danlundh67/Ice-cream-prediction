import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import math
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
def read_data():
    data_path = Path(__file__).parents[0] 
    df = pd.read_csv(data_path / "Data" / "IceCreamData.csv")
    return df

def layout():

    df = read_data()
    st.title("Icecream Revenue Predictor")

    st.write("This is prediction of the Icecream revenue")

    X=df[['Temperature']]
    y=df['Revenue']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
    regressor = RandomForestRegressor(n_estimators=10, random_state=0,oob_score=True)

    regressor.fit(X, y)
   
    #oob_score = regressor.oob_score_
    predictions = regressor.predict(X)
    mse = mean_squared_error(y, predictions)



    st.write("The root mean squared error score on dataset is ", math.sqrt(mse))

    number = st.number_input("Insert a number", step=0.5)
    st.write("The current temperature is: ", number)

    myval=[[number]]

    pred=regressor.predict(myval)
    
    st.write("Revenue prediction is: ", pred[0])

    st.header("Raw data")
    st.write("This shows the raw data")
    st.dataframe(df)
    read_css()

def read_css():
    css_path = Path(__file__).parent / "style.css"

    with open(css_path) as css:
        st.markdown(
            f"<style>{css.read()}</style>",
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    # print(read_data())
    layout()

import pandas as pd
import streamlit as st
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima_model import ARIMA

def forecasting(date1,date2):
    data=pd.read_csv('lacity.csv',index_col=['Date'], parse_dates=True)
    df=pd.DataFrame(data,columns=['Combined Users'])
    col1,col2= st.beta_columns(2)
    df=df.dropna()
    df=df.iloc[::-1]
    col1.subheader("The Graphical View of given data")
    col1.line_chart(df,200,400,True)
    col2.subheader("Data as Follow")
    col2.write(df)
    from datetime import date
    if(date1!=date.today() and date2!=date.today()):
        import datetime
        dates=[]
        d=str(date1)
        d1=d.split('-')
        start_date = datetime.date(int(d1[0]),int(d1[1]),int(d1[2]))
        d=date2-date1
        num=str(d)
        num=num.split(" ")
        n=int(num[0])
        size=df.size
        beststep=auto_arima(df,suppress_warnings=True)
        ord=beststep.to_dict()
        model=ARIMA(df,order=ord['order'])
        result=model.fit()
        x=result.predict(start=date1, end=date2, exog=None, typ='levels')
        for day in range(n):
             a_date = (start_date + datetime.timedelta(days = day)).isoformat()
             dates.append(a_date)
        dataa=[]
        for i in range(n):
            y=[dates[i],int(x[i])]
            dataa.append(y)
        df1=pd.DataFrame(dataa,columns=['Date','Combined Users'])
        df1['Date']=pd.to_datetime(df1['Date'])
        df1=df1.set_index(['Date'])
        c1,c2=st.beta_columns(2)
        c2.subheader("Predicted Data as Follow")
        c2.dataframe(df1)
        c1.subheader('Graphical view of Predicted Value ')
        c1.line_chart(x,0,0,True)
        st.success('Done')
    else:
        st.write('Please Select the dates')

st.set_page_config(
page_title="Forecasting App",
page_icon="https://images.squarespace-cdn.com/content/v1/5572b7b4e4b0a20071d407d4/1487152555739-9KRQHY2BK54CIYBMJ8XF/ke17ZwdGBToddI8pDm48kCMWMBFcqQftRz-JqZZoIB5Zw-zPPgdn4jUwVcJE1ZvWEtT5uBSRWt4vQZAgTJucoTqqXjS3CfNDSuuf31e0tVFI99ncPZu898P4WAmVYNBp8mgB1qWbp5RirnU_Xvq-XCb8BodarTVrzIWCp72ioWw/Extensive+Forecasting",
layout="wide" )
st.header("Web Traffic Forecasting")
st.subheader("Pick Future Date for Forecasting/Prediction")
c1,c2=st.beta_columns(2)
d1=c1.date_input('Start Date')
d2=c2.date_input('End Date')
if st.button('Submit'):
    forecasting(d1,d2)
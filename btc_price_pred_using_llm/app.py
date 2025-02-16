from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pandas_ta as ta
import streamlit as st
import requests as r
import pandas as pd
import numpy as np
import re
import os




load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-specdec",
    temperature=1.5,
    groq_api_key=API_KEY
)



def get_cryptocompare_data(fsym, tsym, limit):
    url = f'https://min-api.cryptocompare.com/data/v2/histoday'
    params = {
        'fsym': fsym,
        'tsym': tsym,
        'limit': limit
    }
    response = r.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data['Data']['Data'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    return df




crypto = st.selectbox("Select Your Desired Crypto", ["BTC"])

df = get_cryptocompare_data(crypto, 'USDT', 30)

st.dataframe(df.tail())

df.drop(df.tail(1).index,inplace=True)
# Some technical analysis indicators that helps improving the model's accuracy
df['rsi'] = ta.rsi(df['close'], length=14)

df['ema10'] = ta.ema(df['close'], length=10)

df["sma10"] = ta.sma(df['close'], length=10)

macd = ta.macd(df['close'], fast=10, slow=20, signal=9)
df['macd'] = macd['MACD_10_20_9']
df['macd_signal'] = macd['MACDs_10_20_9']

# bbands = ta.bbands(df['close'], length=20, std=2)
# df['bb_upper'] = bbands['BBU_20_2.0']
# df['bb_middle'] = bbands['BBM_20_2.0']
# df['bb_lower'] = bbands['BBL_20_2.0']

df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

# Taking the close value of the next row(day)

df.dropna(inplace=True)
df.reset_index(inplace=True)
df.drop(["conversionType", "conversionSymbol"], axis=1, inplace=True)


button =  st.button("Predict Price")

if button:

    with st.spinner("Please Wait..."):
        prompt = PromptTemplate.from_template(
        """
        You are a financial AI assistant specialized in cryptocurrency price prediction.
        Your task is to predict Bitcoin's closing price for the next day based on the past 30 days of historical data, which includes technical analysis indicators.  

        ### Input Data Format:
        The historical data is provided in a pandas DataFrame format with the following columns:
        - `time`: The date of the recorded data (YYYY-MM-DD).
        - `high`: The highest price of Bitcoin during the day.
        - `low`: The lowest price of Bitcoin during the day.
        - `open`: The opening price of Bitcoin.
        - `close`: The closing price of Bitcoin.
        - `volumefrom`: The total number of Bitcoins traded.
        - `volumeto`: The total traded value in the market.
        - `rsi`: 14-day Relative Strength Index.
        - `ema_10`: 10-day Exponential Moving Average.
        - `macd`: MACD (Moving Average Convergence Divergence) value.
        - `macd_signal`: Signal line of the MACD.
        - `atr`: 14-day Average True Range.
        
        ### Task:
        Analyze the given data and predict the closing price (`close`) for the next day based on the trends, patterns, and indicators.  
        
        ### Output Format:
        Provide your prediction in JSON format. Only return your predictions(NO PEAMBLE!):
        "predicted_close": <your_predicted_value>,
        "confidence_score": <confidence_between_0_and_1>


        ### Pandas DataFrame:
        {df}
        """
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        res = chain.invoke({"df": df})

        output = res["text"].split("```")[1]
        numbers = re.findall(r'\d+\.?\d*', output)
        st.header(f"Predicted Price: {numbers[0]}")
        st.header(f"Confidence: {numbers[1]}")
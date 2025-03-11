# **📈Stock Prediction in Python**

***A machine learning-powered application for predicting stock prices using LSTMs built with Python, VS Code and Streamlit.***

<a href="https://stockpredictionpython.streamlit.app/" target="_blank">**Try It Out!**</a>


---

## **🐍Python Libraries Used**
Different Python libraries are used to help develop this application, each with their own purpose.
 - **TensorFlow** - *Training the model*
 - **Pandas & NumPy** - *Data processing*
 - **Scikit-Learn** - *Data manipulation*
 - **Matplotlib** - *Data visualization*
 - **Streamlit** - *Web based dashboard*
 - **yfinance** - *Yahoo finance API for fetching stock data*

---

## **❓How Does it Work?**
 - Stock prices over the years are gathered using **yfinance** and normalized (scaled between 0 and 1) to help the model learn patterns easier.
 - The model is trained over 50 days of information and tested on 1 day, this is to ensure the model is highly accurate while still having data to be tested on.
 - This algorithm repeats until all the data has been used and the results are plotted on a graph in comparison to the actual prices.
 - Because of this training, the model is only able to predict up to 1 day in the future at the moment.

---

## **🚀Future Improvements**
This application can be improved further in the future with help of others, but here are a few ideas that I will be working on:
 - ~~**Error Handling** - currently the user can type in any string, causing the program to crash, adding this will stop the program crashing~~
 - **Improved Model Accuracy** - currently the model is quite accurate, but can be increased more with methods such as hyperparameter tuning
 - **Sentiment Analysis** - the movement of stocks is widely impacted by the mood of investors, adding this feature can increase accuracy even more.

---

**Any ideas and contributions would be greatly appreciated at this time :)**

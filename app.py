#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import joblib


# In[2]:


# Path to your HDF5 model file
model_path = "C:/Users/Mo/Graduation _Project/Deployment/Cloud_model.pkl"

# Load the model
with open(model_path,'rb') as f:
        
    loaded_model = joblib.load(f)


# In[3]:


from flask import Flask, request, jsonify

from flask_restful import Resource, Api

from flask_cors import CORS


# app= Flask(__name__)
# 
# 
# CORS(app)
# 
# 
# api=Api(app)
# 
# 
# @app.route('/')
# def home():
#     return "Run successfully....."
# 
# @app.route('/pridectcsv', methods=['POST'] )
# def predict():
#     
#     import pandas as pd
#     data_file = pd.read_csv(request.files.get("file"))
#     data_file.head()
#     #data_file =  np.fromstring(data_file, np.uint8)
#     #data_csv = pd.read_csv(data_file)
#     from Functions_2 import pre_process_data
#     data = pre_process_data(data_file)
#     from Functions_2 import remove_duplicates
#     data = remove_duplicates(data)
#     from Functions_2 import adv_pre_process_data
#     data = adv_pre_process_data(data)
#     from Functions_2 import scaling_data
#     data_input_model = scaling_data(data)
#     # calculate the loss on the test set
#     data_pred = loaded_model.predict(data_input_model)
#     data_pred = data_pred.reshape(data_pred.shape[0], data_pred.shape[2])
#     print(data_pred)
#     column_name = list(data_file.columns)
#     print(column_name)
#     data_pred = pd.DataFrame(data_pred,columns=data.columns)
#     data_pred.index = data.index
#     print(data_pred)
#     scored = pd.DataFrame(index=data.index)
#     data_test = data_input_model.reshape(data_input_model.shape[0], data_input_model.shape[2])
#     scored['Loss_mae'] = np.mean(np.abs(data_pred-data_test), axis = 1)
#     scored['Threshold'] = 0.5
#     scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
#     print(scored)
# 
# 
#     return scored.to_csv()
# if __name__ =='__main__':
#     app.run()
# 

# In[4]:


"يارب كرمك"


# In[ ]:


app= Flask(__name__)


CORS(app)


api=Api(app)


@app.route('/')
def home():
    return "Run successfully....."

@app.route('/pridectcsv', methods=['POST'] )
def predict():
    
    import pandas as pd
    data_json = request.json
    #print(data_json)
    data_file = pd.DataFrame(data=data_json)
    
    #print(data_file.columns)

    from Functions_2 import pre_process_data
    data = pre_process_data(data_file)
    from Functions_2 import remove_duplicates
    data = remove_duplicates(data)
    from Functions_2 import adv_pre_process_data
    data = adv_pre_process_data(data)
    from Functions_2 import scaling_data
    data_input_model = scaling_data(data)
    # calculate the loss on the test set
    data_pred = loaded_model.predict(data_input_model)
    data_pred = data_pred.reshape(data_pred.shape[0], data_pred.shape[2])
    #print(data_pred)
    column_name = list(data_file.columns)
    #print(column_name)
    data_pred = pd.DataFrame(data_pred,columns=data.columns)
    data_pred.index = data.index
    #print(data_pred)
    scored = pd.DataFrame(index=data.index)
    data_test = data_input_model.reshape(data_input_model.shape[0], data_input_model.shape[2])
    scored['Loss_mae'] = np.mean(np.abs(data_pred-data_test), axis = 1)
    scored['Threshold'] = 0.5
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    #print(scored)


    return scored.to_json()
if __name__ =='__main__':
    app.run()


# In[ ]:





# In[ ]:





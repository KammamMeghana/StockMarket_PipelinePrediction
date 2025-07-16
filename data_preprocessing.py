#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import date
import os
import pandas as pd
from azureml.core import Workspace, Datastore, Dataset

# Print the run date
print("Date of pipeline ML code run is:", date.today())


# In[2]:


# Set up the Azure ML workspace
print("Setting up the ML workspace")
ws = Workspace(subscription_id='6bca6fe6-9cc6-48db-a6ab-0f2f286cc348',
               resource_group='stockmarket',
               workspace_name='MeghanaWorkspace')

print(ws.name, ws.resource_group, ws.location, ws._subscription_id, sep='\n')


# In[3]:


# Configure the blob datastore
datastore = Datastore.get(ws, 'blobconnection')
print('Setting up the blobstore for reading input files')


# In[4]:


# Load the dataset
df = Dataset.Tabular.from_delimited_files(path=[(datastore, "HEAR.csv")]).to_pandas_dataframe()


# In[5]:


# Preprocess the dataset
df = df.iloc[:500]
df = df.copy()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

print("Dataset after preprocessing:")
print(df.head())


# In[6]:


# Save the preprocessed data
output_path = "./preprocessed_data.csv"
df.to_csv(output_path, index=True)
print(f"Preprocessed data saved to {output_path}")


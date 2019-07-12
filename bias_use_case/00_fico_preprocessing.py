
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer


# In[2]:


df = pd.read_csv("../data/heloc_dataset_v1.csv")


# In[3]:


df.head()


# In[4]:


categorical_columns = ["MaxDelqEver", "MaxDelq2PublicRecLast12M"]
continuous_columns = [c for c in df.columns if c not in ["RiskPerformance"] + categorical_columns]


# # Remove Rows with No Data
# * -9, -8 -7 indicates missing values

# In[5]:


df = df.replace([-9], np.nan)


# In[6]:


# Remove all -9 

df = df.dropna()


# In[7]:


df = df.replace([-8, -7], np.nan)


# In[8]:


# Impute -8 and -7 Missing Values

imputer = Imputer(strategy = 'median')
imputed_df = imputer.fit_transform(df[continuous_columns].values)


# # One Hot Encode Categorical

# In[9]:


categorical_df = pd.get_dummies(df[categorical_columns], columns=categorical_columns)


# # Normalize

# In[10]:


target_df = df["RiskPerformance"]
target = np.array(target_df == "Good")



# In[11]:


scaler = StandardScaler()
continuous_features_scaled = scaler.fit_transform(imputed_df)


# # Save as CSV

# In[12]:


continuous_df = pd.DataFrame(continuous_features_scaled, columns=continuous_columns, index = df.index)


# In[13]:


categorical_df.shape


# In[14]:


categorical_df.index


# In[15]:


preprocessed_df = pd.concat([categorical_df, continuous_df], axis=1, join_axes=[categorical_df.index])


# In[16]:


preprocessed_df["target"] = target


# In[17]:


preprocessed_df.to_csv("preprocessed-fico.csv")



# coding: utf-8

import pickle
import pandas as pd
import numpy as np

#import keras
from keras.models import Sequential, load_model
from keras.layers import Dense

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from deepexplain.tensorflow import DeepExplain
from keras import backend as K
from keras.models import Model
import tensorflow as tf

model_name = 'bias-data-model.h5'


# Read in data
df = pd.read_csv('preprocessed-fico-biased.csv', index_col='Unnamed: 0')
features = [x for x in df.columns.tolist() if x!='target']

X = df[features].values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print( X_train.shape )

# load in baseline sample for use with DeepLift and IntGrad
with open('baseline.pkl', 'rb') as f:
    baseline = pickle.load(f)


# # Generate DeepLIFT Local Attributions
BATCH_SIZE = 100
EPOCHS = 20


with DeepExplain(session=K.get_session()) as de:
    # load
    model = load_model(model_name)
    score = model.evaluate(X_test, y_test, verbose=0)

    # explain
    input_tensor = model.layers[0].input
    fModel = Model(inputs=input_tensor, outputs = model.layers[-1].output)
    target_tensor = fModel(input_tensor)

    xs = X_train
    ys = y_train

    baseline = baseline.squeeze()
    deeplift_attributions = de.explain('deeplift', target_tensor * ys, input_tensor, xs, baseline=baseline)
#print( deeplift_attributions )


# Generate Integrated Gradients Local Attributions
with DeepExplain(session=K.get_session()) as de:
    # load
    model = load_model(model_name)

    # explain
    input_tensor = model.layers[0].input
    fModel = Model(inputs=input_tensor, outputs = model.layers[-1].output)
    target_tensor = fModel(input_tensor)

    xs = X_train
    ys = y_train

    intgrad_attributions = de.explain('intgrad', target_tensor * ys, input_tensor, xs, baseline=baseline)
#print( intgrad_attributions)


# Save Local Attributions
deeplift_df = pd.DataFrame(deeplift_attributions, columns = features)
intgrad_df = pd.DataFrame(intgrad_attributions, columns = features)


print('Deeplift sample explanations - ')
print(deeplift_df.head())

print('IntGrad sample explanations - ')
print(intgrad_df.head())

deeplift_df.to_csv("bias_deeplift_attributions-50-50-basline.csv")
intgrad_df.to_csv("bias_intgrad_attributions-50-50-baseline.csv")

#save off the first 1k samples to use with GAM

deeplift_df.head(n=1000).to_csv('bias_deeplift_attributions-50-50-basline-1k-samples.csv')
intgrad_df.head(n=1000).to_csv('bias_intgrad_df_attributions-50-50-basline-1k-samples.csv')


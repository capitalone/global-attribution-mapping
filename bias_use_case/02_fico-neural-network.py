
# coding: utf-8


import pickle
import pandas as pd
import numpy as np

#import keras
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


model_name = 'bias-data-model.h5'

# Read in data
df = pd.read_csv('preprocessed-fico-biased.csv', index_col='Unnamed: 0')

features = [x for x in df.columns.tolist() if x!='target']
X = df[features].values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print( X_train.shape )


# Define Neural Network
model = Sequential()
model.add(Dense(X_train.shape[1] // 2,input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train
batch_size = 100
epochs = 20
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Model score - ', score)
model.save(model_name)


baseline_candidates = []
#sess = tf.Session()

with tf.Session() as sess:
    K.set_session(sess)
    model = load_model('biased-data-model.h5')

    for x in X_train:
        predicted = model.predict(np.array([x]))
        if 0.499 < predicted < 0.501:
            print(x)
            print(predicted)
            baseline_candidates.append(x)

# # Identify 50-50 baseline - find a sample that is roughly on the decision boundary somewhere
# we only pick 1 baseline
# first element produces .5002 output
baseline = baseline_candidates[0]
baseline = baseline.reshape(1,-1)
print(baseline)
print(baseline.shape)


with open('baseline.pkl', 'wb') as f:
    pickle.dump(baseline, f)


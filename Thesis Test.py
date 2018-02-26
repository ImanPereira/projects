import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics
from keras.utils import plot_model
from keras import initializers
import math
from sklearn.preprocessing import StandardScaler
#####
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
########## IMPORT DATA
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

summer = pd.read_csv('C:/Iman/route_winter.csv',sep=';')
winter = pd.read_csv('C:/Iman/route_summer2.csv', sep=';')

route_info = winter.append(summer, ignore_index = True)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
########## DATA PROCESS
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# calculate the actual travel time for each trip

start_date = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f') for x in route_info.start_time]
end_date =[datetime.strptime(y, '%Y-%m-%d %H:%M:%S.%f') for y in route_info.end_time]
route_info['time'] = [(x-y).total_seconds() for x,y in zip(end_date,start_date)]

# generate season(1 means winter, 0 means summer)

route_info['season'] = [x.month for x in start_date]
route_info['season'] = [1 if x>10 or x<5 else 0 for x in route_info.season]


# convert case type to binary value

route_info['vehicle_type'] = [1 if x == 30 else 0 for x in route_info.case_type_id]

# distribution analysis



#fig, ax = plt.subplots(1,2)
#ax[0]=sns.distplot(route_info['time'])
#ax[1]=sns.distplot(route_info['route_length'])
#ax[0].hist(route_info.time, 20, facecolor ='blue', alpha=0.5, ec ='black', label = 'Time')
#ax[1].hist(route_info.route_length, 1000, facecolor ='green', alpha=0.5, ec ='black', label = 'Distance')
#ax[0].set_xlabel('Time')
#ax[1].set_xlabel('Distance')
#ax[0].set_xlim([0,6000])
#ax[1].set_xlim([0,50000])
#plt.show()

# remove negative value in travel time in data
route_info = route_info[route_info['time']>0]


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########  PARAMETERS SETTING
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# the ratio beween test data and training data in size
Test_size=0.25

# Hyper-parameters setting
NO_output = 1
NO_input = 3
neurons_first_layer = math.ceil(((NO_output+2)*route_info.season.size*(1-Test_size))**0.5+2*(route_info.season.size*(1-Test_size)/(NO_output+2))**0.5)
neurons_second_layer = math.ceil(NO_output*(route_info.season.size*(1-Test_size)/(NO_output+2))**0.5)
#neurons_first_layer = 1000
#neurons_second_layer = 500
#neurons_third_layer = 200
NO_neurons = [neurons_first_layer,neurons_second_layer]
 

# activation functions used in hidden layers
activation_function = ['relu','relu']
# weight init
weight_init = ['he_normal','he_normal']
#bias init
bias_init=['zero','zero']

# number of Epochs
Epochs = 20

# batch size
Batch_size = 5

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#### DATA SPLIT
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Specify the data for input
X=route_info.ix[:,5:10]
X=X.drop('geom', axis=1)
X=X.drop('time', axis=1)

# Specify the target labels and flatten the array 
y=np.ravel(route_info.time)
# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Test_size, random_state=42)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
### DATA NORMALIZATION
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#Define the scaler
scaler = StandardScaler().fit(X_train)
#scale the training set
x_train_scale = scaler.transform(X_train)
#scale the test set
x_test_scale = scaler.transform(X_test)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
### MODEL BUILDING
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#start 
model = Sequential()

# first hidden layer
model.add(Dense(NO_neurons[0],
                activation=activation_function[0],
                use_bias =True,
                kernel_initializer = weight_init[0],
                bias_initializer = bias_init[0],
                input_shape=(NO_input,)))
# second hidden layer
model.add(Dense(NO_neurons[1],
                activation=activation_function[1],
                use_bias =True,
                kernel_initializer = weight_init[1],
                bias_initializer = bias_init[1]))

#third hidden layer

'''
model.add(Dense(NO_neurons[2],
                activation=activation_function[2],
                use_bias =True,
                kernel_initializer = weight_init[2],
                bias_initializer = bias_init[2]))
                '''

# output layer
model.add(Dense(NO_output))
          

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
### MODEL TRAINING
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#from graphviz import Digraph
#plot_model(model, show_shapes = True, to_file='model.png')

model.compile(loss='mean_absolute_error',
              optimizer='rmsprop')
                   
tbCallBack = keras.callbacks.TensorBoard(log_dir='C:/Iman/Python/Graph', histogram_freq=0,  
          write_graph=True, write_images=True)

history = model.fit(x_train_scale, y_train, epochs=Epochs, batch_size=Batch_size, verbose=0, callbacks =[tbCallBack])


# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
########## MODEL EVULATION
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

y_pred_train = model.predict(x_train_scale)
y_pred = model.predict(x_test_scale)
# absolute mean error
AMR_test = sklearn.metrics.mean_absolute_error(y_test, y_pred, sample_weight = None, multioutput='uniform_average')
AMR_train= sklearn.metrics.mean_absolute_error(y_train, y_pred_train, sample_weight = None, multioutput='uniform_average')
print(AMR_train)
print(AMR_test)

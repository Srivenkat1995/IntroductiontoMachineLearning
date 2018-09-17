
# coding: utf-8

# ## Logic Based FizzBuzz Function [Software 1.0]

import pandas as pd
#########################################################################################################################
# Understanding of Python Data Analysis Library(Pandas)

# 1. Pandas are used to load the data file as a pandas data frame and analyze data.
# 2. The most important function is that it takes data from a .csv or .tsv or SQL database file 
#    and creates rows and columns called as dataframes.
# 3. These dataframes are almost similar to a table in a excel spreadsheet or any other statistical software.  
########################################################################################################################


def fizzbuzz(n):

    #     # Logic Explanation
     if n % 3 == 0 and n % 5 == 0:
         return 'FizzBuzz'
     elif n % 3 == 0:
         return 'Fizz'
     elif n % 5 == 0:
         return 'Buzz'
     else:
         return 'Other'

#########################################################################################################################
# The above is the Software 1.0 code for the fizzbuzz program. The idea here is to print
#                   1. "Fizzbuzz" ----> If the no is multiple of both 5 and 3.
#                   2. "Fizz"     ----> If the no is multiple of only 3.
#                   3. "Buzz"     ----> If the no is multiple of only 5.
#                   4. "Other"    ----> If the no is neither divisible by 5 nor 3.
# The idea here is to use the modulo operator and if...else statements to come up with the correct solution.
#  
# Example of an iteration of the function "fizzbuzz(n)" when n is 8,
#           
#               if (n%3 == 0 and n%5 == 0) --- > false, so
#               elif(n%3 == 0) ---> false, so
#               elif(n%5 == 0) ---> false, so
#               the function returns "Other"
# This is how it works for each value of n.
########################################################################################################################

###########################################################################################################################
# The below is the Software 2.0 code aka the machine learning approach.
# Definition of a machine learning problem:
#                                           A computer program is said to learn from experience E with respect to some class
# of tasks T and performance measure P, if its performance at tasks in T , as measured by P , improves with experience E
# 
#    Example : Fizzbuzz 
# 
#    E = Experience of playing the fizzbuzz game
#    T = The task of correctly identifying the correct statement to be told i.e "Fizz","Buzz","Fizzbuzz" and "other"
#    P = The probablity that the program will identify the correct statement to be told for the number 

# Machine Learning Problems are classified into Supervised and Unsupervised Learning.
# Supervised Learning: 
#                     In Supervised learning, we are given a data set and already know what the correct answer should look
# like. So the problem of fizzbuzz is a supervised learning problem. Because, we know if the no is multiple of 3 or 5 or both.
#   
###########################################################################################################################

# ## Create Training and Testing Datasets in CSV Format

# In[1]:
############################################################################################################################
# In a machine Learning problem involving the datasets, the algorithm usually works in two stages

#                           A. Training.
#                           B. Testing.

# In an ideal situation, the data split between the training and test is 80% and 20%, and this holds the key to the output of
# the model. We face two issues,
# 
# 1. We overfit the model (Overfitting)
# 2. We underfit the model (Underfitting)
#   
# We should train the model so that we don't facy any of the problems. Because, these issues lead to a model that has either
# low accuracy or is ungeneralized.

############################################################################################################################

def createInputCSV(start,end,filename):
    
    # Why list in Python?
    inputData   = []
    outputData  = []
    
    # Why do we need training Data?
    ################################################################################################################################    
    # In general, machines are much faster at processing and storing knowledge compared to humans. But, how can we leverage their 
    # speed to create intelligent machines ? The answer is training data.

    # Algorithms learn from data. They find relationships, develop understanding , make decisions and evaluate their performance from 
    # the training data they are given. 

    # Training data: Is a labeled data used to train your machine learning algorithm and increase its accuracy.

    # The better the training set, better the machine learning model.
    #################################################################################################################################

    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))
 
    # Why Dataframe?
    ######################################################################################################################################

    # Dataframe is one of the data structures which pandas(Package) provides. It makes manipulating your data easily, from selecting 
    # or replacing columns and indices to reshaping your data.
    
    #######################################################################################################################################  
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData
    
    
    # Writing to csv
    ######################################################################################################################################
    # With respect to pandas, we can store the data in the following ways,

    # 1. Convert a python's list, dictionary or numpy array to a pandas data frame
    # 2. Open a local file using pandas, usually a csv file but could also be a delimited text file, excel etc
    # 3. Open a remote file or database like a csv or a JSON on a website through a URL or read from a table or dataset.

    # In this Fizzbuzz example, we are trying to open a local .csv file 
    
    #######################################################################################################################################  

    
    pd.DataFrame(dataset).to_csv(filename)
    
    print(filename, "Created!")


# ## Processing Input and Label Data

# In[2]:
# ########################################################################################################################################
    # Supervised machine learning entails training a predictive model on historical data with predefined target answers. An algorithm must 
    # be shown which targer answers on attributes to look for. Mapping these target attributes in a dataset is called labeling. Labeling is an
    # indispensible part of data preprocessing in Supervised learning.

    # From the above statements, in order for the machine learning algorithm to know what it should when a no comes up is based on the 
    # training data set we provide. So, we have "training.csv" file, which we use as a training set and labeling is done in that file.

    # Thus, in creating the training data, we make sure that the data preparation and data preprocessing are taken care of.
    
########################################################################################################################################

def processData(dataset):
    
    # Why do we have to process?
   
    data   = dataset['input'].values
    labels = dataset['label'].values
    
    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)
    
    return processedData, processedLabel


# In[3]:


def encodeData(data):
    
    processedData = []
    
    for dataInstance in data:
        
        # Why do we have number 10?
        processedData.append([dataInstance >> d & 1 for d in range(10)])
    
    return np.array(processedData)


# In[4]:


from keras.utils import np_utils

def encodeLabel(labels):
    
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel),4)


# ## Model Definition

# In[5]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard

##################################################################################################################################
# Keras is an API used to build and train deep learning methods.
# Keras has models associated with it. The models associated with it are
#           1. Sequential Model
#           2. Model class used in the function API.
# The fizzbuzz program uses the Sequential model.
# Sequential Model: Linear Stack of layers
# Layers imported are Dense, activation  and Dropout
# Dense Layer : 
#               Output = activation(dot(input, kernal)) + bias
#   where 
#   activation = Element wise activation function passed as the activation argument
#   kernal = weight matrix created by the layer
#   bias = bias vector created by the layer
# Activation layer : Applies the activation fn
# Dropout Layer: Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time to prevent
# overfitting.
# Callbacks: Callbacks are used to get a view on internal states and statistics of the model during training.

##################################################################################################################################


import numpy as np

input_size = 10
drop_out = 0.02
first_dense_layer_nodes  = 1024
second_dense_layer_nodes = 4


#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
#tensorboard("logs/run_a")

def get_model():
    
    # Why do we need a model?
    # Why use Dense layer and then activation?
    # Why use sequential model with layers?

    #################################################################################################################################
    # Machine Learning Model: Refers to the model artifact that is created by the training process
    # The process of training a machine learning model invovles providing a Machine learning algorithm with training data to learn from.
    # We use the machine learning model to get predictions on new data for which you don't know the target. 
    # 
    # In this fizzbuzz program, we have used sequential model.
    # 
    # After intialization, we have to give the input shape rather form a model. So, we have used a dense layer and activation function.
    # 
    # Activation Function: Main purpose is to convert an input signal of a node in ANN to an output signal. That output signal is now
    #                      used as an input in the next layer in the stack.
    # 
    # Without activation layer, our model will not be able to learn and model other complicated kinds of data such as images,videos etc
    # 
    # Most Popular activation functions are
    #           1. Sigmoid or Logistic Function
    #           2. tanh - hyperbolic tangent
    #           3. ReLu - Rectified Linear Units
    # 
    # In this program, we are using ReLu,
    # 
    # The function definition is as follows,
    #               R(x) = max(0,x) i.e
    #               if x < 0, R(x) = 0
    #                  x > 0, R(x) = x
    # 
    # One limitation of ReLu is that it should be used only in the hidden layers of a Neuron Network Model. That is why we specify the 
    # activation functiion after Dense layer
    # 
    # The next layer is the Dropout layer, Dropout is an approach to regularization in neural networks which helps reducing 
    # interdependent learning amongst the neurons.
    # 
    # Dropout forces a neural network to learn more robust features that are useful in conjunction with many different 
    # random subsets of the other neurons.
    # 
    # Dropout is a technique that addresses both these issues. It prevents overfitting and  provides a way of approximately combining
    # exponentially many different neural network architectures efficiently
    # 
    # After the dropout layer, we use another dense layer before using the softmax activation function
    # 
    # One of the most common classification is softmax classifier.
    #
    # Softmax function calculates the probabilities distribution of the event over ‘n’ different events. In general way of saying, 
    # this function will calculate the probabilities of each target class over all possible target classes. Later the calculated 
    # probabilities will be helpful for determining the target class for the given inputs.

    # The main advantage of using Softmax is the output probabilities range. The range will 0 to 1, and the sum of all the 
    # probabilities will be equal to one. If the softmax function used for multi-classification model it returns the probabilities
    # of each class and the target class will have the high probability.
    # 
    # 
    # The next step in the Sequential model is the compilation.
    # 
    # In the compile method, we configure the learning process before put into training. It takes three inputs as seen in line no 322
    #       1. Optimizer ---> String identifier of an exisitng optimizer.
    #       2. Loss      ---> String identifier of a loss function or an objective fn. The objective is that the model will minimize.
    #       3. metrics   ---> A metric could be the string identifier of an existing metric or a custom metric function.
    #
    #  Model is ready to be trained.  
    # 
    #  In this step, we will use our data to incrementally improve our model’s ability to predict whether a given number should return 
    #  fizz, buzz, fizzbuzz or other. Here we use the fit method. Basically, it trains the data for a given no of iterations.
    #   
    #  Once training is done, it is time for testing.

    #  For the data present in testing.csv, we try to predict the output when given a no. The output values for a data set is 
    #  Stored in output.csv
    # 
    #  This how the fizzbuzz program works, from generating the data set , creating a model , training the model and getting the 
    #  output for a completely different no than the one in training set   

    #################################################################################################################################
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    # Why dropout?
    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    # Why Softmax?
    

    model.summary()
    
    # Why use categorical_crossentropy?
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# # <font color='blue'>Creating Training and Testing Datafiles</font>

# In[6]:


# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')


# # <font color='blue'>Creating Model</font>

# In[7]:


model = get_model()


# # <font color = blue>Run Model</font>

# In[8]:


validation_data_split = 0.2
num_epochs = 10000
model_batch_size = 128
tb_batch_size = 32
early_patience = 100

tensorboard_cb   = TensorBoard(log_dir='logs/new_Tensors', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

# Read Dataset
dataset = pd.read_csv('training.csv')

# Process Dataset
processedData, processedLabel = processData(dataset)
history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [TensorBoard(log_dir='./logs/bes_performance', histogram_freq=10, batch_size=32, write_graph=True, write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None),earlystopping_cb]
                   )


# # <font color = blue>Training and Validation Graphs</font>

# In[9]:


# get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))


# # <font color = blue>Testing Accuracy [Software 2.0]</font>

# In[10]:


def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"


# In[11]:


wrong   = 0
right   = 0

testData = pd.read_csv('testing.csv')

processedTestData  = encodeData(testData['input'].values)
processedTestLabel = encodeLabel(testData['label'].values)
predictedTestLabel = []

for i,j in zip(processedTestData,processedTestLabel):
    y = model.predict(np.array(i).reshape(-1,10))
    predictedTestLabel.append(decodeLabel(y.argmax()))
    
    if j.argmax() == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))



# Please input your UBID and personNumber 
testDataInput = testData['input'].tolist()
testDataLabel = testData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "srivenka")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50288730")

predictedTestLabel.insert(0, "")
predictedTestLabel.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabel

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')


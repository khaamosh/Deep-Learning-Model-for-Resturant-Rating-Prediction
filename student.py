#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""


"""
Student Name : Uttkarsh Sharma
zid -> z5269665

Parameter Notes :
    1. Stop words : i tried different set of stop words and the most significant improvement was the only form the subset from the following link.
        https://gist.github.com/sebleier/554280
    
    2. Pre-Processing :
        For pre-processing i only have letters and digits and removed everything else.
        
    3. Size of word vectors : After testing i finalize the size to be 300 which gave the best results while testing.


Models Notes :
    
    Model Parameters :
        
        for LSTM
        ==================
        300 for word vector size
        256 for input layers
        batch_first=True for maintaining the size and other parameters.
        dropout=0.5
        num_layers=2, -> this is our number of hidden layers
        bidirectional=False
        ===================
        
        for Linear
        ===================
        256 for input to linear layer
        2 for output from linear layer
        
        
        
    Architecture of the model :
    ==================================        
        LSTM -> Fully Connected Linear Layer -> Leaky ReLU(Activation Function)
        
        I decided to use LSTM since that can fully augment the temporal layer and can closely match the learning 
        which is constitued by humans. Moreover i also leveraged the droput value of 0.5 which helps in factoring in unseen data.
        
    Loss function for the model :
        
        I tried both L1Loss and MSELoss, while MSE provided significant improvement, over L1Loss, 
        but since the variation of the diff of incremental loss values was less due to the fact that the difference range being between [0-1]
        
        i decided to use a  mean weighted linear loss function for the model.
        
        The procedure of the weighted loss :
            
            1. First i calculate the absoute difference of the target and output.
            2. then for the tensor i am comparing the values of the difference.
                
                2.1 if the difference is between 0 to 0.5 then the value is going towards the minima,hence weight is
                    0.2 for rating
                    0.17 for category
                2.2
                    if the difference is between 0.5 and 1 then the value is towards the correct value and hence weight is 
                    0.98 both for raing and category
                
                2.3
                    if the differnce is more than 1 then we need to penalize and hence we add
                    15 for rating 
                    100 for category
            
               2.4 in the end we take a mean of the weighted loss and then add them before returning.
    
    ConvertNetOuput:
        
        I also converted the ouput from my model by rounding the values, and transforming them to the following:
            rating is convered to 0 or 1
            category is converted to {1,2,3,4}
            
    Optimiser for the model:
        
    i decided to use ADAM as the optimiser for the model instead of SGD.
    I understand that this may be contrary to the one stated in the following paper
    --> https://arxiv.org/abs/1705.08292 
    
    thus in order to tackle the issue about generalization of the model, i had reduced the learning rate of the ADAM optimizer.
    While this may seem counter productive, but in my limited set of experimental iterations i observed, the following :
        1. levearging a smaller learning rate helps in getting the model not over generalized and also able to tackle the issue about hidden data.
        
    
    Note : I tested both the bi-directional and dropout changes.
    
    1. For the bi-directional changes the model was missing out the global minima and overshooting it, which resulted in a lower accuracy.
       i reckon this can also be attributed to the loss function that i wrote.
       However without the feature of bi-directiionality the model is giving an accuracy of 83(averaged over 50 tests)
    
    2. Having the dropout value of 0.5 proved to be the most useful as it helps in factoring in for the hidden data, thus improving the learning.     
        
               
"""





import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe

import re 

# import numpy as np
# import sklearn

from config import device


def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    #Enable for logging and checking the sample that is being passed.
    #print("sample")
    
    # for processing of the sample 
    
    processed = []
    
    #this is the regular expression that i used which factors in all english letters and words.
    pre_sample = re.compile("[^a-zA-Z\s\d]")
    
    
    for i in sample:
        i = pre_sample.sub(' ', i)
        if (len(i) > 1):
            processed.append(i)
    
    return processed

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """
    #Enable for logging and checking the batch of words being passed to the function.
    #print("batch")
    
    return batch

#this is the list of stop words that i looked into, this comprehensive list is refrencing
# 1. Refrence -> 

stopWords = {"a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are",
             "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
             "can", "d", "did", "do", "does", "doing", "don", "down",
             "during", "each", "few", "for", "from", "further", "had", "hadn", "has", 
             "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
             "if", "in", "into", "is",
            "it", "it's", "its", "itself", "just", "ll", "m",
             "ma", "me", "mightn", "more", "most", "my", "myself", "needn", "now", "o", "of",
             "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
             "over", "own", "re", "s", "same", "shan", "she", "she's", "should", "should've", "so", "some",
             "such", "t", "than", "that",
             "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
             "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very",
             "was", "we", "were", "what", "when", "where",
             "which", "while", "who", "whom", "why", "will", "with",
             "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
             "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's",
             "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's",
             "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's",
             "when's", "where's", "who's", "why's", "would"}

#Note :: That after testing i finally included the max which is 300 size for word vectors.

wordVectors = GloVe(name='6B', dim=300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    
    #Enable for logging and checking the parameters being sent.
    #print("here")
    #print(ratingOutput)
    #print(categoryOutput)
    
    #Note : the values need to be converted to the following range to match the input
    
    # For rating the values must be between 0 and 1.
    # For category the values must be between 0 and 4.
    
    #rating category conversion .
    ratingOutput = ratingOutput.round()
    ratingOutput[ratingOutput > 1] = 1
    ratingOutput[ratingOutput < 0] = 0
    
    
    #category output conversion .
    
    categoryOutput = categoryOutput.round()
    categoryOutput[categoryOutput > 4] = 4
    categoryOutput[categoryOutput < 0] = 0
    
    
    #Enable for post processing values for rating and category
    
    #print("New Rating and Category")
    #print(ratingOutput)
    
    #print("Category is below")
    #print(categoryOutput)
    
    return ratingOutput.type(torch.long), categoryOutput.type(torch.long)

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        
        
        # The parameters for the LSTM are as follows 
        # 300 from the word vector
        # 256 the input to the LSTM
        # batch_first being used refering to the course forum
        
        #creating a LSTM function here as follows 
        self.lstm  = tnn.LSTM(300,256,batch_first=True,dropout=0.5,num_layers=2,bidirectional=False) 
        
        # a fully conncected linear layer which will ouput the values.
        self.linear_1 = tnn.Linear(256, 2)
        
        #The activation function used is Leaky Relu
        self.relu = tnn.LeakyReLU()
        
        
    def forward(self, input, length):
        
        
        #because of the pytorch GPU error and hence length needs to be updated.
        #only set for when GPU is enabled.
        # this error is present due to pytorch version as the library assumes that the size of the layer needs to be CPU.
        
        length = torch.as_tensor(length, dtype=torch.int64, device='cpu')
        
        # using rnn.pack_padded for factoring the size constraints when the input is passed to LSTM layer.
        resized_padded = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True)
        
        output, (h_n, c_n) = self.lstm(resized_padded)
        
        # Enable for logging the value of the output shape.
        # print(output.shape)
        
        #Enable for logging : for the last hidden layer values.
        #print(h_n[-1])
        
        
        # the input given to the linear layers is the last hidden layer.
        lin_out = self.linear_1(h_n[-1, :, :])
        
        # the output from the linear layer is then sent to the activation function.
        rel_out = self.relu(lin_out)
        
        # Enable for logging to check the sensor data type
        #tensor.type(torch.DoubleTensor)
        
        #Note to self : for pytorch 1.2 data types are changed leveraging the .type function and providing the data type values ..
        return rel_out[:,0].type(torch.float), rel_out[:,1].type(torch.float)
    
        
class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        
        #Enable for testing case only.
        #self.loss = tnn.MSELoss()
        
        # Final iteration used weighted linear loss function since the values range from 0 - 1 and are small.
        self.loss = tnn.L1Loss()
        
    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        
        
        #trying to define a new loss function 
        
        # taking the absolute difference of the output to input .
        category_diff = torch.abs(categoryOutput - categoryTarget.type(torch.float))
        
        # taking the absolute difference of the rating between output and target.
        rating_diff = torch.abs(ratingOutput - ratingTarget.type(torch.float))
        
        # #adding the weights to the loss here.
        
        # Enable for logging only -> for rating loss and category loss.
        # #print(rating_loss)
        # #print(category_loss)
        
        # here wer are adding weights to the loss function.
        
        # tested few values and ended up with these variations.
        # 0.2 when it is beween 0 and 0.5
        # 0.98 when it is nearer to 1
        
        # adding a greter weight when the value is greter than 1 checked few variations and ended up with 15 for category 
        # loss function for rating is as follows 
        
        # 0.17 for rating diff
        # 0.99 when closer to 1
        # 100 for when greater than 1.
        
        for (i,j) in enumerate(category_diff):
                
                if j < 0.5:
                    category_diff[i]*= 0.2
                
                elif j < 1:
                    category_diff[i]*= 0.98
                
                else:
                    category_diff[i]+=15
        
        
        for (k,l) in enumerate(rating_diff):
            
            if l < 0.5:
                
                rating_diff[k]*=0.17
            
            elif l <1:
                rating_diff[k]*=0.99
            
            else:
                rating_diff[k]+=100
            
        
    
        # taking the mean of the losses and adding them before returning the values.
        
        return torch.mean(category_diff) + torch.mean(rating_diff)

        
    
   # def forward(self,ra) 

net = network() 
lossFunc = loss()


################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.85
batchSize = 32
epochs = 10
#optimiser = toptim.SGD(net.parameters(), lr=0.01)
optimiser = toptim.Adam(net.parameters(), lr=0.0005)
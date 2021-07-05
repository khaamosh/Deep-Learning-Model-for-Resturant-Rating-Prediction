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

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe

import re 

# import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################




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
    #print("sample")
    
    processed_sample = []
    
     #[re.sub(r'[^\x00-\x7f]', r'', w) for w in sample]
    #pre_sample = re.compile("[[^\x00-\x80]+]")
    pre_sample = re.compile("[^a-zA-Z\s\d]")
    
    for i in sample:
        i = pre_sample.sub(' ', i)
        if (len(i) > 1):
            processed_sample.append(i)
    
    return processed_sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """
    #print("batch")
    
    return batch


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
    #print("here")
    
    #print(ratingOutput)
    #print(categoryOutput)
    
    ratingOutput = ratingOutput.round()
    
    ratingOutput[ratingOutput > 1] = 1
    
    ratingOutput[ratingOutput < 0] = 0
    
    categoryOutput = categoryOutput.round()
    
    categoryOutput[categoryOutput > 4] = 4
    categoryOutput[categoryOutput < 0] = 0
    
    
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
        
        # This was the first iteration and gave score of 41.78
        
        #creating a LSTM function here as follows 
        self.lstm  = tnn.LSTM(300,256,batch_first=True,dropout=0.5,num_layers=2,bidirectional=False) 
        
        self.linear_1 = tnn.Linear(256, 2)
        
        #self.relu = tnn.ReLU()
        self.relu = tnn.LeakyReLU()
        
        #self.linear_2 = tnn.Linear(50, 2)
        
    def forward(self, input, length):
        
        #print(input.shape)
        
        #because of the pytorch GPU error and henc length needs to be updated.
        #only set for when GPU is enabled.
        length = torch.as_tensor(length, dtype=torch.int64, device='cpu')
        
        embeded = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True)
        
        output, (h_n, c_n) = self.lstm(embeded)
        
       # print(output.shape)
        
        #print(h_n[-1])
        
       # x = torch.cat((output[:, -1, :], output[:, 0, :]), dim=1)
        
        lin_out = self.linear_1(h_n[-1, :, :])
        
        rel_out = self.relu(lin_out)
        
        #print("output is as follows ")
        #print(rel_out.shape)
        
        #print(len(output))
        
        #tensor.type(torch.DoubleTensor)
        return rel_out[:,0].type(torch.float), rel_out[:,1].type(torch.float)
    
        
class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        
        self.loss = tnn.MSELoss()
        #self.loss = tnn.L1Loss()
        
    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        
        
        #the first loss functoin with accuracy of 68 percent
        
        #print(ratingOutput.shape)
        #print("ratingOutput type is as follows ")
        #print(ratingOutput.dtype)
        
        #print("rating Target is as follows ")
        #print(ratingTarget.dtype)
        
        #rating_diff = torch.abs(ratingOutput - ratingTarget)
        
        rating_loss = self.loss(ratingTarget.type(torch.float),ratingOutput)
        
        category_loss = self.loss(categoryTarget.type(torch.float) ,categoryOutput)
        
        #torch.cat((out[:, -1, :], out[:, 0, :]), dim=1)
        
        #print(category_loss.s)
        #print(type(rating_loss))
        
        
        
        
        # #trying to define a new loss function 
        
        # category_diff = torch.abs(categoryOutput - categoryTarget.type(torch.float))
        
        # rating_diff = torch.abs(ratingOutput - ratingTarget.type(torch.float))
        
        # #adding the weights to the loss here.
        
        # #print(rating_loss)
        # #print(category_loss)
        # for (i,j) in enumerate(category_diff):
                
        #         if j < 0.5:
        #             category_diff[i]*= 0.2
                
        #         elif j < 1:
        #             category_diff[i]*= 0.98
                
        #         else:
        #             category_diff[i]+=1.5
        
        # category_diff = torch.pow(category_diff,2)
        
        # for (k,l) in enumerate(rating_diff):
            
        #     if l < 0.5:
                
        #         rating_diff[k]*=0.2
            
        #     elif l <1:
        #         rating_diff[k]*=0.98
            
        #     else:
        #         rating_diff[k]+=1.5
            
        
        # rating_diff = torch.pow(rating_diff,2)
    
    
        # return torch.mean(category_diff) + torch.mean(rating_diff)
        return rating_loss + category_loss
    
        
    
   # def forward(self,ra) 

net = network()
#lossFunc = tnn.MSELoss() 
lossFunc = loss()


################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.85
batchSize = 32
epochs = 10
#optimiser = toptim.SGD(net.parameters(), lr=0.01)
optimiser = toptim.Adam(net.parameters(), lr=0.0005)
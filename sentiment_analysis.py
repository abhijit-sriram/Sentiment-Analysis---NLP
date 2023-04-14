#!/usr/bin/env python
# coding: utf-8

# ## SENTIMENT ANALYSIS USING PYTORCH

# In this code we will be building a machine learning model to detect the sentiment (i.e., if a sentence is positive or negative) based on movie reviews provided by a user using PyTorch. A recurrent neural network (RNN) processes sequence input by iterating through the elements. RNNs pass the outputs from one timestep to their input on the next timestep.

# In[1]:


import torch
import random
import torch.nn as nn
import torchtext
from torchtext import data
from torchtext.legacy.data import Field
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch.optim as optim


# In[2]:


SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# In[3]:


# The parameters of a Field specify how the data should be processed.
# We use the TEXT field to define how the review should be processed, and the LABEL field to process the sentiment.
TEXT = data.Field(tokenize = 'spacy',tokenizer_language = 'en_core_web_sm')
LABEL = data.LabelField(dtype = torch.float)


# In[4]:


# Feature of TorchText is that it has support for common datasets used in natural language processing (NLP)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)


# In[5]:


train_data, valid_data= train_data.split(random_state=random.seed(SEED))


# In[6]:


max_vocab_size=25000
TEXT.build_vocab(train_data,max_size= max_vocab_size)
LABEL.build_vocab(train_data)


# In[7]:


print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')
print(f"unique tokens in Text vocabulary :{len(TEXT.vocab)}")


# In[8]:


print(TEXT.vocab.freqs.most_common(40))
print(TEXT.vocab.itos[:30])
print(LABEL.vocab.stoi)


# In[9]:


BATCH_SIZE = 64


# In[10]:


# Check if a GPU is available, use it if available, otherwise use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create data iterators for the training, validation, and test sets
# Set the batch size and device for the iterators
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)


# In[11]:


# Define a custom RNN class that inherits from nn.Module
class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        
        # Define an Embedding layer
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        # Define an RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        
        # Define a Linear layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # Embed the input text
        embedded_text = self.embedding(text)
        
        # Pass the embedded text through the RNN layer
        rnn_output, hidden_state = self.rnn(embedded_text)
        
        # Assert that the last output of the RNN is equal to the final hidden state
        assert torch.equal(rnn_output[-1,:,:], hidden_state.squeeze(0))
        
        # Pass the final hidden state through the Linear layer
        output = self.fc(hidden_state.squeeze(0))
        return output


# In[12]:


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1


# In[13]:


model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)


# In[14]:


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[15]:


# Stochastic Gradient Descent optimizer with a learning rate of 0.001
optimizer = optim.SGD(model.parameters(), lr=1e-3)


# In[16]:


# Binary Cross Entropy with Logits loss function for binary classification
criterion = nn.BCEWithLogitsLoss()


# In[17]:


model = model.to(device)
criterion = criterion.to(device)


# In[26]:


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

# The function binary_accuracy calculates the accuracy of the model's predictions, given the predicted values and the actual labels. It takes two arguments:
# preds: the predicted values of the model.
# y: the actual labels.


# In[27]:


def train(model, iterator, optimizer, criterion):
    """
    Trains the given model on the given data iterator, using the given optimizer and loss criterion.
    """
    # Set the model to training mode
    model.train()

    # Initialize epoch loss and accuracy
    epoch_loss = 0.0
    epoch_acc = 0.0

    # Iterate over the batches in the iterator
    for i, batch in enumerate(iterator):
        # Zero the gradients of the optimizer
        optimizer.zero_grad()

        # Compute the model predictions for the current batch
        text = batch.text
        label = batch.label
        preds = model(text).squeeze(1)

        # Compute the loss and accuracy for the current batch
        loss = criterion(preds, label)
        acc = binary_accuracy(preds, label)

        # Backpropagate the loss and update the model parameters
        loss.backward()
        optimizer.step()

        # Accumulate the loss and accuracy for the epoch
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    # Compute the average loss and accuracy for the epoch
    num_batches = len(iterator)
    avg_loss = epoch_loss / num_batches
    avg_acc = epoch_acc / num_batches

    return avg_loss, avg_acc


# In[28]:


def evaluate(model, iterator, criterion):
    # Set model to evaluation mode
    model.eval()
    
    # Initialize loss and accuracy
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        # Loop over batches in iterator
        for batch in iterator:
            # Make predictions using the model
            predictions = model(batch.text).squeeze(1)
            
            # Compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            # Accumulate loss and accuracy across batches
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    # Compute and return average loss and accuracy across batches
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[29]:


# Load the trained model parameters saved in 'tut1-model.pt'
model.load_state_dict(torch.load('tut1-model.pt'))


# In[30]:


loss, acc = evaluate(model, test_iterator, criterion)


# In[31]:


print(f'Test Acc: {acc*100:.2f}%')
print(f'Test Loss: {loss:.3f}%')


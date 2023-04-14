# Sentiment-Analysis - NLP

Install PyTorch - see installation instructions on the PyTorch website - https://pytorch.org/get-started/locally/
OR
pip install torchtext

IMDB Dataset used - http://ai.stanford.edu/~amaas/data/sentiment/ 

Sentiment Analysis - Sentiment Analysis, as the name suggests, it means to identify the view or emotion behind a situation. It basically means to analyze and find the emotion or intent behind a piece of text or speech or any mode of communication. Sentiment analysis (or opinion mining) is a natural language processing (NLP) technique used to determine whether data is positive, negative or neutral. Sentiment analysis is often performed on textual data to help businesses monitor brand and product sentiment in customer feedback, and understand customer needs. Her we perform it on movie reviews from IMDB dataset.

RNN - A recurrent neural network (RNN) is a type of artificial neural network which uses sequential data or time series data. These deep learning algorithms are commonly used for ordinal or temporal problems, such as language translation, natural language processing (nlp), speech recognition, and image captioning; they are incorporated into popular applications such as Siri, voice search, and Google Translate. Like feedforward and convolutional neural networks (CNNs), recurrent neural networks utilize training data to learn. They are distinguished by their “memory” as they take information from prior inputs to influence the current input and output. While traditional deep neural networks assume that inputs and outputs are independent of each other, the output of recurrent neural networks depend on the prior elements within the sequence. While future events would also be helpful in determining the output of a given sequence, unidirectional recurrent neural networks cannot account for these events in their predictions.

In this project, the goal is to build a machine learning model to detect the sentiment (positive or negative) of movie reviews using PyTorch and TorchText. A recurrent neural network (RNN) will be used for this task, which takes in a sequence of words and produces a hidden state for each word. The TorchText library will be used to define how the data should be processed, including the spaCy tokenizer and en_core_web_sm model for tokenization, and the LabelField class for sentiment processing. The IMDb dataset, consisting of 50,000 movie reviews marked as positive or negative, will be used for training and testing the model.

To build the vocabulary, a lookup table that assigns each unique word in the dataset a corresponding index, only the top 25,000 most common words in the training set are kept. Any words that appear in examples but not in the vocabulary are replaced with a special unknown token. A BucketIterator is then used to prepare the data for iteration, returning batches of examples with similar lengths. The model architecture for sentiment analysis is defined using PyTorch, containing an embedding layer, an RNN layer, and a linear layer.

In the training process, an optimizer (SGD) and a loss function (BCEWithLogitsLoss criterion) are created, along with a function to calculate the accuracy of predictions. The model is then trained through multiple epochs, with the best parameters being saved based on validation loss. The saved model is then used to make predictions on the test set. If the loss is not decreasing significantly and the accuracy is low, this can be attributed to a number of problems with the model that will be addressed in the next notebook. At the end, the model's performance on the test set is evaluated using the saved parameters that resulted in the best validation loss.

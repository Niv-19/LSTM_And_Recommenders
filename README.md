# LSTM Based Sentiment Prediction And Recommenders
This project illustrates Neural Network approach to sentiment prediction and is a continuation of my previous project (link) to further enhance sentiment prediction of Amazon products reviews using most recent technique: Long Short-Term Memory (LSTM) Neural Network. <br/>

Also, two types of product recommender systems are created:
*   Content Based 
*   Collabortive Filtering using unsupervised and supervised KNN.

## Data
The data consisted on 1.6M reviews for Electronics with its associated ratings between 1 to 5. The data was taken from here: [Amazon Customer reviews](http://jmcauley.ucsd.edu/data/amazon "Dataset for project").
<br/>
Only the Electronics sub-category and its metadata have been used for this project.<br/>

As part of chosing the best LSTM model, a subset of first 500K reviews were taken out of 1.6M reviews for training and validating the model. When the best model was selected, the model was retrained on 1M data and last 600K rows were left as hidden unseen data.<br/>

The Ratings between 1 and 5 were used to create Sentiment Feature which converts ratings 3,4 & 5 as Positive, 1 & 2 as negative reviews.<br/>

### Data Preparation
The data contained 9 columns out of which, product ID, reviewer ID, rating and review-summary were used to perform feature engineering and create required fields.

Before fitting the model, following processing were done: <br/>
* Lowercasing all words in reviews.
* Removing Special Characters ( . , ! ? ‘ etc)
* Convert words into vectors using Tokenizer with 2500 most common words.
* Convert text to sequence that indicate the ordered frequency of each word in the dataset.
* Include max length and padding to the text sequences.
* Sentiments converted into 1 and 0.
* Split the data into training and validation data.

Review summaries were cleaned as mentioned above and used as the feature and sentiments (positive/negative) as target for LSTM model.

### What is LSTM?
LSTMs, a very special kind of Recurrent Neural Network which works, for many tasks, much better than the standard version RNN.

An LSTM has three of these gates, to protect and control the cell state.
Forget Gate
Input Gate
Output Gate

In RNN, as the gap between depenedent words increase, model becomes unable to learn to connect the information.

LSTMs are explicitly designed to avoid the long-term dependency problem. The key to LSTMs is the cell state c(t). A cell state is an additional way to store memory, beyond using the hidden state h(t). <br/>

LSTM Architecture:<br/>
![LSTMmodel](images/LSTMarchitecture)

### Word Embedding
Word Embeddings are a distributed representation for text that is perhaps one of the key breakthroughs for the impressive performance of deep learning methods on challenging NLP problems.
**complete**

### Fitting and Evaluating Model
Naturally, LSTMs are bound to overfitting due to its complex architecture. <br/>

Different LSTM models were created using different combinations of hyper-parameters as mentioned below:<br/>
* Number of layers (LSTMs, Dense).
* Number of units (nodes) in a layer.
* Strength of Dropouts
* Type of Activation Function (relu, tanh, sigmoid)
* Batch size ( for Gradient Descend)
* Learning rate 
* Number of Epochs

Best model that was obtained was with 1 embedding layer of 70 output dim, 4 LSTMs layers with units of 100, 100, 72 and 100 respectively and 1 output Dense layer of 2 units.







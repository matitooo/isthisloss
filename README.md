#ANLP-Sarcasm-IsThisLoss
Shared repo for group NLP assignment on sarcasm detection

The following project is a Sarcasm detection machine that combines an (BI)LSTM, an attention layer, and manually extracted features.

You can check the requirements file to know which libraries are required.

# Why the project is useful.

The importance of sarcasm detection as an NLP
problem comes from the fact that sarcasm can af-
fect the polarity of sentences (Poria et al., 2016),
causing an inverse of meaning. This can seriously
affect how automatic text processors understand
texts, and, as explained by Medhat et al. (2014),
is then relevant for sentiment analysis, impacting
areas such as marketing, investment, and adminis-
tration. We have decided to combine the power of
neural networks with old NLP techniques, mainly
manual feature extraction, to create a model ca-
pable of handling sarcasm detection with a low
computational cost.

# How users can get started with the project.

#USAGE

The available models are the following: 

'lstm'
'lstm-attention'
'lstm-features'
'lstm-attention-features'
'bi-lstm-attention-features'

We devised three user modes, one for training, one for fine tuning, 
and one for testing.
They can be found with the following names:

'training'
'tuning'
'testing'

The model can be used by launching the following command: 

python main.py —mode 'selected_mode' —model 'selected_model'

where 'selected_mode' and 'selected_model' must be replaced with 
one of the available parameters.

    #TRAINING

    The 'training' command trains the chosen model with the optimal configuration
    and saves results as a Weights&Biases run.

    #TUNING

    The 'tuning' command retrieves a sweep configuration from the sweep_config.yaml 
    and performs hyperparameter tuning with Bayesian optimization on the selected model
    and saves results as a Weights&Biases run.

    #TESTING

    The 'testing' command  retrieves model weights from a pre-trained instance of the
    selected model stored in the weights folder and returns metrics by evaluating the model
    on the train set and on the test set.



# MODEL

In the model folder you will find all possibles variations of the LSTM:
- LSTM
- LSTM + attention
- LSTM + manual features
- LSTM + attention + manual features
- BI-LSTM + attention + manual features

# CLASSIFICATION FOLDER 

The models were implemented in modular code. 
So, for example, the LSTM class is inherited by LSTMWithAttention class, later 
inherited by LSTMWithAttentionFeatures. 
For each final architecture a classificator class (sarcasm predictors) 
was created in the classificator folder, which gets a padded batch of tweets 
as input and outputs a prediction tensor with 1 and 0.


# DATA

In the data folder you will find the original training data and our test data.
This comes in the form of a CSV file, containing 3470 tweets .

The first column corresponds to the text of the tweet.
The second column corresponds to the label of the tweet (1 for sarcastic and 0 for non-sarcastic).
The third column corresponds to a rephrased non-sarcastic version of the tweet.
The following columns each correspond to a sarcasm subcategory (sarcasm, irony, satirem understatement, overstatement, rhetorical_question).

Example:
Today my pop-pop told me I was not “forced” to go to college 🙃 okay sure sureeee, 1, Today my pop-pop told me I was not "forced" to go to college. That's not true. 1,0,0,0,0,0

Test data is more simple, only having the text and the label. It contains 1400 examples.

Additionally, there is the GloVE folder, that contains a dictionary with the GloVe pretrained embedding dictionary.


# Features

In features.py you will find a file containing a script that goes through 
the tweets and analysis them for sarcasm features, including:

- Emojis,
- Repeated letters,
- Contains question mark and/or exclamation mark,
- Repeated punctuation,
- Three dots,
- Quotations,
- Has negation,
- Interjections.


# Metrics

In metrics there are two functions. The first one computes accuracy, precision, recall, 
and F1 score in one pass to save computation.
While the other creates a confusion matrix. 

# Preprocess

In preprocess.py we normalized our data for better processing. 
This includes: 
- Turning words to lower case.
- Removing punctuation
- Removing stopwords
- Lemmatizing
- Removing hashtags


# sweep_config.yaml

Contains the configuration we used for tuning.


# utils

utils.py contains two functions for extracting data from the csv files:
one for the train data and one for the test data respectively.
Additionallly, there is a function for the implementation of the GloVe embeddings 
that applies manual features extraction as well.
Finally, there is one function that embeds the tweets. 
This function uses the preprocess_text from preprocess_data.py
and outputs packed batches of padded tweets.


# OTHER INFORMATION

Feel free to check our bibliography.bib for our bibliography.

Contact us freely to our emails:

Juan Pedro Danza Rovira – danzarovir@uni-potsdam.de
Mattia D’Agostini – mattia.dagostini@uni-potsdam.de
Evangelia Bourazani – evangelia.bourazani@uni-potsdam.de
Ivan Samodelkin - ivan.samodelkin@uni-potsdam.de





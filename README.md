# WSDM
Kaggle competition : https://kaggle.com/c/fake-news-pair-classification-challenge/

This project allows to find the link between 2 claims (A and B). There is the A and B claims as input and as output the link (agreed, disagreed or unrelated).

* agreed: B talks about the same fake news as A
* disagreed: B refutes the fake news in A
* unrelated: B is unrelated to A 

We have tried several models that are described below. There are also other files that will also be described.

## The different models

For all the models we have as input 2 sentences A and B (x1 and x2 in the code) and we want the predict one of the 3 classes.

### modelSSS
This model is divided into 2 major parts : inter-attention and LSTM and the prediction part.

#### inter-attention
Here we want to determine an attention score for each word of A using B (and conversely). First we contatenate every words of A with every words of B, then we apply a dense layer on each pair of words (one word from A and one from B) to obtain the score of each pair. Finaly we have a score matrix and we obtain the score of each word by taking the highest score of rows and of columns. 

After that we reduce sentences A and B by multiplying each word with its attention score and by adding every words of the sentence. At the end we concatenate the sentence A's representation with the B's one and have an inter attention representation of these 2 sentences.

#### LSTM
We apply a recurrent layer using LSTM cell on the concatenation of the 2 sentences (A and B). We take the last output as representation of these 2 sentences.

#### Prediction
Finally we concatenate these 2 differente representations of the 2 sentences and apply a dense layer. The output is a 3 dimensions vector to classify as agreed, disagreed or unrelated.

### modelSDD
Same model as modelSSS exept that we apply LSTM on each sentence separately (and not the concatenation of A and B). So we have LSTM representation for sentence A and sentence B. 

To predict we concatenate inter-attention representation of A and LSTM representation of A and apply dense layer. We do the same for the sentence B and finally we predict with a dense layer.

### modelDDD
Same model as modelSDD exept that we use 2 different score matrixs. One matrix obtains by concatenating A and B and the other one by concatenating B and A. So there is one matrix for words from A and one for words from B. hen we obtain the score of each word by taking the highest score only of rows. The end of the model is identical to modelSDD.


### modelSSS_dropout
Same model as modelSSS but with dropout one every layer (inter-attention layer, LSTM and prediction).

### modelSSS_attention
Same model as modelSSS but with a self attention layer after LSTM. In this model we take LSTM outputs of all words and apply self attention. 

### modelSSS_multi_LSTM
Same model as modelSSS expect that we apply LSTM on the concatenation, the dot product and the substraction of sentences A and B. After we use a dense layer to obtain the LSTM representation of these 2 sentences.

### modelSSS_multi_compare
Same model as modelSSS expect that we don't only concatenate inter-attention of A and B but we also do a dot product and a subtraction of them.

### modelSSS_AMCMR
Same model as modelSSS with multi_LSTM and attention on each one and multi_compare.


## Dataset

The dataset.py script allows to create pickle files of the data. The train_dataset pickle file contains 4 lists :
* train_X : containing training data,
* train_y : containing training labels,
* test_X : containing testing data,
* test_y : containing testing labels.

data is a (-1, 2, max_sentence_length, embedding_dimension) array. -1 is the number of data and 2 is for sentences A and B.

labels is a (-1, 3) array. -1 is the number of data and 3 is the number of classes (agreed, disagreed and unrelated).

## Train, test and submission

These scripts allow to train the model, test to obtain accuracy and create a submission csv file for the Kaggle competition.

## Results

results.txt stores the test results for WSDM dataset and results2.txt for Stanford dataset. Results.odt contains the same results as results.txt with some comments


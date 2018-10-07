# Speaker Verification

## Abstract
Implementing and Training of a Convolutional Neural Network to do Speaker Verification, i.e. decide whether two utterances (speech segments) were produced by the same speaker.

## Basic Description

The **input** is a *trial*, a pair of utterances.

The **CNN output** is a speaker embedding - a fixed-length vector representing acoustic features that vary across different speakers.

Given this speaker embedding, the system will use an appropriate metric (cosine similarity or negative L2 distance) to produce the similarity scores.

The **system output** is a numeric score quanitifying how similar the speakers of the two utterances seem to be. 

System: pass each utterance through the CNN, obtain the speaker embeddings and then compute the similarity score.

## Training

### N-way classification

A simple approach is to treat this as an N-way classification task, where the system classifies each utterance among N classes (N being the number of unique speakers in the training set).

Network architecture: 
- several convolutional layers for feature extraction
- output of last feature extraction layer is the speaker embedding
- linear layer of dimensionality `embedding_dim x num_speakers`, then softmax
- using cross-entropy loss for optimizer

#### Alternative loss functions

This N-way classification system does not optimize for the similarity metric mentioned above. More specialized (and interesting) loss functions include:
- center loss
- contrastive loss
- triplet loss
- angular loss

## Evaluation

A threshold score is needed to accept or reject pairs as *same-speaker*  or *different-speaker* pairs. 

A natural choice for this threshold is one which equates the *false acceptance* and *false rejection* rates. Thus, evaluation will be based on the Equal Error Rate (EER) metric.

## Dataset

The training data contains a few thousand hours of speaker-labeled speech (set of utterances, each labeled with a speaker ID). This raw data is about 150GB in size. 

Development and debugging will obviously be done in chunks (< 2GB), prototyping and tuning on larger chunks (< 10GB after preprocessing), training the final model on the full set (40GB after preprocessing).

### Training set

Training data is structured into two arrays, `features` and `speakers`, where `features[i]` contains the `i`th utterance, spoken by speaker with ID `speakers[i]`. 

An utterance is a (variable-length!) array of frames. Each second of speech has 100 frames, each frame has 64 log mel filters, encoded as 16-bit floats.

Preprocessing will filter out silent frames in order to reduce file size and increase performance.

### Validation and test set

Validation and test will be done using `enrollment` utternces, `test` utterances and `trials`. Trials specify the indices of `enrollment` and `test` utterances, for each of which the speaker similarity will be tested and assessed.

Trials in the validation set are labeled (the label being `True` *iff* the two utterances belong to the same speaker).

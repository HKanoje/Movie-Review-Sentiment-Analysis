# Movie Review Sentiment Analysis

## Overview
This project performs sentiment analysis on a corpus of movie reviews. It uses natural language processing (NLP) techniques to preprocess text data, vectorize words, and train several machine learning models to classify reviews as positive or negative.

## Features
- Preprocessing of text data, including tokenization, lemmatization, and stop-word removal.
- Coverage analysis to determine the proportion of text data represented by the most common words.
- Vectorization of preprocessed text using both CountVectorizer (TF) and TfidfVectorizer (TF-IDF).
- Training and evaluation of different classifiers including Multinomial Naive Bayes, Logistic Regression, and Multilayer Perceptron (MLP).
- Performance evaluation using accuracy, true positive rate (TPR), and false positive rate (FPR).
- Visualization of the distribution of movie reviews, coverage analysis, and classifier performance.

## Prerequisites
Before running this task 2, you need to have the following installed:
- Python 3.6 or later
- NLTK
- scikit-learn
- Matplotlib
- NumPy

You can install the necessary libraries using the following command:
`pip install -r requirements.txt`


## Data
The dataset used is the `movie_reviews` dataset from the NLTK library, which consists of 2,000 movie reviews categorized as either positive or negative.

## Usage
To run the project, execute the `main.py` script:


The script will perform all the steps from preprocessing to evaluation and show plots. Please close the plots as they open after having a look in order to execute next part.

## Structure
- `classifiers.py`: Contains the definition and training of the machine learning models as well as functions for performance evaluation.
- `main.py`: The main script that orchestrates data loading, preprocessing, visualization, and interaction with the classifiers.



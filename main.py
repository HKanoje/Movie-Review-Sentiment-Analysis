import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from classifiers import nb_classifier, logistic_classifier, mlp_classifier, train_and_evaluate_classifier
import numpy as np
from sklearn.metrics import confusion_matrix
import random

# Download necessary NLTK resources
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load the movie reviews dataset
print("Loading movie reviews dataset...")
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

print("Sample Review:", documents[0][0][:10]) # Displaying the first 10 words for brevity
print("Label:", documents[0][1])

# Plotting the distribution of movie reviews
print("Plotting distribution of movie reviews...")
categories = movie_reviews.categories()
counts = [len(movie_reviews.fileids(category)) for category in categories]
plt.bar(categories, counts, color=['green', 'red'])
plt.title('Distribution of Movie Reviews')
plt.xlabel('Review Sentiment')
plt.ylabel('Number of Reviews')
plt.xticks(categories, ['Positive', 'Negative'])
plt.show()

# Preprocessing
print("Preprocessing the data...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return lemmatized_tokens

all_words_after = [word for fileid in movie_reviews.fileids() for word in preprocess(movie_reviews.raw(fileid))]

print(all_words_after[:10])

# Coverage analysis
print("Performing coverage analysis...")
word_counts_after = Counter(all_words_after)
cumulative_coverage = []
cumulative = 0
for i, word in enumerate(word_counts_after.most_common(), start=1):
    cumulative += word[1]
    cumulative_coverage.append(cumulative / len(all_words_after))
    if i % (len(word_counts_after) // 20) == 0:
        print(f"Progress: {i/len(word_counts_after):.2%} - Coverage: {cumulative_coverage[-1]:.2%}")
plt.plot(list(range(1, len(cumulative_coverage) + 1)), cumulative_coverage)
plt.title('Coverage Analysis of Preprocessed Tokens')
plt.xlabel('Number of Tokens Considered')
plt.ylabel('Coverage Percentage')
plt.show()

# Splitting data
print("Splitting the data into training and testing sets...")
train_docs, test_docs = train_test_split(documents, test_size=0.2, random_state=42)
X_train = [" ".join(words) for words, category in train_docs]
y_train = [category for words, category in train_docs]
X_test = [" ".join(words) for words, category in test_docs]
y_test = [category for words, category in test_docs]

# Vectorization (TF and TF-IDF)
print("Vectorizing the data...")
vectorizer_tf = CountVectorizer()
X_train_tf = vectorizer_tf.fit_transform(X_train)
X_test_tf = vectorizer_tf.transform(X_test)
vectorizer_tfidf = TfidfVectorizer()
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

# Training and evaluating classifiers
y_pred_nb_tf = train_and_evaluate_classifier(nb_classifier, X_train_tf, y_train, X_test_tf, y_test, "TF")
y_pred_logistic_tf = train_and_evaluate_classifier(logistic_classifier, X_train_tf, y_train, X_test_tf, y_test, "TF")
y_pred_mlp_tf = train_and_evaluate_classifier(mlp_classifier, X_train_tf, y_train, X_test_tf, y_test, "TF")

y_pred_nb_tfidf = train_and_evaluate_classifier(nb_classifier, X_train_tfidf, y_train, X_test_tfidf, y_test, "TF-IDF")
y_pred_logistic_tfidf = train_and_evaluate_classifier(logistic_classifier, X_train_tfidf, y_train, X_test_tfidf, y_test, "TF-IDF")
y_pred_mlp_tfidf = train_and_evaluate_classifier(mlp_classifier, X_train_tfidf, y_train, X_test_tfidf, y_test, "TF-IDF")

# Performance metrics calculations and plot results
def calculate_performance_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    tpr = tp / (tp + fn)  # True Positive Rate (Sensitivity)
    fpr = fp / (fp + tn)  # False Positive Rate
    return accuracy, tpr, fpr

# Calculate metrics for TF features
accuracy_nb_tf, tpr_nb_tf, fpr_nb_tf = calculate_performance_metrics(y_test, y_pred_nb_tf)
accuracy_logistic_tf, tpr_logistic_tf, fpr_logistic_tf = calculate_performance_metrics(y_test, y_pred_logistic_tf)
accuracy_mlp_tf, tpr_mlp_tf, fpr_mlp_tf = calculate_performance_metrics(y_test, y_pred_mlp_tf)

# Calculate metrics for TF-IDF features
accuracy_nb_tfidf, tpr_nb_tfidf, fpr_nb_tfidf = calculate_performance_metrics(y_test, y_pred_nb_tfidf)
accuracy_logistic_tfidf, tpr_logistic_tfidf, fpr_logistic_tfidf = calculate_performance_metrics(y_test, y_pred_logistic_tfidf)
accuracy_mlp_tfidf, tpr_mlp_tfidf, fpr_mlp_tfidf = calculate_performance_metrics(y_test, y_pred_mlp_tfidf)

# Plotting accuracy, TPR, and FPR
classifiers = ['Naive Bayes', 'Logistic Regression', 'MLP']
metrics_tf = [accuracy_nb_tf, accuracy_logistic_tf, accuracy_mlp_tf]
metrics_tfidf = [accuracy_nb_tfidf, accuracy_logistic_tfidf, accuracy_mlp_tfidf]

x = np.arange(len(classifiers))
width = 0.35

# Plot Accuracy
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, metrics_tf, width, label='TF')
rects2 = ax.bar(x + width/2, metrics_tfidf, width, label='TF-IDF')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by classifier and feature representation')
ax.set_xticks(x)
ax.set_xticklabels(classifiers)
ax.legend()
plt.show()


# Plot TPR
tpr_tf = [tpr_nb_tf, tpr_logistic_tf, tpr_mlp_tf]
tpr_tfidf = [tpr_nb_tfidf, tpr_logistic_tfidf, tpr_mlp_tfidf]
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, tpr_tf, width, label='TF')
rects2 = ax.bar(x + width/2, tpr_tfidf, width, label='TF-IDF')
ax.set_ylabel('True Positive Rate (TPR)')
ax.set_title('TPR by classifier and feature representation')
ax.set_xticks(x)
ax.set_xticklabels(classifiers)
ax.legend()
plt.show()

# Plot FPR
fpr_tf = [fpr_nb_tf, fpr_logistic_tf, fpr_mlp_tf]
fpr_tfidf = [fpr_nb_tfidf, fpr_logistic_tfidf, fpr_mlp_tfidf]
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, fpr_tf, width, label='TF')
rects2 = ax.bar(x + width/2, fpr_tfidf, width, label='TF-IDF')
ax.set_ylabel('False Positive Rate (FPR)')
ax.set_title('FPR by classifier and feature representation')
ax.set_xticks(x)
ax.set_xticklabels(classifiers)
ax.legend()
plt.show()

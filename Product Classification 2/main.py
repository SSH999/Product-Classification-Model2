import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load the dataset
df = pd.read_csv('product_data1.csv')

# Preprocess the data
stop_words = set(stopwords.words('english'))
df['cleaned_text'] = df['product_description'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in stop_words]))
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['category'].values

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(classification_report(y_test, y_pred))
# Define a sample dataset
sample_data = {'product_description': ['This book is a great read', 
                                        'These headphones have poor sound quality', 
                                        'I love this new phone', 
                                        'This shirt is uncomfortable', 
                                        'This cleaning product is not effective'], 
               'category': ['books', 'electronics', 'electronics', 'clothing', 'home and kitchen']}

# Convert sample dataset to a DataFrame
sample_df = pd.DataFrame(sample_data)

# Preprocess the sample data
sample_df['cleaned_text'] = sample_df['product_description'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in stop_words]))

# Transform the preprocessed text to features
sample_X = vectorizer.transform(sample_df['cleaned_text'])

# Make predictions on the sample dataset
sample_y_pred = nb_classifier.predict(sample_X)

# Print the predicted categories
print(sample_y_pred)






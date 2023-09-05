import pandas as pd
import nltk
import re

from nltk.tokenize import word_tokenize
from collections import Counter

# Function to read a CSV file and handle potential file errors
def read_csv(filepath):
    try:
        # Attempt to read the CSV file
        return pd.read_csv(filepath, encoding="ISO-8859-1")
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'.")
        exit()
    except Exception as e:
        print(f"Error: File not found at '{filepath}'")
        exit()

# Function to preprocess and tokenize a given message
def clean_message(message):
    message = message.lower()
    # Remove special characters
    message = re.sub(r'[^a-zA-Z0-9\s]', '', message)
    # Tokenize the cleaned message
    return word_tokenize(message)

# Function to gather words from a DataFrame based on a specific label (ham or spam)
def accumulate_words(dataframe, label_value):
    # List comprehension to extract words for a given label
    return [word for label, msg in zip(dataframe['label'], dataframe['message']) if label == label_value for word in msg]

# Calculate the smoothed probability of a word in a given category (ham/spam)
def calculate_word_probability(word, word_count, total_words, vocabulary):
    laplace_smoothing = 1
    word_occurrences = word_count[word]
    return (word_occurrences + laplace_smoothing) / (len(total_words) + len(vocabulary))

# Function to classify messages in the test dataset as ham or spam
def classify_messages(test_data, ham_word_count, spam_word_count, ham_words, spam_words, vocabulary):
    # Calculate prior probabilities for ham and spam
    p_spam = len(spam_words) / (len(spam_words) + len(ham_words))
    p_ham = len(ham_words) / (len(spam_words) + len(ham_words))
    message_prediction = []
    for _, row in test_data.iterrows():
        # Initialize probability product with prior probabilities
        p_word_spam_product = p_spam
        p_word_ham_product = p_ham

        for word in clean_message(row["message"]):
            # Multiply the word probabilities to get message probability
            p_word_spam_product *= calculate_word_probability(word, spam_word_count, spam_words, vocabulary)
            p_word_ham_product *= calculate_word_probability(word, ham_word_count, ham_words, vocabulary)

        # Determine if the message is more likely to be ham or spam
        if p_word_spam_product > p_word_ham_product:
            message_prediction.append("SPAM")
        else:
            message_prediction.append("HAM")
    test_data['predicted_label'] = message_prediction


def main():
    # Paths to the training and test datasets
    test_data_path = "Datasets/TestData.csv"
    training_data_path = "Datasets/TrainingData.csv"
    result_path = "Datasets/ResultData.csv"

    # Load data from the CSV files
    test_data = read_csv(test_data_path)
    training_data = read_csv(training_data_path)

    # Preprocess training data: remove duplicates and null messages
    training_data.drop_duplicates(inplace=True)
    training_data.dropna(subset=['message'], inplace=True)

    # Clean and tokenize the messages in the training dataset
    print("Cleaning data...")
    training_data["message"] = training_data["message"].apply(clean_message)

    # Extract ham and spam words from the training data
    print("Extracting ham and spam word from dataset...")
    ham_words = accumulate_words(training_data, 'ham')
    spam_words = accumulate_words(training_data, 'spam')
    
    # Build the vocabulary from the training data
    vocabulary = set(word for message in training_data["message"] for word in message)

    # Count occurrences of each word for ham and spam
    ham_word_count = Counter(ham_words)
    spam_word_count = Counter(spam_words)

    # Classify the messages in the test dataset
    print("Classifying messages...")
    classify_messages(test_data, ham_word_count, spam_word_count, ham_words, spam_words, vocabulary)

    # Save the updated DataFrame with predicted labels to a new CSV file
    test_data.to_csv(result_path, index=False)
    print(f"Predicted data saved to {result_path}")

# Entry point of the script
if __name__ == '__main__':
    main()


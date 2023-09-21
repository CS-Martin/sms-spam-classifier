import pandas as pd
import nltk
import classifier
from classifier import clean_message
from collections import Counter

def read_csv(filepath):
    try:
        return pd.read_csv(filepath, encoding="ISO-8859-1")
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'.")
        exit()
    except Exception as e:
        print(f"Error: File not found at '{filepath}'")
        exit()

# Function to gather words from a DataFrame based on a specific label 
def accumulate_words(dataframe, label_value):
    return [word for label, msg in zip(dataframe['label'], dataframe['message']) if label == label_value for word in msg]

def main():
    test_data_path = "Datasets/TestData.csv"
    training_data_path = "Datasets/TrainingData.csv"
    result_path = "Datasets/ResultData.csv"

    test_data = read_csv(test_data_path)
    training_data = read_csv(training_data_path)

    training_data.drop_duplicates(inplace=True)
    training_data.dropna(subset=['message'], inplace=True)

    print("Cleaning data...")
    training_data["message"] = training_data["message"].apply(clean_message)

    print("Extracting ham and spam word from dataset...")
    ham_words = accumulate_words(training_data, 'ham')
    spam_words = accumulate_words(training_data, 'spam')
    
    vocabulary = set(word for message in training_data["message"] for word in message)

    ham_word_count = Counter(ham_words)
    spam_word_count = Counter(spam_words)

    print("Classifying messages...")
    classifier.classify_messages(test_data, ham_word_count, spam_word_count, ham_words, spam_words, vocabulary)

    test_data.to_csv(result_path, index=False)
    print(f"Predicted data saved to {result_path}")

if __name__ == '__main__':
    main()
import pandas as pd
from model import clean_message, classify_messages
from precision_recall import compute_precision_recall
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
    # Read the data
    test_data_path = "Datasets/TestData.csv"
    training_data_path = "Datasets/TrainingData.csv"
    result_data_path = "Results/ResultData.csv"

    test_data = read_csv(test_data_path)
    training_data = read_csv(training_data_path)

    # Remove duplicate messages and messages with no content
    training_data.drop_duplicates(inplace=True)
    training_data.dropna(subset=['message'], inplace=True)

    # Clean the messages in the training data
    print("Cleaning data...")
    training_data["message"] = training_data["message"].apply(clean_message)

    # Gather all the words from the training data
    print("Extracting ham and spam word from dataset...")
    ham_words = accumulate_words(training_data, 'ham')
    spam_words = accumulate_words(training_data, 'spam')
    
    # vocabulary is a set of all the words in the training data
    vocabulary = set(word for message in training_data["message"] for word in message)

    # Count the number of occurrences of each word in the training data
    ham_word_count = Counter(ham_words)
    spam_word_count = Counter(spam_words)

    # Classify the messages in the test data
    print("Classifying messages...")
    classify_messages(test_data, ham_word_count, spam_word_count, ham_words, spam_words, vocabulary)

    # Save the predicted data to a CSV file
    test_data.to_csv(result_data_path, index=False)
    print(f"Predicted data saved to {result_data_path}")

    # store labeled data path
    labeled_data_path = "Datasets/LabeledTestData.csv"

    # Read result data and the labeled data
    resultData = read_csv(result_data_path)
    labeledTestData = read_csv(labeled_data_path)
    
    # Compute precision and recall
    compute_precision_recall(resultData, labeledTestData)
  
if __name__ == '__main__':
    main()
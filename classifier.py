import re
from nltk.tokenize import word_tokenize

# Function to preprocess and tokenize a given message
def clean_message(message):
    message = message.lower()
    message = re.sub(r'[^a-zA-Z0-9\s]', '', message)
    return word_tokenize(message)

# Calculate the smoothed probability of a word in a given category 
def calculate_word_probability(word, word_count, total_words, vocabulary):
    laplace_smoothing = 1
    word_occurrences = word_count[word]
    return (word_occurrences + laplace_smoothing) / (len(total_words) + len(vocabulary))

# Function to classify messages in the test dataset as ham or spam
def classify_messages(test_data, ham_word_count, spam_word_count, ham_words, spam_words, vocabulary):
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

        if p_word_spam_product > p_word_ham_product:
            message_prediction.append("SPAM")
        else:
            message_prediction.append("HAM")
    test_data['predicted_label'] = message_prediction
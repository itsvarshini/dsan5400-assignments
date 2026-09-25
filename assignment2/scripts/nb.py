import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def build_dataframe(folder):
    """
    Takes as input a directory containing presidential speeches and returns two
    DataFrames storing the text from those files, one for the training data
    and one for the test data (unlabeled)
    :param folder: a path to a directory containing presidential speeches
    :return: a tuple of pandas DataFrames
    """
    path = Path(folder)
    df_train = pd.DataFrame(columns=["author"])
    df_test = pd.DataFrame(columns=["author"])
    author_to_id_map = {"kennedy": 0, "johnson": 1}

    def make_df_from_dir(dir_name, df):
        """
        Takes as input directory to construct df from and returns updated df
        :param dir_name: a Path to a directory
        :param df: an empty pandas DataFrame
        :return: updated pandas DataFrames
        """
        rows = []
        for f in path.glob(f"./{dir_name}/*.txt"):
            with open(f) as fp:
                text = fp.read()

                # Create a pandas DataFrame with column "authors" and "text" containing each speech
                if dir_name in ("kennedy", "johnson"):
                    # For labeled speeches
                    author = dir_name
                    #df = pd.concat([df, pd.DataFrame({"author": [author], "text": [text]})], ignore_index = True)
                else:
                    # For unlabeled speeces
                    fname = f.stem
                    author = fname.split("_")[2]
                    #df = pd.concat([df, pd.DataFrame({"author": [author], "text": [text]})], ignore_index = True)
                rows.append({"author": author, "text": text})
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index = True)           
        return df

    for p in path.iterdir():
        if p.name in ("kennedy", "johnson"):
            df_train = make_df_from_dir(p.name, df_train)
        elif p.name == "unlabeled":
            df_test = make_df_from_dir(p.name, df_test)
    # replace the strings for the author names with numeric codes (0, 1)
    df_train["author"] = df_train["author"].apply(lambda x: author_to_id_map.get(x))
    # do the same for the test data
    df_test["author"] = df_test["author"].apply(lambda x: author_to_id_map.get(x))
    return df_train, df_test

# testing code:
# df_train1, df_test1 = build_dataframe("data")
# print(df_train1.head())
# raise SystemExit


def train_nb(df, alpha=0.1):
    """
    Takes as input a pandas DataFrame containing Federalist
    files text to determine priors and likelihoods
    :param df: a pandas DataFrame
    :return: two numpy arrays for the priors and likelihoods
    """
    # Create a dictionary that maps whitespace-separated tokens in the file to a unique index
    vocabulary = {}
    idx = 0
    for text in df["text"]:
        for word in text.split():
            if word not in vocabulary:
                vocabulary[word] = idx
                idx += 1

    # Create variables for the number of documents and the number of classes.
    n_docs = df.shape[0]
    n_classes = df["author"].nunique()
    
    # Compute the priors
    priors = []
    for c in range(n_classes):
        priors.append((df["author"] == c).sum() / n_docs)
    priors = np.array(priors)

    # Create a matrix containing all 0s called training_matrix of size (n_docs, len(vocabulary)), then fill it with the counts of each word for each document. 
    # This is the bag-of-words matrix for all the documents
    training_matrix = np.zeros((n_docs, len(vocabulary)), dtype = int)

    # for each word in each text in df, map its index in vocabulary dict to the same no. column in the matrix, and then add a count for every occurrence
    for idx, text in enumerate(df["text"]):
        for word in text.split():
            j = vocabulary[word]
            training_matrix[idx, j] += 1

    # Get word counts for both classes: for each class label, pick the rows in the training matrix whose value matches the label and calculate the sum of the occurrences of the words (by row)
    word_counts_per_class = np.array([training_matrix[df["author"].values == c].sum(axis=0) for c in range(n_classes)])
    
    # Initialize a matrix to store the likelihoods
    likelihoods = np.zeros((n_classes, len(vocabulary)))

    # Then fill it in using Lidstone smoothing
    # P(word | class) = (count(words in class) + alpha) / (count(total words in class) + (alpha * len(vocabulary)))
    for c in range(n_classes):
        likelihoods[c] = (word_counts_per_class[c] + alpha) / (word_counts_per_class[c].sum() + (alpha * len(vocabulary)))

    return vocabulary, priors, likelihoods

# testing code:
# a = train_nb(df_train1)
# print(x[2])
# raise SystemExit

def test(df, vocabulary, priors, likelihoods):
    """
    Takes as input a pandas DataFrame representing the disputed Federalist
    Papers and returns predictions for every text document
    :param df: a pandas DataFrame
    :return: a numpy array of predictions
    """
    class_predictions = []
    for text in df["text"]:
        test_vector = np.zeros(shape=(len(vocabulary)))

        # Fill test_vector with counts for the words that appear in the vocabulary
        for word in text.split():
            if word in vocabulary:
                test_vector[vocabulary[word]] += 1

        # Compute predictions p(y|text)
        preds = np.zeros(df["author"].nunique())

        # Calculating: likelihood * prior
        for cp in range(len(priors)):
            preds[cp] = np.log(priors[cp]) + np.sum(test_vector * np.log(likelihoods[cp]))

        # Then get your predictions, yhat
        yhat = np.argmax(preds)
        class_predictions.append(yhat)

    return class_predictions

# testing code:
# b = test(df_test1, a[0], a[1], a[2])
# print(b)
# raise SystemExit

def sklearn_nb(training_df, test_df):
    """
    Performs Naive Bayes classification using scikit-learn implementation
    :param training_df: training data
    :param test_df: test data
    :return: predictions
    """
    vectorizer = CountVectorizer()

    # Fit the vectorizer on the training set text
    vectorizer.fit(training_df["text"])

    # Then transform the text using the vectorizer
    training_data = vectorizer.transform(training_df["text"])
    training_data.toarray()

    # Do the same for the test data
    test_data = vectorizer.transform(test_df["text"])
    test_data.toarray()

    nb_classifier = MultinomialNB()
    # Fit the Naive Bayes classifier
    nb_classifier.fit(X = training_data, y = training_df["author"])

    pred_nb = nb_classifier.predict(test_data)
    return pred_nb

# testing code:
# d = sklearn_nb(df_train1, df_test1)
# print(d)
# raise SystemExit

def get_metrics(true, preds):
    """
    Takes gold labels and predictions to compute performance metrics
    :param true: array-like object
    :param preds: array-like object
    :return: a tuple of various performance metrics
    """
    # TODO Compute performance measures
    accuracy = metrics.accuracy_score(true, preds)
    f1_score = metrics.f1_score(true, preds)
    conf_matrix = metrics.confusion_matrix(true, preds)

    return accuracy, f1_score, conf_matrix

# testing code:
# e = get_metrics(true = df_test1["author"], preds = b)
# print(e)
# raise SystemExit

def plot_confusion_matrix(conf_matrix_data, labels):
    """
    Takes as input confusion matrix data from get_metrics() and prints out a
    confusion matrix
    :param conf_matrix_data:
    :return: None
    """
    plt.title("Confusion matrix")
    axis = sns.heatmap(conf_matrix_data)
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    axis.set(xlabel="Predicted", ylabel="True")
    plt.show()
    return

# testing code:
# f = plot_confusion_matrix(e[2], labels = ["kennedy", "johnson"])
# print(f)
# raise SystemExit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive Bayes Algorithm")
    parser.add_argument("-f", "--indir", required=True, help="Data directory")
    args = parser.parse_args()

    training_df, test_df = build_dataframe(args.indir)
    vocabulary, priors, likelihoods = train_nb(training_df)
    class_predictions = test(test_df, vocabulary, priors, likelihoods)
    acc, f1, conf = get_metrics(test_df, class_predictions)
    plot_confusion_matrix(conf, [0, 1])
    sklearn_preds = sklearn_nb(training_df, test_df)
    sklearn_metrics = get_metrics(test_df, sklearn_preds)

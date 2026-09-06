# imports
import argparse
from collections import Counter
from matplotlib import pyplot as plt

def get_ranks_and_frequencies(infile):
    """ 
    Produces a list of rank, frequency pairs for each word in a text file
    :param infile: a textfile
    :return: a list containing rank, frequency, pairs for each word
    """

    # open & read file
    with open(infile) as f:
        contents = f.read()

    # obtain counter (i.e. frequencies for each word in the file)
    c = Counter(contents.split())

    # sort the words by their frequencies and assemble in a list
    freqs = sorted(c.items(), key = lambda i: i[1], reverse = True)

    # create a list that contains the frequencies and their associated rank
    ranks_and_frequencies = [(rank, freq) for rank, (word, freq) in enumerate(freqs, start = 1)]

    # return (rank, frequency) list
    return ranks_and_frequencies

def plot(infile):
    """
    Plots rank and frequency pairs to demonstrate Zipf's Law
    :param infile: a text file
    :return: None, produces a matplotlib plot
    """

    # get the frequencies and their ranks of each word in the text file using above defined function
    ranks_and_frequencies = get_ranks_and_frequencies(infile)

    # plot the ranks and frequencies by word in the text on a log scale
    plt.plot([col[0] for col in ranks_and_frequencies], [col[1] for col in ranks_and_frequencies])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Log of the Rank')
    plt.ylabel('Log of the Frequency')
    plt.title('Rank and Frequency of Words in Given Text')

    # save the plot as a picture 
    #plt.savefig('VarshiniChellapilla_assignment1_plot.png')

    # display plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Constructs a curve demonstrating Zipf\'s Law by plotting a rank, frequency plot.')
    parser.add_argument('--path', type = str, required = True, help = 'Path to file')
    args = parser.parse_args()
    plot(args.path)
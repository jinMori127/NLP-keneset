
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def draw_zipf_law_on_log_log_scale(csv_path, out_path):

    df = pd.read_csv(csv_path, encoding='utf-8')

    # extracting all the sentences from the csv file
    full_text = ' '.join(df['Sentence_text'].dropna())

    # tokenize the text into words with excluding specified punctuation marks
    words = []
    for word in full_text.split():
        just_hebrew_word = word.strip('.,:?"/')
        if just_hebrew_word and not any(char.isdigit() for char in just_hebrew_word):
            words.append(just_hebrew_word)

    # count the frequency of each word
    word_frequency = Counter(words)
    # get the top chosen words
    draw_used_words = word_frequency.most_common(len(word_frequency))

    # extract the words and frequencies for plotting
    top_words, word_frequencies = zip(*draw_used_words)

    # rank the chosen words
    word_ranks = np.arange(1, len(top_words) + 1)

    # use log scale for both x and y axes
    log_ranks = np.log(word_ranks)
    log_frequencies = np.log(word_frequencies)

    # drawing the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(log_ranks, log_frequencies, color='blue', marker='o', label='words')
    plt.plot(log_ranks, log_frequencies, color='red', linestyle='-', label='line connector')
    plt.xlabel('log(rank)')
    plt.ylabel('log(frequency)')
    plt.title('zipf law on log(Rank) log(frequency) scale')
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path)  # save the plot as an image in the file path we get
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)

    input_csv_file = sys.argv[1]
    output_path = sys.argv[2]

    draw_zipf_law_on_log_log_scale(input_csv_file, output_path)

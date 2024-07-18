import math
from collections import Counter
import pandas as pd

class LM_Trigram:
    marks = [".", ",", "?", '"','']  # List of punctuation marks and empty string

    def __init__(self, corpus_path, protocol_type):
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.total_tokens = 0
        self.vocab_size = 0
        self.type_p = protocol_type
        self.corpus_df = pd.read_csv(corpus_path)  # Read corpus data from CSV
        self.calc_grams_corpus()  # Calculate unigrams, bigrams, and trigrams from the corpus

    # Preprocesses text by splitting it into tokens
    def preprocess_text(self, text):
        return text.split(" ")

    # Calculate unigrams, bigrams, and trigrams from the corpus
    def calc_grams_corpus(self):

        for index, row in self.corpus_df.iterrows():
            sentence = row['sentence_text']
            prot_type = row['protocol_type']

            # Update counters based on protocol type
            if self.type_p == prot_type:
                tokens = sentence.split(" ")
                self.unigrams.update(tokens)
                self.bigrams.update(zip(tokens, tokens[1:]))
                self.trigrams.update(zip(tokens, tokens[1:], tokens[2:]))

        # Update total tokens count and vocabulary size
        self.total_tokens = sum(self.unigrams.values())
        self.vocab_size = len(self.unigrams)

    # Calculate log probability of a sentence
    def calculate_prob_of_sentence(self, sentence, smoothing="Laplace", v_lambda=(0.7, 0.2, 0.1)):
        log_prob = self.do_calculate_prob_of_sentence(sentence, smoothing, v_lambda)
        print(f"Log Probability: {log_prob:.3f}")
        return log_prob

    # Calculate the log probability of a sentence from inside
    def do_calculate_prob_of_sentence(self, sentence, smoothing="Laplace", v_lambda=(0.7, 0.2, 0.1)):
        tokens = ["<s>"] + ["<s>"] + sentence.split() + ["<s>"]  # Add start and end tokens
                                                                 # sentece = "" we will calc the prop of an seen tgram (<s>, <s>, <s>)
                                                                 # sentece = "one word","two word" ill calc the prob of an seen tgram (<s>, <s>, word ,<s>)
                                                                 # and this added for the sentences (of len 0,1,2) with no tgram.
                                                                 # as we can see that the probility of <s> will not affect the over all probility
                                                                 # beacause it will be always ~ 0  1/larg_nimber

        log_prob = 0.0

        # Iterate over trigrams in the sentence
        for i in range(2, len(tokens)):
            trigram = (tokens[i - 2], tokens[i - 1], tokens[i])  # Current trigram
            bigram = (tokens[i - 2], tokens[i - 1])  # Previous bigram
            unigram = tokens[i - 2]  # Previous unigram

            # Count trigram, bigram, and unigram
            trigram_count = self.trigrams.get(trigram, 0)
            bigram_count = self.bigrams.get(bigram, 0)
            unigram_count = self.unigrams.get(unigram, 0)

            if smoothing == "Laplace":
                # Laplace smoothing
                trigram_prob = (trigram_count + 1) / (bigram_count + self.vocab_size)
                prob = trigram_prob

            elif smoothing == "Linear":
                # linear smoothing
                trigram_prob = trigram_count / bigram_count if bigram_count > 0 else 0
                bigram_prob = bigram_count / unigram_count if unigram_count > 0 else 0
                unigram_prob = (self.unigrams[tokens[i]] + 1) / (self.total_tokens + self.vocab_size)
                prob = v_lambda[0] * trigram_prob + v_lambda[1] * bigram_prob + v_lambda[2] * unigram_prob
            else:
                # if inserted neither Linear nor Laplace
                raise ValueError(f"You should put Laplace or Linear, {smoothing} is not supported.")

            log_prob += math.log(prob) if prob > 0 else float('-inf')

        return log_prob

    # Generate the next token given a sentence as the word with the highest prob
    def generate_next_token(self, sequence):
        max_prob = float('-inf')
        next_token = None

        # Run over all tokens in the vocabulary
        for token in self.unigrams.keys():
            if token == "</s>" or token in self.marks:
                continue
            sentence = sequence + " " + token
            prob = self.do_calculate_prob_of_sentence(sentence, "Linear")

            if prob > max_prob:
                max_prob = prob
                next_token = token

        # Print and return the most probable next token
        print(next_token)
        return next_token

    # Get top k collocations of size n using PMI
    def get_k_n_collocations(self, k, n):
        if k < 0 or n < 0:
            raise ValueError("Invalid k or n")
        ngrams = Counter()
        count_ngrams = 0

        # Iterate over sentences in the corpus
        for index, row in self.corpus_df.iterrows():
            sentence = row['sentence_text']
            prot_type = row['protocol_type']
            tokens = self.preprocess_text(sentence)

            # Consider sentences of length at least n and matching protocol type
            if len(tokens) >= n and self.type_p == prot_type:
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i + n])
                    ngrams.update([ngram])
                    count_ngrams += 1

        # Calculate PMI for each n-gram
        pmi_scores = {}
        for ngram, count in ngrams.items():
            joint_prob = count / count_ngrams  # Joint probability of n-gram
            # Product of probabilities
            ind_probs_product = math.prod([self.unigrams[word] / self.total_tokens for word in ngram])
            pmi = math.log(joint_prob / ind_probs_product)  # PMI calculation
            pmi_scores[ngram] = pmi

        # Sort n-grams by PMI and return top k
        sorted_ngrams_by_pmi = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        final_list = [token for token, _ in sorted_ngrams_by_pmi]
        return final_list


class SolveMasked:
    numbers_map= {
        2: "Two",
        3: "Three",
        4: "Four"
    }

    # Initialize SolveMasked object and LM_Trigram objects for plenary and committee protocols
    def __init__(self):
        self.tgram_p = LM_Trigram('example_knesset_corpus.csv', "plenary")
        self.tgram_c = LM_Trigram('example_knesset_corpus.csv', "committee")
        self.sentence_list_c = {}
        self.sentence_list_p = {}
        self.tokens_list_c = {}
        self.tokens_list_p = {}

    # Generate masked sentences for plenary
    def masked_sentence_plenary(self):
        with open('masked_sentences.txt', 'r', encoding='utf-8') as file:
            for line in file:
                sentence_to_gen = ""
                list_token = []
                split_in_star = line.strip().split("[*]")  # Split line by "[*]"
                for i in range(len(split_in_star) - 1):
                    sentence_to_gen = sentence_to_gen + " " + split_in_star[i]
                    list_token.append(self.tgram_p.generate_next_token(sentence_to_gen))
                    sentence_to_gen = sentence_to_gen + " " + list_token[-1]

                sentence_to_gen = sentence_to_gen + " " + split_in_star[-1]
                self.tokens_list_p[line] = list_token
                self.sentence_list_p[line] = sentence_to_gen

    # Generate masked sentences for committee
    def masked_sentence_committee(self):
        with open('masked_sentences.txt', 'r', encoding='utf-8') as file:
            for line in file:
                sentence_to_gen = ""
                list_token = []
                split_in_star = line.strip().split("[*]")  # Split line by "[*]"
                for i in range(len(split_in_star) - 1):
                    sentence_to_gen = sentence_to_gen + " " + split_in_star[i]
                    list_token.append(self.tgram_c.generate_next_token(sentence_to_gen))
                    sentence_to_gen = sentence_to_gen + " " + list_token[-1]

                sentence_to_gen = sentence_to_gen + " " + split_in_star[-1]
                self.tokens_list_c[line] = list_token
                self.sentence_list_c[line] = sentence_to_gen

    # Save masked sentences and their probabilities in a file
    def masked_save_in_file(self):
        with open(f'sentences_result.txt', 'w', encoding='utf-8') as f:
            for original in self.sentence_list_c.keys():
                # Write Committee corpus info
                f.write(f"Original sentence: {original}\n")
                f.write(f"Committee sentence: {self.sentence_list_c[original]}\n")
                c_prop_c = self.tgram_c.do_calculate_prob_of_sentence(self.sentence_list_c[original])
                c_prop_p = self.tgram_p.do_calculate_prob_of_sentence(self.sentence_list_c[original])
                f.write(f"Committee tokens: {self.tokens_list_c[original]}\n")
                f.write(f"Probability of committee sentence in committee corpus: {c_prop_c}\n")
                f.write(f"Probability of committee sentence in plenary corpus: {c_prop_p}\n")
                c_appear = "committee" if c_prop_c > c_prop_p else "plenary"
                f.write(f"This sentence is more likely to appear in corpus: {c_appear}\n")

                # Write Plenary corpus info
                f.write(f"Plenary sentence: {self.sentence_list_p[original]}\n")
                p_prop_c = self.tgram_c.do_calculate_prob_of_sentence(self.sentence_list_p[original])
                p_prop_p = self.tgram_p.do_calculate_prob_of_sentence(self.sentence_list_p[original])
                f.write(f"Committee tokens: {self.tokens_list_p[original]}\n")
                f.write(f"Probability of plenary sentence in plenary corpus: {p_prop_p}\n")
                f.write(f"Probability of plenary sentence in committee corpus: {p_prop_c}\n")
                p_appear = "plenary" if p_prop_p > p_prop_c else "committee"
                f.write(f"This sentence is more likely to appear in corpus: {p_appear}\n\n")

    # Get k-n collocations and save them in a file
    def get_k_n_collocations_save_in_file(self, list_k, list_n):
        with open(f'knesset_collocation.txt', 'w', encoding='utf-8') as f:
            for k, n in zip(list_k, list_n):
                try:
                    c_ngrams_by_pmi = self.tgram_c.get_k_n_collocations(k, n)
                    p_ngrams_by_pmi = self.tgram_p.get_k_n_collocations(k, n)
                except:
                    print("error out of range")

                f.write(f"{self.numbers_map[n]}-gram collocations\n")
                f.write(f"Committee corpus:\n")
                for i in range(k):
                    f.write(f"{c_ngrams_by_pmi[i]}\n")

                f.write(f"Plenary corpus:\n")
                for i in range(k):
                    f.write(f"{p_ngrams_by_pmi[i]}\n")

                f.write(f"\n")


if __name__ == "__main__":
    solve = SolveMasked()
    solve.masked_sentence_plenary()  # Generate masked sentences for plenary protocol
    solve.masked_sentence_committee()  # Generate masked sentences for committee protocol
    solve.masked_save_in_file()  # Save masked sentences and their probabilities in a file
    solve.get_k_n_collocations_save_in_file([10, 10, 10], [2, 3, 4])  # Get k-n collocations and save them in a file

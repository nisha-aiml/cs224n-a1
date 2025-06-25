#!/usr/bin/env python

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join('..')))

def distinct_words(corpus):
    
    corpus_words = []
    num_corpus_words = 0

    # ### START CODE HERE ###
    corpus_words = sorted(set(word for sentence in corpus for word in sentence))
    num_corpus_words = len(corpus_words)
    # ### END CODE HERE ###

    return corpus_words, num_corpus_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    #Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    words, num_words = distinct_words(corpus)
    M = np.zeros((num_words, num_words))
    word2Ind = {word: i for i, word in enumerate(words)}

    for sentence in corpus:
        for i, center_word in enumerate(sentence):
            center_idx = word2Ind[center_word]
            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                context_word = sentence[j]
                context_idx = word2Ind[context_word]
                M[center_idx][context_idx] += 1
    # ### END CODE HERE ###

    return M, word2Ind

def reduce_to_k_dim(M, k=2):
    #Reduce a co-occurrence count matrix to k-dim using Truncated SVD \"\"\"
    np.random.seed(4355)
    n_iter = 10
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    # ### START CODE HERE ###
    svd = TruncatedSVD(n_components=k, n_iter=10)
    M_reduced = svd.fit_transform(M)
    # ### END CODE HERE ###

    print("Done.")
    return M_reduced

def main():
    matplotlib.use('agg')
    plt.rcParams['figure.figsize'] = [10, 5]

    assert sys.version_info[0] == 3
    assert sys.version_info[1] >= 5

    def plot_embeddings(M_reduced, word2Ind, words, title):
        for word in words:
            idx = word2Ind[word]
            x = M_reduced[idx, 0]
            y = M_reduced[idx, 1]
            plt.scatter(x, y, marker='x', color='red')
            plt.text(x, y, word, fontsize=9)
        plt.savefig(title)

    reuters_corpus = read_corpus()

    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
    M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis]

    words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
    plot_embeddings(M_normalized, word2Ind_co_occurrence, words, 'co_occurrence_embeddings_(soln).png')

if __name__ == "__main__":
    main()

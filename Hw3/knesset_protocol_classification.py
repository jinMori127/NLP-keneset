import numpy as np 
import pandas as pd 
import time
import random
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def devide_to_classes(corpus_path):
    corpus_df = pd.read_csv(corpus_path)
    return corpus_df.groupby('protocol_type')['sentence_text'].apply(list).to_dict()


def sentences_classification(sentence_map):
    devided_chunks_map = {}
   
    c_sentences = sentence_map['committee']
    p_sentences = sentence_map['plenary']
    # put each five sentences in a chunck and delete the rest of them
    c_chunks = [c_sentences[x:x+5] for x in range(0, len(c_sentences) - len(c_sentences) % 5, 5)]
    p_chunks = [p_sentences[x:x+5] for x in range(0, len(p_sentences) - len(p_sentences) % 5, 5)]

    devided_chunks_map['committee'] = c_chunks
    devided_chunks_map['plenary'] = p_chunks
    return devided_chunks_map


def down_sampling(devided_chunks_map):
    random.seed(42)

    down_sampled_sentences_map = devided_chunks_map
    b_prot_type, s_prot_type = ('committee', 'plenary') if len(devided_chunks_map['committee']) > len(devided_chunks_map['plenary']) else ('plenary', 'committee')
    big_class_len , small_class_len = len(devided_chunks_map[b_prot_type]), len(devided_chunks_map[s_prot_type])
    print(f"before down sampling {b_prot_type} len: {big_class_len} and {s_prot_type} with len {small_class_len}")
    diff = big_class_len - small_class_len

    if diff == 0 :
        return devided_chunks_map
    # delete chunck as the gab size 
    for _ in range(diff):
        rand_indx = random.randint(0, len(down_sampled_sentences_map[b_prot_type]) - 1)
        del down_sampled_sentences_map[b_prot_type][rand_indx]

    return down_sampled_sentences_map


def build_feature_vector(down_sampled_sentences_map):
    vectorizer = TfidfVectorizer()
    sentence_list = []
    labels = []

    for prot_type, chunks in down_sampled_sentences_map.items():
        for chunk in chunks:
            # we represented each chunck with five sentences so we need to combine them 
            sentence_list.append(' '.join(chunk))
            labels.append(prot_type)

    # create feature feature
    features = vectorizer.fit_transform(sentence_list)
    features_name = vectorizer.get_feature_names_out()

    return features, labels, features_name,vectorizer


def m_build_feature_vector(down_sampled_sentences_map):
    labels = []
    avg_length = []
    has_number = []  
    has_a_dash = []
    has_kenneset = []
    for prot_type, chunks in down_sampled_sentences_map.items():
        for chunk in chunks:
            sentence = ' '.join(chunk)
            labels.append(prot_type)

            avg_len = np.mean([len(sentence.split(' ')) for sentence in chunk])
            avg_length.append(avg_len)
            # check if there is a number in the sentence
            value = 1 if re.search(r'\d', sentence) else 0  
            has_number.append(value)  

            value = 1 if '–' in sentence else 0  
            has_a_dash.append(value)

            value = 1 if 'כנסת' in sentence else 0  
            has_kenneset.append(value)

    # reshape the new features to fit in hstack
    avg_length = np.array(avg_length).reshape(-1, 1)
    has_number = np.array(has_number).reshape(-1, 1)
    has_a_dash = np.array(has_a_dash).reshape(-1, 1)
    has_kenneset= np.array(has_kenneset).reshape(-1, 1)

    # hstack new features
    features = np.hstack([avg_length, has_number,  has_kenneset])
    features_name = ['avg_length', 'has_number',  'has_kenneset']

    return features, labels, features_name


def test_cross_split_acc(features_M, labels, N_NUMBER, string_to_print, kernel_type):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    train_1, test_1, train_y, test_y = train_test_split(features_M, y, test_size=0.1, stratify=y, random_state=42)
    svm = SVC(kernel=kernel_type)
    knn= KNeighborsClassifier(N_NUMBER, n_jobs=-1)

    # 10 fold cross validation way 
    cv = StratifiedKFold(n_splits=10)
    print(f"{string_to_print}\n start of 10 fold cross validation:")
    knn_accuracy = cross_val_score(knn, features_M, y, cv=cv)
    svm_accuracy = cross_val_score(svm, features_M, y, cv=cv)
    print(f"knn Cross Validation accuracies: {knn_accuracy*100}")
    print(f"svm Cross Validation accuracies: {svm_accuracy*100}")

    # Train test split evaluation
    print(f"start of Train test split:")
    knn.fit(train_1, train_y)
    svm.fit(train_1, train_y)

    t_knn_accuracy = knn.score(test_1, test_y)
    t_svm_accuracy = svm.score(test_1, test_y)
    print(f"knn Test accuracy: {t_knn_accuracy*100}%")
    print(f"svm Test accuracy: {t_svm_accuracy*100}%")

    return knn,svm,label_encoder

if __name__ == "__main__":
    start_time = time.time()

    sentences_map = devide_to_classes('example_knesset_corpus.csv')
    devided_chunks_map = sentences_classification(sentences_map)
    down = down_sampling(devided_chunks_map)

    print(f"afte the down sampling {len(down["committee"])}\n")

    # build the two feature vectors
    features_M, labels, feature_names, vectorizer = build_feature_vector(down)
    m_features_M, m_labels, m_feature_names = m_build_feature_vector(down)

    knn, svm, label_encoder =test_cross_split_acc(features_M,labels,11,"TfidfVectorizer vector:", 'linear')
    m_knn, m_svm, m_label_encoder = test_cross_split_acc(m_features_M,m_labels,15,"my feature vector:", 'linear')

    # after we saw the two result will take the vector that gives the best accuracy 
    # in our case it is my feature vector with svm
    with open('knesset_text_chunks.txt', 'r', encoding='utf-8') as f:
        senteces = f.readlines()

    line_sentences = [sentence.strip() for sentence in senteces]


    v_features = vectorizer.transform(line_sentences)
    
    predictions = svm.predict(v_features)
    pred_labels = label_encoder.inverse_transform(predictions)

    with open(f'classification_results.txt', 'w', encoding='utf-8') as f:
        for label in pred_labels:
            f.write(f"{label}\n")

    end_time = time.time()

    total_time = end_time - start_time
    print(f"total time: {total_time/60} minutes")
import numpy as np 
import pandas as pd 
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
import sys 

def devide_to_classes(corpus_path):
    corpus_df = pd.read_csv(corpus_path)
    return corpus_df.groupby('protocol_type')['sentence_text'].apply(list).to_dict()


def sentences_classification(sentence_map, chunk_size):
    devided_chunks_map = {}
   
    c_sentences = sentence_map['committee']
    p_sentences = sentence_map['plenary']
    # put each five sentences in a chunck and delete the rest of them
    c_chunks = [c_sentences[x:x+chunk_size] for x in range(0, len(c_sentences) - len(c_sentences) % chunk_size, chunk_size)]
    p_chunks = [p_sentences[x:x+chunk_size] for x in range(0, len(p_sentences) - len(p_sentences) % chunk_size, chunk_size)]

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


def embadding_chunks(down_sampled_sentences_map, model):
    # embadd each chunk like we did in section b sum(v_i)/k i0...k
    sentence_map_emb = {'committee': [], 'plenary': []}
    for prot_type, chunks in down_sampled_sentences_map.items():
        for chunk in chunks:
            sentence =' '.join(chunk)
            tokenized_sentence = sentence.split(' ')
            clean_words = [word for word in tokenized_sentence if word in model.wv.key_to_index]
            
            words_vector = np.array([model.wv[word] for word in clean_words])
            sentence_vector = np.mean(words_vector, axis=0)
            sentence_map_emb[prot_type].append(sentence_vector)
    return sentence_map_emb

def build_embedding_feature_vector(sentence_map_emb):
    # using the embadded chunks to create our feature vector 
    features = []
    labels = []

    for prot_type, embeddings in sentence_map_emb.items():
        for embedding in embeddings:
            features.append(embedding)
            labels.append(prot_type)

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


def test_cross_split_acc(features_M, labels, N_NUMBER, string_to_print):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    train_1, test_1, train_y, test_y = train_test_split(features_M, y, test_size=0.1, stratify=y, random_state=42)
    knn= KNeighborsClassifier(N_NUMBER, n_jobs=-1)
 
    # Train test split evaluation
    print(f"start of Train test split using {string_to_print}:")
    knn.fit(train_1, train_y)

    t_knn_accuracy = knn.score(test_1, test_y)
    print(f"knn Test accuracy: {t_knn_accuracy*100}%")

    test_predictions = knn.predict(test_1)

    report = classification_report(test_y, test_predictions, target_names=label_encoder.classes_)
    print(report)

    return knn,label_encoder

def prepare_model_for_chunck(sentences_map,model,chunk_size):
    devided_chunks_map = sentences_classification(sentences_map,chunk_size)
    down = down_sampling(devided_chunks_map)

    print(f"after the down sampling {len(down["committee"])}\n")
    emd_chunk = embadding_chunks(down,model)
    # build the feature vector
    features_M, labels = build_embedding_feature_vector(emd_chunk)

    knn, label_encoder =test_cross_split_acc(features_M,labels,31,"embbading feature vectore")
    return knn, label_encoder

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provode all the required arguments run file corpus_path and model_path.")
        sys.exit(1)
        
    corpus_path, model_path = sys.argv[1], sys.argv[2]
    sentences_map = devide_to_classes(corpus_path)
    model = Word2Vec.load(model_path)

    for chunk_size in [1,3,5]:
        print(f"\nthe results for chunk size: {chunk_size}\n")
        prepare_model_for_chunck(sentences_map, model, chunk_size)


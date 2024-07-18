from gensim.models import Word2Vec
import pandas as pd 
import re 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys


words = ['ישראל', 'כנסת', 'ממשלה', 'חבר', 'שלום', 'שולחן'] # define the set of words that need to
                                                                # find for it the top most similar words 

clean_text_set = re.compile(r'^[./\'"\-,\!?]$|^\d+$|^-$|^\.\.\.$')
clean_words_with_numbers = re.compile(r'.*\d.*')


# define a list of the 10 sentences that we will get other similar sentence for it 
list_of_ten_sentences = ['עמדו פה חברי כנסת , הציגו מוצג מסוים , והוציאו אותם , הורידו אותם גם מעל הדוכן .',  
                         'הכלל הזה חל על הגשת מסמכים לבית המשפט ולערכאות שיפוטיות , הוא לא חל על הגשת מסמכים לטאבו או למקומות אחרים .',
                         'המשמעות היא , שבעצם בתוך חודשיים - שלושה חודשים יגיע קיצוץ נוסף משמעותי במשרדי הממשלה לפה , אל שולחן הכנסת .',
                         'אנחנו נלחמים גם ב- 2011 כדי להגיע למצב הזה אבל אנחנו עדיין לא שם .',
                         'כמו כן מוצע להוסיף , שלא יהיה תוקף לוויתור על זכויות מצד הרוכש , לרבות בכתב , בקשר לאי - התאמות או אי - התאמות יסודיות בדירה שמהוות תנאי לקבלת החזקה בדירה .',
                         'זה חוק חסינות חברי הכנסת , זכויות וחובות , אדוני שר המשפטים .',
                         'אני חושב שאנחנו חיים במשטר משפטי אשר בו זכויות היוצרים והמבצעים אינן מוגנות מספיק , ולכן יש חשיבות להצעת החוק .',
                         'אני מזמין את כבוד שר המשפטים להציג לפנינו את ההצעה .',
                         'נכון , לאורך תקופות ארוכות היו המון חברי כנסת שהגישו נגדם בקשה להסרת החסינות , והם עצמם הסירו את החסינות .',
                         'אני קוראת לכם לתמוך בהצעה החיונית הזאת .']

'''
    part a: 
    -------------------------------
'''

# section a+b:
def preb_knesset(c_path, output_dir):
    corpus_df = pd.read_csv(c_path, encoding='utf-8')

    tokenized_sentences = []
    for index, row in corpus_df.iterrows():
            sentence = row['sentence_text']
            tokenized_sentence = sentence.split(' ')

            clean_tokenized_sentence = [word for word in tokenized_sentence if not clean_text_set.match(word) and not clean_words_with_numbers.match(word)]
            tokenized_sentences.append(clean_tokenized_sentence)

    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1)
    path = output_dir + "knesset_word2vec.model"
    model.save(path)

'''
    part b: 
    -------------------------------
'''

# section a:
def similarity_words(model, output_dir):
    path = output_dir + 'knesset_similar_words.txt'
    with open(path, 'w', encoding='utf-8') as f:

      for word in words:
        f.write(f"{word}: ")
        if word in model.wv.key_to_index:  
            top5_similar_words = model.wv.most_similar(word, topn=5) # using directly the most similar function to get all the five word in one go 
            word_num = 0
            for similar_word, similarity_score in top5_similar_words:
                word_num +=1
                f.write(f"({similar_word},{similarity_score:.4f})")  # write the word to the txt file 
                if word_num != 5:
                    f.write(f",")
            
            f.write(f"\n")        
        else:
            print(f"'{word}' is not in the model's vocabulary.\n")


# section b+c:
def corpus_sentence_embeddings(corpus_path, model, output_dir):
    corpus_df = pd.read_csv(corpus_path, encoding='utf-8')

    sentence_embading_map = {}
    for index, row in corpus_df.iterrows():
        sentence = row['sentence_text']
        tokenized_sentence = sentence.split(' ')

        # extraction just the word that in our model in this we are cleaning the sentence
        clean_tokens = [word for word in tokenized_sentence if word in model.wv.key_to_index]

        words_vector = np.array([model.wv[word] for word in clean_tokens])
        sentence_vector = np.mean(words_vector, axis=0)
        sentence_embading_map[sentence] = sentence_vector
        
    path = output_dir + 'knesset_similar_sentences.txt'
    with open(path, 'w', encoding='utf-8') as f:

        for sentence in list_of_ten_sentences:       
            sentence_embading_map_values = np.array(list(sentence_embading_map.values()))

            sentence_similarity = cosine_similarity([sentence_embading_map[sentence]], sentence_embading_map_values)[0]

            similar_sentence_index = sentence_similarity.argsort()[-2]  

            most_similar_sentence = list(sentence_embading_map.keys())[similar_sentence_index]
            f.write(f"{sentence}: most similar sentence: {most_similar_sentence}\n")
    return sentence_embading_map


#section d:
sentences = ['ברוכים הבאים , הכנסו בבקשה לחדר','אני מוכנה להאריך את ההסכם באותם תנאים.','בוקר טוב , אני  פותח את הישיבה .','שלום , הערב התבשרנו שחברינו  היקר  לא ימשיך איתנו  בשנה הבאה . ']
word2replace = ['לחדר','מוכנה','ההסכם','טוב','פותח','שלום','היקר','בשנה']

lpositive_words = [["לבית"],["מוכנה","יכלה"],["ההסכם","חוזה"],["טוב","צהריים","אופטימי"],["מתחיל",'פותח'],["ברכה","לכולם"],["המכובד","ביותר",'היקר'],["בשנה"]]
lnegative_words = [[],["גבר"],["רבים"],[],["סוגר"],[],["זול"],[]]


def replace_good_word(model, output_dir):

    for positive_words, negative_words in zip(lpositive_words, lnegative_words):
        top3_similar_word = model.wv.most_similar(positive=positive_words, negative=negative_words, topn=3)        
        print(top3_similar_word)
    path = output_dir + 'red_words_sentences.txt'

    # here we will be choosing the best word out of the top 3 word we get
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{sentences[0]}: ")
        mod_sen = sentences[0].replace('לחדר', 'לבתי' )
        f.write(f"{mod_sen}\n")

        f.write(f"{sentences[1]}: ")

        mod_sen =sentences[1].replace('מוכנה', 'יכולה' )
        mod_sen = mod_sen.replace('ההסכם', 'ההסדר' )
        f.write(f"{mod_sen}\n")

        f.write(f"{sentences[2]}: ")

        mod_sen = sentences[2].replace('טוב', 'בריא' )
        mod_sen = mod_sen.replace('פותח', 'ממשיך' )
        f.write(f"{mod_sen}\n")
        
        f.write(f"{sentences[3]}: ")
       
        mod_sen= sentences[3].replace('שלום', 'אוקיי' )
        mod_sen=mod_sen.replace('היקר', 'הראוי' )
        mod_sen=mod_sen.replace('בשנה', 'השנה' )
        f.write(f"{mod_sen}\n")


if __name__=="__main__":

    if len(sys.argv) != 3:
        print("Usage: python knesset_word2vec_classification.py <path/to/a_corpus_csv_file.csv> <path/to/Knesset_word2vec.model>")
        sys.exit(1)
    preb_knesset('example_knesset_corpus.csv')

    corpus_path, output_dir = sys.argv[1], sys.argv[2]
    path = output_dir + "knesset_word2vec.model"
    model = Word2Vec.load(path)

    similarity_words(model, output_dir)
    corpus_sentence_embeddings(corpus_path, model, output_dir)
    replace_good_word(model, output_dir)
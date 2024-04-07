
# topic_identification.py

import gensim.corpora as corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
import spacy
import nltk

# Download stopwords from NLTK
nltk.download('stopwords')

# Load spacy model for lemmatization
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def preprocess_texts(documents):
    stop_words = stopwords.words('english')
    texts = []
    for document in documents:
        doc = nlp(document)
        texts.append([token.lemma_ for token in doc if token.lemma_ not in stop_words and token.lemma_.isalpha()])
    return texts

def build_lda_model(documents, num_topics=5):
    # Preprocess documents
    texts = preprocess_texts(documents)
    
    # Create Dictionary
    id2word = corpora.Dictionary(texts)
    
    # Create Corpus
    texts = [id2word.doc2bow(text) for text in texts]
    
    # Build LDA model
    lda_model = LdaModel(corpus=texts,
                         id2word=id2word,
                         num_topics=num_topics, 
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)
    
    return lda_model, id2word, texts

def display_topics(lda_model, id2word, num_words=10):
    topics = lda_model.print_topics(num_topics=-1, num_words=num_words)
    for topic in topics:
        print(topic)

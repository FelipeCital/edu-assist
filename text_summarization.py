
# text_summarization.py

from langdetect import detect
import spacy
import networkx as nx
from googletrans import Translator
from collections import Counter

import fitz  # PyMuPDF
from docx import Document

# Ensure spaCy models for desired languages are downloaded
MODEL_NAMES = {
    'en': 'en_core_web_sm',
    'es': 'es_core_news_sm',
    'fr': 'fr_core_news_sm',
    'de': 'de_core_news_sm',
    'pt': 'pt_core_news_sm',
}

MODELS = {lang: spacy.load(name) for lang, name in MODEL_NAMES.items()}

translator = Translator()

def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Error detecting language: {e}")
        return None

def text_rank_summarize(text, nlp_model, n_sentences=3):
    doc = nlp_model(text)
    sentences = [sent for sent in doc.sents]
    sentence_vectors = [nlp_model(sent.text).vector for sent in sentences]

    similarity_matrix = nx.Graph()
    for i, vector in enumerate(sentence_vectors):
        for j, other_vector in enumerate(sentence_vectors):
            if i != j:
                similarity = vector @ other_vector
                if similarity > 0.5:
                    similarity_matrix.add_edge(i, j, weight=similarity)

    scores = nx.pagerank(similarity_matrix)
    ranked_sentences = sorted(scores, key=scores.get, reverse=True)
    summary = " ".join(sentences[idx].text for idx in ranked_sentences[:n_sentences])
    return summary

def extract_keywords(nlp_model, text, num_keywords=10):
    doc = nlp_model(text)
    nouns = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    noun_freq = Counter(nouns)
    keywords = noun_freq.most_common(num_keywords)
    return keywords

def display_named_entities(nlp_model, text):
    doc = nlp_model(text)
    unique_entities = set((ent.text, ent.label_) for ent in doc.ents)
    return sorted(list(unique_entities), key=lambda x: x[0])

def translate_text(text, dest_language):
    translation = translator.translate(text, dest=dest_language)
    return translation.text


def extract_text_from_pdf(pdf_content):
    text = ""
    with fitz.open(stream=pdf_content, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(docx_content):
    text = ""
    doc = Document(docx_content)
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text.strip()

def summarize_text(text_input, num_sentences=3):
    # Detecting the language of the text
    language = detect_language(text_input)
    if not language or language not in MODELS:
        raise Exception("Unsupported language or language could not be detected.")

    nlp_model = MODELS[language]
    # Using the existing text_rank_summarize function
    summary = text_rank_summarize(text_input, nlp_model, n_sentences=num_sentences)
    return summary
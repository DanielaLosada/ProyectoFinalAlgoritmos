import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from joblib import dump, load

from Normalizacion import preprocess

# Función para calcular la similitud TF-IDF y guardar el modelo
def tfidf_similarity(abstracts, save_model=True):
    processed_abstracts = [' '.join(preprocess(ab)) for ab in abstracts]  # Preprocesar abstracts
    vectorizer = TfidfVectorizer(
        max_features=5000,       # Limitar el vocabulario a las 5K palabras más frecuentes
        ngram_range=(1, 3),      # Incluir unigramas, bigramas y trigramas
        stop_words='english'     # Eliminar stopwords (redundante con preprocess, pero útil)
    )
    tfidf_matrix = vectorizer.fit_transform(processed_abstracts)  # Calcular matriz TF-IDF
    
    #if save_model:
     #   dump(vectorizer, "C:/Users/dani4/OneDrive/Escritorio/Ws8/ProyectoFinalAlgoritmos/Data/DataAlgoritmos/TFIDF_model.joblib")
    
    return cosine_similarity(tfidf_matrix)

# Función para calcular la similitud Doc2Vec
def doc2vec_similarity(abstracts, save_model=True):
    tagged_data = [TaggedDocument(preprocess(ab), [str(i)]) for i, ab in enumerate(abstracts)]  # Etiquetar documentos
    model = Doc2Vec(
        vector_size=100,
        min_count=2,
        epochs=20,
        dm=1,
        workers=4
    )
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    
   # if save_model:
    #    model.save("C:/Users/dani4/OneDrive/Escritorio/Ws8/ProyectoFinalAlgoritmos/Data/DataAlgoritmos/Doc2vec_model.model")
    
    top_n = 1000
    similarity_matrix = []
    for i in range(len(abstracts)):
        sims = model.dv.most_similar(str(i), topn=top_n)
        similarity_matrix.append(sims)
    
    return similarity_matrix

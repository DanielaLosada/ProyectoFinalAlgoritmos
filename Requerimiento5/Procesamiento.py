import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer

from Algoritmos import tfidf_similarity, doc2vec_similarity 
from Normalizacion import load_bibtex, preprocess

# Función para comparar modelos y guardar resultados
# Esta función compara los modelos de similitud TF-IDF y Doc2Vec
def compare_models_and_save(abstracts, top_k=5, tfidf_similarity_func=None, doc2vec_similarity_func=None):
    # Calcular similitudes
    tfidf_sim = tfidf_similarity_func(abstracts)
    doc2vec_sim = doc2vec_similarity_func(abstracts)

    # Determinar el abstract con más coincidencias (mayor suma de similitudes)
    tfidf_sums = tfidf_sim.sum(axis=1)
    doc2vec_sums = np.array([sum(sim[1] for sim in doc_sim) for doc_sim in doc2vec_sim])

    # Suma total de todas las similitudes
    tfidf_total_sum = tfidf_sums.sum()  # Suma de todas las similitudes en TF-IDF
    doc2vec_total_sum = doc2vec_sums.sum()  # Suma de todas las similitudes en Doc2Vec

    # Cantidad de documentos
    total_docs = len(abstracts)

    # Porcentajes totales de similitud
    porcentaje_totaltfidf = (tfidf_total_sum / (total_docs * (total_docs - 1))) * 100
    porcentaje_total2vec = (doc2vec_total_sum / (total_docs * (total_docs - 1))) * 100

    most_similar_tfidf_index = np.argmax(tfidf_sums)
    most_similar_doc2vec_index = np.argmax(doc2vec_sums)
 

    # Resumen en consola
    print("\nTF-IDF:")
    print(f"Abstract con más coincidencias (índice {most_similar_tfidf_index}):")
    print(abstracts[most_similar_tfidf_index][:200] + "...")
    print(f"Suma total de similitudes: {tfidf_sums[most_similar_tfidf_index]:.3f}")
    # Mostrar el porcentaje total de similitud
    print(f"Porcentaje total de similitud: {porcentaje_totaltfidf:.2f}%")
    

    print("\nDoc2Vec:")
    print(f"Abstract con más coincidencias (índice {most_similar_doc2vec_index}):")
    print(abstracts[most_similar_doc2vec_index][:200] + "...")
    print(f"Suma total de similitudes: {doc2vec_sums[most_similar_doc2vec_index]:.3f}")
    # Mostrar el porcentaje total de similitud
    print(f"Porcentaje total de similitud: {porcentaje_total2vec:.2f}%")

    # TF-IDF: Guardar en formato DataFrame
    tfidf_df = pd.DataFrame(tfidf_sim, columns=[f"Abstract {i}" for i in range(len(abstracts))])
    tfidf_df.insert(0, "Abstract", [f"Abstract {i}" for i in range(len(abstracts))])
    #tfidf_df.to_csv(tfidf_csv, index=False)

    # Doc2Vec: Guardar similitudes
    doc2vec_data = []
    for i, sims in enumerate(doc2vec_sim):
        for idx, sim in sims:
            doc2vec_data.append({
                "Abstract ID": f"Abstract {i}",
                "Similar Abstract ID": f"Abstract {int(idx)}",
                "Similarity": sim
            })

    doc2vec_df = pd.DataFrame(doc2vec_data)
    #doc2vec_df.to_csv(doc2vec_csv, index=False)

    print(f"\nSimilitudes TF-IDF guardadas en Requerimiento 5: ")
    print(f"Similitudes Doc2Vec guardadas en Requerimiento 5: ")


# Funcion encargada de calcular la similitud TF-IDF por lotes de 500
# para evitar problemas de memoria y mejorar la velocidad de cálculo
def batch_tfidf_similarity(abstracts, batch_size=500):
    """Calcula similitudes TF-IDF por lotes para ahorrar memoria."""
    tfidf_matrices = [] # Lista para almacenar las matrices de similitud por lotes
    for i in range(0, len(abstracts), batch_size): # Se divide el procesamiento en lotes
        # Procesar cada lote de abstracts
        batch = abstracts[i:i + batch_size]
        processed_batch = [' '.join(preprocess(ab)) for ab in batch] # Preprocesar los abstracts
        # Calcular la matriz TF-IDF para el lote
        tfidf_matrix = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english').fit_transform(processed_batch)
        tfidf_matrices.append(tfidf_matrix)

    # Combina los lotes en una matriz grande
    tfidf_combined = vstack(tfidf_matrices)
    return cosine_similarity(tfidf_combined)


# Funciones auxiliares para clustering
def calculate_clusters(similarity_matrix, cutoff_distance):
    """
    Calcula los clusters usando una matriz de similitud y una distancia de corte.
    """
    # Convertir similitud a distancia
    distance_matrix = 1 - similarity_matrix
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    distance_matrix[distance_matrix < 0] = 0
    np.fill_diagonal(distance_matrix, 0)
    
    # Generar linkage
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')
    
    # Generar clusters
    cluster_labels = fcluster(linkage_matrix, t=cutoff_distance, criterion='distance')

    clusters = {}
    for doc_id, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(doc_id)

    #mostrar el tamaño de los clusters
    print("Número de clusters generados:", len(clusters))
    #cluster_sizes = {k: len(v) for k, v in clusters.size()}
    #print("Distribución de tamaños de clusters:", cluster_sizes)


    return clusters, linkage_matrix
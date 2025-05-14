import nltk
import time


from Normalizacion import parse_large_bib, load_bibtex, preprocess
from Algoritmos import tfidf_similarity, doc2vec_similarity
from Procesamiento import batch_tfidf_similarity, calculate_clusters, compare_models_and_save
from Almacenados import save_cluster_summary_to_csv


nltk.download('stopwords') # Descargar stopwords
# WordNet es una base de datos léxica del inglés que agrupa palabras en conjuntos de sinónimos (synsets), 
# proporcionando definiciones y ejemplos de uso.
nltk.download('wordnet') # Descargar wordnet


def main():
    # Cargar y procesar el archivo BibTeX
    file_path = "C:/Users/dani4/OneDrive/Escritorio/Ws8/ProyectoFinalAlgoritmos/Data/unificados.bib"
    abstracts = load_bibtex(file_path)
    print(f"Cargando y procesando {len(abstracts)} abstracts...")
    

    # Procesamiento por lotes batch para los 11k abstracts
    # Procesar similitud por lotes y guardar resultados
    print("Calculando similitud TF-IDF por lotes...")
    start_time = time.time() # Guardar tiempo de inicio
    similarity_matrix = batch_tfidf_similarity(abstracts, batch_size=500)
    #save_batch_results(similarity_matrix, 0, output_dir)
    end_time = time.time()
    print(f"Tiempo de cálculo por lotes: {end_time - start_time:.2f} segundos")

    # Calcular clusters en toda la matriz
    print("\nCalculando clusters...")
    cutoff_distance = 0.8  # Parámetro de corte para definir los clusters
    clusters, _ = calculate_clusters(similarity_matrix, cutoff_distance)

    # Guardar resumen de clusters en CSV
    print("Guardando resumen de clusters")
    save_cluster_summary_to_csv(clusters, abstracts)
    
    # Comparar resultados para un abstract específico
    print("\nComparando modelos...")
    #tiempo de comparacion
    start_time = time.time()
    #compare_models(abstracts, doc_index=0, top_k=10)
    compare_models_and_save(abstracts, top_k=10, tfidf_similarity_func=tfidf_similarity, doc2vec_similarity_func=doc2vec_similarity)
    end_time = time.time()
    print(f"\nTiempo de comparación: {end_time - start_time:.2f} segundos")
    

if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd

# Función para guardar el resumen de clusters en un archivo CSV
def save_cluster_summary_to_csv(clusters, abstracts, output_file='C:/Users/dani4/OneDrive/Escritorio/Ws8/ProyectoFinalAlgoritmos/Data/DataAlgoritmos/Cluster_summary.csv'):
    # Crear una lista para almacenar los datos
    cluster_data = []

    # Recorrer cada cluster y resumir los datos
    for cluster_id, abstract_indices in clusters.items():
        cluster_texts = [abstracts[i] for i in abstract_indices]
        cluster_summary = {
            "Cluster ID": cluster_id,
            "Número de Abstracts": len(abstract_indices),
            "Ejemplo de Abstract": cluster_texts[0] if cluster_texts else ""
        }
        cluster_data.append(cluster_summary)
    
    # Crear un DataFrame y guardarlo como CSV
    df = pd.DataFrame(cluster_data)
    df.to_csv(output_file, index=False)
    print(f"Resumen de clusters guardado en {output_file}")

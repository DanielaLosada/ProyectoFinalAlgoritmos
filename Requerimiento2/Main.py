import pandas as pd
import sys
import os


# Agregar la carpeta raíz al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Requerimiento2.Graficos import generate_and_save_charts
from Requerimiento2.LimpiezaNormalizacion import normalize_authors, clean_journal_name, normalize_product_type, parse_large_bib
from Requerimiento2.Estadisticas import generate_statistics, save_statistics

def main():
    # Procesamiento del archivo
    file_path = "C:/Users/dani4/OneDrive/Escritorio/Ws8/ProyectoFinalAlgoritmos/Data/unificados.bib"
    print(f"Procesando archivo: {file_path}")
    
    # Paso 1: Parsear el archivo BibTeX y crear el DataFrame
    entries = parse_large_bib(file_path)
    df = pd.DataFrame(entries)  # Aquí se define el DataFrame df
    
    # Paso 2: Normalización de datos
    print("\nNormalizando datos...")
    df['author'] = df['author'].apply(normalize_authors)
    df['tipo_normalizado'] = df['tipo'].apply(normalize_product_type)  # Normalización de tipos
    
    if 'journal' in df.columns:
        df['journal'] = df['journal'].apply(clean_journal_name)
    if 'publisher' in df.columns:
        df['publisher'] = df['publisher'].apply(clean_journal_name)
    
    # Limpieza de años
    if 'year' in df.columns:
        df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')[0]
        valid_years = df['year'].notna()
        print(f"- Publicaciones con año válido: {valid_years.sum()}/{len(df)}")
    
    # Paso 3: Generar estadísticas
    print("\nGenerando estadísticas...")
    stats = generate_statistics(df)
    
    # Paso 4: Mostrar resumen
    print("\nResumen de estadísticas:")
    print(f"- Total publicaciones: {len(df)}")
    print("- Distribución por tipo normalizado:")
    print(df['tipo_normalizado'].value_counts().to_string())
    
    # Paso 5: Exportar resultados
    output_stats_path = "C:/Users/dani4/OneDrive/Escritorio/Ws8/ProyectoFinalAlgoritmos/Data/DataEstadistica/Estadisticas.xlsx"
    folder_graficos = "C:/Users/dani4/OneDrive/Escritorio/Ws8/ProyectoFinalAlgoritmos/Data/DataEstadistica"
    # Generar y guardar gráficos como imágenes
    generate_and_save_charts(stats, folder_graficos)
    save_statistics(stats, output_stats_path)
    print(f"\nEstadísticas y graficos exportadas a: {output_stats_path}")

if __name__ == "__main__":
    main()
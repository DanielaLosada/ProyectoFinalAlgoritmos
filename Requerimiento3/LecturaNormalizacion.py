import re
import unicodedata
from collections import Counter
from tqdm import tqdm

# Funcion encargada de analizar archivos BibTeX grandes
# y extraer los campos relevantes, incluyendo el manejo de campos complejos
def parse_large_bib(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    entry_pattern = re.compile(
        r'@(?P<type>\w+)\s*{\s*(?P<id>[^,\s]+)\s*,\s*'
        r'(?P<fields>(?:[^@]*?)\s*(?=\s*@\w+\s*{|$))',
        re.DOTALL
    )
    
    field_pattern = re.compile(
        r'(?P<key>\w+)\s*=\s*'
        r'(?P<value>\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})+\}|"[^"]*"|[^,\n]+)\s*,?\s*',
        re.DOTALL
    )
    
    entries = []
    for match in tqdm(entry_pattern.finditer(content), desc="Analizando entradas"):
        entry = {
            'ENTRYTYPE': match.group('type').lower(),
            'ID': match.group('id').strip()
        }
        
        fields_content = match.group('fields')
        for field in field_pattern.finditer(fields_content):
            key = field.group('key').lower()
            value = field.group('value').strip()
            
            if value.startswith('{') and value.endswith('}'):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            
            value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('utf-8')
            entry[key] = value.strip()
        
        entries.append(entry)
    
    return entries

# Cargar el archivo BibTeX y extraer abstracts
def load_bibtex(file_path):
    entries = parse_large_bib(file_path)
    abstracts = []
    for entry in entries:
        # Buscar abstract en diferentes variaciones de campo
        abstract = entry.get('abstract') or entry.get('abstr') or entry.get('summary')
        if abstract:
            abstracts.append(abstract)
    
    print(f"\nSe encontraron {len(abstracts)} abstracts en el archivo.")
    print(f"Total de entradas en el archivo: {len(entries)}")
    
    # Depuración: mostrar algunos abstracts si no se encuentran
    if not abstracts and len(entries) > 0:
        print("\nDepuración: Mostrando las claves de la primera entrada:")
        print(list(entries[0].keys()))
    
    return abstracts

# Función para contar palabras clave en abstracts
# y devolver un diccionario con la frecuencia de cada palabra clave
def count_keywords(abstracts, keywords_dict):
    keyword_data = []  # Lista detallada con categoría
    keyword_counts = Counter()  # Dict para conteo rápido compatible con funciones existentes

    for abstract in abstracts:
        abstract_lower = abstract.lower()
        for category, terms in keywords_dict.items():
            for term, synonyms in terms.items():
                for synonym in synonyms:
                    if synonym.lower() in abstract_lower:
                        # Actualizar lista detallada
                        found = False
                        for entry in keyword_data:
                            if entry["Término"] == term and entry["Categoría"] == category:
                                entry["Frecuencia"] += 1
                                found = True
                                break
                        if not found:
                            keyword_data.append({
                                "Término": term,
                                "Categoría": category,
                                "Frecuencia": 1
                            })
                        # Actualizar contador rápido
                        keyword_counts[term] += 1
                        break  # Solo contar una vez por término principal
    return keyword_data, keyword_counts
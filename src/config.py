import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'abstracts_sessions_authors_topics.json')
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_dataframe.parquet')
PREPARED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'prepared_dataframe.parquet')
FINAL_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_dataframe.parquet')
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'abstract_embeddings.npy')
GRAPH_ORIGINAL_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'kg_original_data.gexf')
GRAPH_GENERATED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'kg_generated_data.gexf')
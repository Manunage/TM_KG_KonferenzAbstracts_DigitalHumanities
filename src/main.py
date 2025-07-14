import os
from keybert import KeyBERT
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import logging
import numpy as np

# --- Configuration ---
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.join(current_script_dir, '..')
project_root = os.path.abspath(project_root)
CLEANED_DATA_PATH = os.path.join(project_root, 'data', 'processed', 'cleaned_dataframe.parquet')
EMBEDDINGS_PATH = os.path.join(project_root, 'data', 'processed', 'abstract_embeddings.npy')
PREPARED_DATA_PATH = os.path.join(project_root, 'data', 'processed', 'prepared_dataframe.parquet')
FINAL_DATA_PATH = os.path.join(project_root, 'data', 'processed', 'final_dataframe.parquet')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ---

def extract_keywords_by_language(row):
    text = row['combined_text']
    language = row['language']
    stop_words_lang = 'english'
    if language == 'German':
        stop_words_lang = 'german'
    return [keyword[0] for keyword in kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words=stop_words_lang,
        top_n=5
    )]


def create_embeddings():
    logger.info("Generating abstract embeddings...")
    abstract_embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)
    logger.info("Embeddings generated.")
    return abstract_embeddings


def create_clustering(number_of_topics, embeddings):
    logger.info(f"Attempting to find {number_of_topics} session topics.")
    logger.info("Starting clustering...")
    kmeans_model = KMeans(n_clusters=number_of_topics, random_state=42, n_init=10)
    result = kmeans_model.fit_predict(embeddings)
    logger.info("Abstracts assigned to clusters.")
    return result


def extract_keyword_column():
    logger.info("Extracting keywords for each abstract...")
    result = df.apply(extract_keywords_by_language, axis=1)
    logger.info("Individual abstract keywords extracted.")
    return result


def create_session_topics():
    logger.info("Defining session topics from clusters...")
    session_topics = {}
    for i in range(num_topics):
        logger.info("At topic number " + str(i))
        cluster_abstracts = df[df['cluster_label'] == i]['combined_text'].tolist()
        # cluster_abstracts_flattened = [word for sublist in cluster_abstracts for word in sublist]
        if cluster_abstracts:
            combined_text = " ".join(cluster_abstracts)
            # Extract keywords from the combined text to represent the topic
            topic_keywords = kw_model.extract_keywords(combined_text, keyphrase_ngram_range=(1, 3),
                                                       stop_words=['english', 'german'], top_n=10)
            session_topics[i] = ", ".join([kw[0] for kw in topic_keywords])
        else:
            session_topics[i] = "Undefined Topic"  # Should not happen with enough data
    logger.info("Session topics defined.")
    return session_topics


if __name__ == "__main__":
    if os.path.exists(PREPARED_DATA_PATH):
        df = pd.read_parquet(PREPARED_DATA_PATH)
    else:
        df = pd.read_parquet(CLEANED_DATA_PATH)

    df.info()
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    kw_model = KeyBERT(model=model)
    num_topics = df['session_id'].nunique()

    if os.path.exists(EMBEDDINGS_PATH):
        abstract_embeddings = np.load(EMBEDDINGS_PATH)
    else:
        abstract_embeddings = create_embeddings()
        np.save(EMBEDDINGS_PATH, abstract_embeddings)

    if 'cluster_label' not in df.columns:
        df['cluster_label'] = create_clustering(num_topics, abstract_embeddings)

    if 'keywords' not in df.columns:
        df['keywords'] = extract_keyword_column()


    created_session_topics = create_session_topics()

    for cluster_id, topic_name in created_session_topics.items():
        print(f"Cluster {cluster_id}: {topic_name}")

    topic_suggestions = {
        cluster_id: [keyword.strip() for keyword in keywords_string.split(',')]
        for cluster_id, keywords_string in created_session_topics.items()
    }

    df['session_topic_suggestions'] = df['cluster_label'].map(topic_suggestions)

    if not os.path.exists(FINAL_DATA_PATH):
        os.makedirs(os.path.dirname(FINAL_DATA_PATH), exist_ok=True)
        df.to_parquet(FINAL_DATA_PATH, index=False)
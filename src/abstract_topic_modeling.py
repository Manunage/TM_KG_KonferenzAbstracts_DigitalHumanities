import os
from keybert import KeyBERT
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN
from sentence_transformers import SentenceTransformer
import logging
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ---

def create_embeddings():
    logger.info("Generating abstract embeddings...")
    embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)
    logger.info("Embeddings generated.")
    return embeddings

'''

# EXPERIMENTAL, UNRESOLVED. DOES NOT PRODUCE GOOD CLUSTERS.

def create_clustering_HDBSCAN(embeddings):
    logger.info("Starting clustering using HDBSCAN")
    hdbscan_model = HDBSCAN(min_cluster_size=2)
    cluster_labels = hdbscan_model.fit_predict(embeddings)

    # Remove noise, as all points should be assigned to a topic
    noise_indices = np.where(cluster_labels == -1)[0]
    if len(noise_indices) > 0:
        logger.info(f"Reassigning {len(noise_indices)} noise points to closest clusters...")
        most_probable_clusters = np.argmax(hdbscan_model.probabilities_[noise_indices], axis=1)
        cluster_labels[noise_indices] = most_probable_clusters
        logger.info("Noise points reassigned.")
    else:
        logger.info("No noise points found, all points already assigned to clusters.")

    num_found_clusters = len(np.unique(cluster_labels))
    logger.info(f"HDBSCAN found {num_found_clusters} clusters (all points now assigned).")

    return cluster_labels

'''

def create_clustering_KMeans(number_of_topics, embeddings):
    logger.info("Starting clustering using KMeans")
    logger.info(f"Attempting to find {number_of_topics} topics.")
    kmeans_model = KMeans(n_clusters=number_of_topics, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(embeddings)
    logger.info("Abstracts assigned to clusters.")
    return cluster_labels


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
        top_n=3
    )]


def extract_keyword_column():
    logger.info("Extracting keywords for each abstract...")
    result = df.apply(extract_keywords_by_language, axis=1)
    logger.info("Individual abstract keywords extracted.")
    return result


def create_topics(df):
    logger.info("Defining session topics from clusters...")
    number_of_topics = df['cluster_label'].nunique()
    created_topics = {}
    for i in range(number_of_topics):
        logger.info("At topic number " + str(i))

        # cluster_abstracts = df[df['cluster_label'] == i]['keywords'].tolist()
        # cluster_abstracts = [keyword for sublist in cluster_abstracts for keyword in sublist]

        cluster_abstracts = df[df['cluster_label'] == i]['combined_text'].tolist()

        if cluster_abstracts:
            combined_text = " ".join(cluster_abstracts)
            topic_keywords = kw_model.extract_keywords(combined_text, keyphrase_ngram_range=(1, 3),
                                                       stop_words=['english', 'german'], top_n=5)
            created_topics[i] = ", ".join([kw[0] for kw in topic_keywords])
        else:
            created_topics[i] = "Undefined Topic"
    logger.info("Session topics defined.")
    return created_topics


def abstract_topic_modeling_pipeline(force_override=False):
    global df, model, kw_model

    if force_override:
        logger.info("Forcing creation of final data file.")
        df = pd.read_parquet(config.CLEANED_DATA_PATH)
    else:
        if os.path.exists(config.FINAL_DATA_PATH):
            df = pd.read_parquet(config.FINAL_DATA_PATH)
            logger.info("Final data file already exists.")
        elif os.path.exists(config.PREPARED_DATA_PATH):
            df = pd.read_parquet(config.PREPARED_DATA_PATH)
        else:
            df = pd.read_parquet(config.CLEANED_DATA_PATH)



    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    kw_model = KeyBERT(model=model)
    useHDBSCAN = False
    num_topics = 0
    if not useHDBSCAN:
        num_topics = df['session_id'].nunique()
    if os.path.exists(config.EMBEDDINGS_PATH):
        combined_text_embeddings = np.load(config.EMBEDDINGS_PATH)
    else:
        combined_text_embeddings = create_embeddings()
        np.save(config.EMBEDDINGS_PATH, combined_text_embeddings)
    if 'cluster_label' not in df.columns:
        if useHDBSCAN:
            # df['cluster_label'] = create_clustering_HDBSCAN(embeddings=abstract_embeddings) # DO NOT USE
            # num_topics = df['cluster_label'].nunique()
            pass
        else:
            df['cluster_label'] = create_clustering_KMeans(number_of_topics=num_topics,
                                                           embeddings=combined_text_embeddings)
    if 'keywords' not in df.columns:
        df['keywords'] = extract_keyword_column()
    if not os.path.exists(config.PREPARED_DATA_PATH):
        os.makedirs(os.path.dirname(config.PREPARED_DATA_PATH), exist_ok=True)
        df.to_parquet(config.PREPARED_DATA_PATH, index=False)
    if 'session_topic_suggestions' not in df.columns:
        created_topics = create_topics(df)
        for cluster_id, topic_name in created_topics.items():
            print(f"Cluster {cluster_id}: {topic_name}")
        session_topic_suggestions = {
            cluster_id: [keyword.strip() for keyword in keywords_string.split(',')]
            for cluster_id, keywords_string in created_topics.items()
        }
        df['session_topic_suggestions'] = df['cluster_label'].map(session_topic_suggestions)
    if not os.path.exists(config.FINAL_DATA_PATH) or force_override:
        os.makedirs(os.path.dirname(config.FINAL_DATA_PATH), exist_ok=True)
        df.to_parquet(config.FINAL_DATA_PATH, index=False)
        logger.info("Final data file created.")


if __name__ == "__main__":
    abstract_topic_modeling_pipeline()
import os
from keybert import KeyBERT
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import logging
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_embeddings(df_input, model_instance):
    embeddings = model_instance.encode(df_input['combined_text'].tolist(), show_progress_bar=True)
    return embeddings

def create_clustering_KMeans(df_input, number_of_topics, embeddings):
    logger.info("Starting clustering using KMeans")
    logger.info(f"Attempting to find {number_of_topics} sessions.")

    kmeans_model = KMeans(n_clusters=number_of_topics, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(embeddings)
    logger.info("Abstracts assigned to clusters.")
    return cluster_labels


def extract_keywords_by_language(row, kw_model_instance):
    text = row['combined_text']
    language = row['language']
    stop_words_lang = 'english'
    if language == 'German':
        stop_words_lang = 'german'
    return [keyword[0] for keyword in kw_model_instance.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words=stop_words_lang,
        top_n=3
    )]


def create_topics(df_input, kw_model_instance):
    logger.info("Defining session topics from clusters within subset...")
    number_of_topics = df_input['cluster_label'].nunique()
    created_topics = {}
    for i in range(number_of_topics):
        logger.info(f"Processing cluster number {i} within current subset.")

        cluster_abstracts = df_input[df_input['cluster_label'] == i]['combined_text'].tolist()

        combined_text = " ".join(cluster_abstracts)
        topic_keywords = kw_model_instance.extract_keywords(combined_text, keyphrase_ngram_range=(1, 3),
                                                   stop_words=['english', 'german'], top_n=5)
        created_topics[i] = ", ".join([kw[0] for kw in topic_keywords])
    logger.info("Session topics defined for subset.")
    return created_topics


def abstract_topic_modeling_pipeline(force_override=False):
    global df

    if force_override or not os.path.exists(config.FINAL_DATA_PATH):
        df = pd.read_parquet(config.CLEANED_DATA_PATH)
        logger.info("Loaded initial data from CLEANED_DATA_PATH for full processing.")
    elif os.path.exists(config.FINAL_DATA_PATH):
        df = pd.read_parquet(config.FINAL_DATA_PATH)
        logger.info("Loaded existing final data file. Set 'force_override=True' to reprocess.")
        return
    else:
        logger.error(f"Neither {config.FINAL_DATA_PATH} nor {config.CLEANED_DATA_PATH} exists. Cannot proceed.")
        return

    model_instance = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    kw_model_instance = KeyBERT(model=model_instance)

    if os.path.exists(config.EMBEDDINGS_PATH):
        combined_text_embeddings = np.load(config.EMBEDDINGS_PATH)
        logger.info("Loaded existing embeddings for entire dataset.")
    else:
        combined_text_embeddings = create_embeddings(df, model_instance)
        os.makedirs(os.path.dirname(config.EMBEDDINGS_PATH), exist_ok=True)
        np.save(config.EMBEDDINGS_PATH, combined_text_embeddings)
        logger.info("Generated and saved new embeddings for entire dataset.")

    if 'cluster_label' not in df.columns:
        df['cluster_label'] = -1
    if 'keywords' not in df.columns:
        df['keywords'] = [[] for _ in range(len(df))]
    if 'session_topic_suggestions' not in df.columns:
        df['session_topic_suggestions'] = [[] for _ in range(len(df))]

    unique_topic_ids = df['topic_id'].unique().tolist()
    logger.info(f"Identified unique topic_ids for partitioning: {unique_topic_ids}")

    # Initialize a counter for global cluster IDs
    current_global_cluster_offset = 0

    for topic_id in unique_topic_ids:
        logger.info(f"\n--- Processing subset for topic_id: {topic_id} ---")

        current_subset_indices = df[df['topic_id'] == topic_id].index
        current_subset_df = df.loc[current_subset_indices].copy()
        current_subset_embeddings = combined_text_embeddings[current_subset_indices]

        subset_num_sessions = current_subset_df['session_id'].nunique()

        # Generate local cluster labels (0-indexed within the subset)
        subset_cluster_labels = create_clustering_KMeans(
            df_input=current_subset_df,
            number_of_topics=subset_num_sessions,
            embeddings=current_subset_embeddings
        )

        # Apply the current global offset to the cluster labels before assigning to the main DataFrame
        df.loc[current_subset_indices, 'cluster_label'] = subset_cluster_labels + current_global_cluster_offset

        individual_keywords = current_subset_df.apply(lambda row: extract_keywords_by_language(row, kw_model_instance), axis=1)
        df.loc[current_subset_indices, 'keywords'] = individual_keywords

        # Pass the local cluster labels to create_topics as it expects 0-indexed clusters
        temp_df_for_topics = current_subset_df.copy()
        temp_df_for_topics['cluster_label'] = subset_cluster_labels

        subset_created_topics = create_topics(temp_df_for_topics, kw_model_instance)

        # Map the topic suggestions using the global, offset cluster IDs as keys
        session_topic_suggestions_map = {
            (cluster_id + current_global_cluster_offset): [keyword.strip() for keyword in keywords_string.split(',')]
            for cluster_id, keywords_string in subset_created_topics.items()
        }
        df.loc[current_subset_indices, 'session_topic_suggestions'] = \
            df.loc[current_subset_indices, 'cluster_label'].map(session_topic_suggestions_map)

        # Increment the global offset for the next topic_id
        current_global_cluster_offset += subset_num_sessions

        logger.info(f"--- Finished processing topic_id: {topic_id} ---")

    os.makedirs(os.path.dirname(config.PREPARED_DATA_PATH), exist_ok=True)
    df.to_parquet(config.PREPARED_DATA_PATH, index=False)
    logger.info(f"Intermediate prepared data saved to {config.PREPARED_DATA_PATH}.")

    os.makedirs(os.path.dirname(config.FINAL_DATA_PATH), exist_ok=True)
    df.to_parquet(config.FINAL_DATA_PATH, index=False)
    logger.info(f"Final data file created at {config.FINAL_DATA_PATH}.")


if __name__ == "__main__":
    abstract_topic_modeling_pipeline(force_override=False)

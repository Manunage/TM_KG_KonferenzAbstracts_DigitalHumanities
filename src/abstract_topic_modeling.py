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
    logger.info("Generating abstract embeddings...")
    embeddings = model_instance.encode(df_input['combined_text'].tolist(), show_progress_bar=True)
    logger.info("Embeddings generated.")
    return embeddings

def create_clustering_KMeans(df_input, number_of_topics, embeddings):
    """
    Performs KMeans clustering on the provided embeddings to group abstracts into topics.

    Args:
        df_input (pd.DataFrame): The DataFrame being processed (used for logging context).
        number_of_topics (int): The desired number of clusters (topics).
        embeddings (np.ndarray): The embeddings to cluster.

    Returns:
        np.ndarray: An array of cluster labels for each abstract.
    """
    logger.info("Starting clustering using KMeans")
    logger.info(f"Attempting to find {number_of_topics} topics.")

    # KMeans requires at least 2 clusters. Handle the case where only one topic/session is present.
    if number_of_topics < 2:
        logger.warning(f"KMeans requires n_clusters >= 2. Given {number_of_topics}. Assigning all to cluster 0.")
        return np.zeros(len(df_input), dtype=int) # Assign all documents in this single-session case to cluster 0

    kmeans_model = KMeans(n_clusters=number_of_topics, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(embeddings)
    logger.info("Abstracts assigned to clusters.")
    return cluster_labels


def extract_keywords_by_language(row, kw_model_instance):
    """
    Extracts keywords from a single row's 'combined_text' based on its 'language'.

    Args:
        row (pd.Series): A row from the DataFrame.
        kw_model_instance (KeyBERT): The KeyBERT model instance.

    Returns:
        list: A list of extracted keywords.
    """
    text = row['combined_text']
    language = row['language']
    stop_words_lang = 'english'
    if language == 'German':
        stop_words_lang = 'german'
    # Extract top 3 keywords as keyphrases (1 to 3 words long)
    return [keyword[0] for keyword in kw_model_instance.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words=stop_words_lang,
        top_n=3
    )]


def create_topics(df_input, kw_model_instance):
    """
    Defines session topics by aggregating keywords from abstracts within each cluster of a subset.

    Args:
        df_input (pd.DataFrame): A subset of the DataFrame, expected to have 'cluster_label' and 'combined_text'.
        kw_model_instance (KeyBERT): The KeyBERT model instance.

    Returns:
        dict: A dictionary where keys are cluster IDs and values are comma-separated topic keywords.
    """
    logger.info("Defining session topics from clusters within subset...")
    number_of_topics = df_input['cluster_label'].nunique()
    created_topics = {}
    for i in range(number_of_topics):
        logger.info(f"Processing cluster number {i} within current subset.")

        cluster_abstracts = df_input[df_input['cluster_label'] == i]['combined_text'].tolist()

        if cluster_abstracts:
            # Combine all text from abstracts within the current cluster to extract topic keywords
            combined_text = " ".join(cluster_abstracts)
            topic_keywords = kw_model_instance.extract_keywords(combined_text, keyphrase_ngram_range=(1, 3),
                                                       stop_words=['english', 'german'], top_n=5)
            created_topics[i] = ", ".join([kw[0] for kw in topic_keywords])
        else:
            created_topics[i] = "Undefined Topic"
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
        return # Exit if final data exists and not forcing override
    else:
        logger.error(f"Neither {config.FINAL_DATA_PATH} nor {config.CLEANED_DATA_PATH} exists. Cannot proceed.")
        return

    model_instance = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    kw_model_instance = KeyBERT(model=model_instance)

    # Generate embeddings for the entire dataset once to avoid redundant computations
    if os.path.exists(config.EMBEDDINGS_PATH):
        combined_text_embeddings = np.load(config.EMBEDDINGS_PATH)
        logger.info("Loaded existing embeddings for entire dataset.")
    else:
        # If embeddings don't exist, create them for the full DataFrame
        combined_text_embeddings = create_embeddings(df, model_instance)
        os.makedirs(os.path.dirname(config.EMBEDDINGS_PATH), exist_ok=True) # Ensure directory exists
        np.save(config.EMBEDDINGS_PATH, combined_text_embeddings)
        logger.info("Generated and saved new embeddings for entire dataset.")

    # Initialize new columns to avoid KeyError during assignment
    if 'cluster_label' not in df.columns:
        df['cluster_label'] = -1
    if 'keywords' not in df.columns:
        df['keywords'] = [[] for _ in range(len(df))]
    if 'session_topic_suggestions' not in df.columns:
        df['session_topic_suggestions'] = [[] for _ in range(len(df))]

    # Determine unique topic_ids. If 'topic_id' column doesn't exist, treat the entire dataset as one topic.
    unique_topic_ids = df['topic_id'].unique().tolist() if 'topic_id' in df.columns and not df['topic_id'].isnull().all() else ['__all_data__']
    logger.info(f"Identified unique topic_ids for partitioning: {unique_topic_ids}")

    # Process each topic_id subset
    for topic_id in unique_topic_ids:
        logger.info(f"\n--- Processing subset for topic_id: {topic_id} ---")

        if topic_id == '__all_data__':
            # If no 'topic_id' column, or it's all null, process the entire DataFrame as one subset
            current_subset_df = df
            current_subset_indices = df.index
            current_subset_embeddings = combined_text_embeddings # Use all embeddings
        else:
            # Filter the DataFrame and embeddings for the current topic_id
            current_subset_indices = df[df['topic_id'] == topic_id].index
            current_subset_df = df.loc[current_subset_indices].copy() # Use .copy() to avoid SettingWithCopyWarning
            current_subset_embeddings = combined_text_embeddings[current_subset_indices]

        # Determine the number of unique sessions within this specific topic_id subset
        subset_num_sessions = current_subset_df['session_id'].nunique()

        if subset_num_sessions == 0:
            logger.warning(f"No sessions found for topic_id: {topic_id}. Skipping clustering and topic creation for this subset.")
            continue

        # Special handling for subsets with only one session, as KMeans needs n_clusters >= 2
        if subset_num_sessions == 1:
            logger.warning(f"Only one session found for topic_id: {topic_id}. Assigning all abstracts to cluster 0 and generating single topic suggestion.")
            df.loc[current_subset_indices, 'cluster_label'] = 0

            # Extract individual keywords for abstracts within this single session subset
            individual_keywords = current_subset_df.apply(lambda row: extract_keywords_by_language(row, kw_model_instance), axis=1)
            df.loc[current_subset_indices, 'keywords'] = individual_keywords

            # Generate a single topic suggestion for this entire single-session subset
            single_session_combined_text = " ".join(current_subset_df['combined_text'].tolist())
            if single_session_combined_text:
                topic_keywords = kw_model_instance.extract_keywords(single_session_combined_text, keyphrase_ngram_range=(1, 3), stop_words=['english', 'german'], top_n=5)
                single_topic_suggestion = [kw[0] for kw in topic_keywords]
            else:
                single_topic_suggestion = ["Undefined Topic (No content)"]
            # Apply the same single topic suggestion to all rows in this subset
            df.loc[current_subset_indices, 'session_topic_suggestions'] = [single_topic_suggestion] * len(current_subset_indices)
            continue # Move to the next topic_id


        # Step 1: Clustering within the current topic_id subset
        subset_cluster_labels = create_clustering_KMeans(
            df_input=current_subset_df, # Pass the subset DataFrame
            number_of_topics=subset_num_sessions,
            embeddings=current_subset_embeddings
        )
        # Assign the generated cluster labels back to the corresponding rows in the main DataFrame
        df.loc[current_subset_indices, 'cluster_label'] = subset_cluster_labels

        # Step 2: Extract individual keywords for each abstract within the current topic_id subset
        individual_keywords = current_subset_df.apply(lambda row: extract_keywords_by_language(row, kw_model_instance), axis=1)
        df.loc[current_subset_indices, 'keywords'] = individual_keywords

        # Step 3: Create topic suggestions for the clusters within the current topic_id subset
        # Temporarily create a copy of the subset DataFrame and add the cluster labels.
        # This is needed because `create_topics` expects 'cluster_label' to be a column in its input DataFrame.
        temp_df_for_topics = current_subset_df.copy()
        temp_df_for_topics['cluster_label'] = subset_cluster_labels # Add computed labels to the temp DF

        subset_created_topics = create_topics(temp_df_for_topics, kw_model_instance)

        # Map the topic suggestions back to the main DataFrame based on the cluster_label
        session_topic_suggestions_map = {
            cluster_id: [keyword.strip() for keyword in keywords_string.split(',')]
            for cluster_id, keywords_string in subset_created_topics.items()
        }
        # Apply the mapping to the 'session_topic_suggestions' column for the current subset's rows
        df.loc[current_subset_indices, 'session_topic_suggestions'] = \
            df.loc[current_subset_indices, 'cluster_label'].map(session_topic_suggestions_map)

        logger.info(f"--- Finished processing topic_id: {topic_id} ---")

    # After processing all topic_id subsets, save the intermediate prepared data
    os.makedirs(os.path.dirname(config.PREPARED_DATA_PATH), exist_ok=True)
    df.to_parquet(config.PREPARED_DATA_PATH, index=False)
    logger.info(f"Intermediate prepared data saved to {config.PREPARED_DATA_PATH}.")

    # Finally, save the complete processed DataFrame
    os.makedirs(os.path.dirname(config.FINAL_DATA_PATH), exist_ok=True)
    df.to_parquet(config.FINAL_DATA_PATH, index=False)
    logger.info(f"Final data file created at {config.FINAL_DATA_PATH}.")


if __name__ == "__main__":
    abstract_topic_modeling_pipeline(force_override=True)
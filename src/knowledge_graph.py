import os
import platform
import subprocess

import networkx as nx
import numpy as np
import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import FOAF, RDF, RDFS, XSD

import config

DC = Namespace("http://purl.org/dc/elements/1.1/")
EX = Namespace("http://example.org/abstract_kg#")


def create_rdf_graph_from_original_data(df):
    g = Graph()

    g.bind("foaf", FOAF)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("dc", DC)
    g.bind("ex", EX)

    created_abstracts = set()
    created_authors = set()
    created_topics = set()
    created_sessions = set()

    for index, row in df.iterrows():

        # ABSTRACTS
        abstract_id = row['id']
        abstract_uri = EX[f"abstract_{abstract_id}"]
        if abstract_uri not in created_abstracts:
            g.add((abstract_uri, RDF.type, EX.Abstract))
            g.add((abstract_uri, DC.title, Literal(row['title'], lang=row['language'])))
            g.add((abstract_uri, DC.language, Literal(row['language'])))
            g.add((abstract_uri, EX.hasAbstractText, Literal(row['content_raw'], lang=row['language'])))

            abstract_keywords_val = row['abstract_keywords']
            keywords_to_add = []
            if pd.isna(abstract_keywords_val).all() if isinstance(abstract_keywords_val,
                                                                  (pd.Series, np.ndarray)) else pd.isna(
                abstract_keywords_val):
                pass  # No keywords to process if NaN or all elements are NaN
            elif isinstance(abstract_keywords_val, (list, pd.Series, np.ndarray)):
                for item in abstract_keywords_val:
                    if item is not None and str(
                            item).strip():  # Ensure individual keyword is not empty or just whitespace
                        keywords_to_add.append(str(item).strip())
            else:
                if abstract_keywords_val is not None and str(
                        abstract_keywords_val).strip():  # Ensure it's not just whitespace
                    keywords_to_add.append(str(abstract_keywords_val).strip())

            for keyword_text in keywords_to_add:
                g.add((abstract_uri, EX.hasAbstractKeyword, Literal(keyword_text)))
            created_abstracts.add(abstract_uri)

        # AUTHORS
        author_id = row['author_id']
        author_uri = EX[f"author_{author_id}"]
        if author_uri not in created_authors:
            g.add((author_uri, RDF.type, FOAF.Person))
            g.add((author_uri, FOAF.name, Literal(f"Author {author_id}")))
            g.add((author_uri, EX.academicDegree, Literal(row['academicdegree'])))
            g.add((author_uri, EX.affiliationOrganization, Literal(row['affiliationorganisation'])))
            g.add((author_uri, EX.affiliationCity, Literal(row['affiliationcity'])))
            g.add((author_uri, EX.affiliationCountry, Literal(row['affiliationcountry'])))
            created_authors.add(author_uri)

        # TOPICS
        topic_id = row['topic_id']
        topic_title = row['topic_title']
        topic_uri = EX[f"topic_{topic_id}"]
        if topic_uri not in created_topics:
            g.add((topic_uri, RDF.type, EX.Topic))
            g.add((topic_uri, DC.title, Literal(topic_title, lang=row['language'])))

            topic_keywords_val = row['topic_keywords']
            keywords_to_add = []
            if pd.isna(topic_keywords_val).all() if isinstance(topic_keywords_val,
                                                               (pd.Series, np.ndarray)) else pd.isna(
                topic_keywords_val):
                pass  # No keywords to process if NaN or all elements are NaN
            elif isinstance(topic_keywords_val, (list, pd.Series, np.ndarray)):
                for item in topic_keywords_val:
                    if item is not None and str(
                            item).strip():  # Ensure individual keyword is not empty or just whitespace
                        keywords_to_add.append(str(item).strip())
            else:
                if topic_keywords_val is not None and str(
                        topic_keywords_val).strip():  # Ensure it's not just whitespace
                    keywords_to_add.append(str(topic_keywords_val).strip())

            for keyword_text in keywords_to_add:
                g.add((topic_uri, EX.hasTopicKeyword, Literal(keyword_text)))
            created_topics.add(topic_uri)

        # SESSIONS
        session_id = row['session_id']
        session_title = row['session_title']
        session_uri = EX[f"session_{session_id}"]
        if session_uri not in created_sessions:
            g.add((session_uri, RDF.type, EX.Session))
            g.add((session_uri, DC.title, Literal(session_title, lang=row['language'])))
            created_sessions.add(session_uri)

        # LINKS
        g.add((abstract_uri, DC.creator, author_uri))
        g.add((abstract_uri, EX.dealsWithTopic, topic_uri))
        g.add((abstract_uri, EX.partOfSession, session_uri))

    return g


def create_rdf_graph_from_generated_data(df):
    g = Graph()

    g.bind("foaf", FOAF)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("dc", DC)
    g.bind("ex", EX)

    created_abstracts = set()
    created_authors = set()
    created_topics = set()
    created_sessions = set()

    for index, row in df.iterrows():

        # ABSTRACTS
        abstract_id = row['id']
        abstract_uri = EX[f"abstract_{abstract_id}"]
        if abstract_uri not in created_abstracts:
            g.add((abstract_uri, RDF.type, EX.Abstract))
            g.add((abstract_uri, DC.title, Literal(row['title'], lang=row['language'])))
            g.add((abstract_uri, DC.language, Literal(row['language'])))
            g.add((abstract_uri, EX.hasAbstractText, Literal(row['content_raw'], lang=row['language'])))

            abstract_keywords_val = row['abstract_keywords']
            keywords_to_add = []
            if pd.isna(abstract_keywords_val).all() if isinstance(abstract_keywords_val,
                                                                  (pd.Series, np.ndarray)) else pd.isna(
                abstract_keywords_val):
                pass  # No keywords to process if NaN or all elements are NaN
            elif isinstance(abstract_keywords_val, (list, pd.Series, np.ndarray)):
                for item in abstract_keywords_val:
                    if item is not None and str(
                            item).strip():  # Ensure individual keyword is not empty or just whitespace
                        keywords_to_add.append(str(item).strip())
            else:
                if abstract_keywords_val is not None and str(
                        abstract_keywords_val).strip():  # Ensure it's not just whitespace
                    keywords_to_add.append(str(abstract_keywords_val).strip())

            for keyword_text in keywords_to_add:
                g.add((abstract_uri, EX.hasAbstractKeyword, Literal(keyword_text)))
            created_abstracts.add(abstract_uri)

        # AUTHORS
        author_id = row['author_id']
        author_uri = EX[f"author_{author_id}"]
        if author_uri not in created_authors:
            g.add((author_uri, RDF.type, FOAF.Person))
            g.add((author_uri, FOAF.name, Literal(f"Author {author_id}")))
            g.add((author_uri, EX.academicDegree, Literal(row['academicdegree'])))
            g.add((author_uri, EX.affiliationOrganization, Literal(row['affiliationorganisation'])))
            g.add((author_uri, EX.affiliationCity, Literal(row['affiliationcity'])))
            g.add((author_uri, EX.affiliationCountry, Literal(row['affiliationcountry'])))
            created_authors.add(author_uri)

        # TOPICS
        topic_id = row['topic_id']
        topic_title = row['topic_title']
        topic_uri = EX[f"topic_{topic_id}"]
        if topic_uri not in created_topics:
            g.add((topic_uri, RDF.type, EX.Topic))
            g.add((topic_uri, DC.title, Literal(topic_title, lang=row['language'])))

            topic_keywords_val = row['topic_keywords']
            if pd.isna(topic_keywords_val).all() if isinstance(topic_keywords_val,
                                                               (pd.Series, np.ndarray)) else pd.isna(
                topic_keywords_val):
                pass  # No keywords to process if NaN or all elements are NaN
            elif isinstance(topic_keywords_val, (list, pd.Series, np.ndarray)):
                for item in topic_keywords_val:
                    if item is not None and str(
                            item).strip():  # Ensure individual keyword is not empty or just whitespace
                        keywords_to_add.append(str(item).strip())
            else:
                if topic_keywords_val is not None and str(
                        topic_keywords_val).strip():  # Ensure it's not just whitespace
                    keywords_to_add.append(str(topic_keywords_val).strip())

            for keyword_text in keywords_to_add:
                g.add((topic_uri, EX.hasTopicKeyword, Literal(keyword_text)))
            created_topics.add(topic_uri)

        # SESSIONS
        session_id = row['cluster_label']
        cluster_keywords_val = row['cluster_keywords']
        if pd.isna(cluster_keywords_val).all() if isinstance(cluster_keywords_val,
                                                             (pd.Series, np.ndarray)) else pd.isna(
            cluster_keywords_val):
            session_title = ""  # Default to empty string if NaN or all elements are NaN
        elif isinstance(cluster_keywords_val, (list, pd.Series, np.ndarray)):
            # Filter out None/empty strings and join
            valid_keywords = [str(k).strip() for k in cluster_keywords_val if k is not None and str(k).strip()]
            session_title = ", ".join(valid_keywords)
        else:
            session_title = str(cluster_keywords_val).strip()  # Ensure it's a string and strip whitespace

        session_uri = EX[f"session_{session_id}"]
        if session_uri not in created_sessions:
            g.add((session_uri, RDF.type, EX.Session))
            g.add((session_uri, DC.title, Literal(session_title, lang=row['language'])))
            created_sessions.add(session_uri)

        # LINKS
        g.add((abstract_uri, DC.creator, author_uri))
        g.add((abstract_uri, EX.dealsWithTopic, topic_uri))
        g.add((abstract_uri, EX.partOfSession, session_uri))

    return g


def create_nx_graph_from_rdf_graph(g):
    nx_graph = nx.DiGraph()

    # Helper function to get a clean label from a URI
    def get_label_from_uri(uri_str):
        if '#' in uri_str:
            return uri_str.rsplit('#', 1)[-1]
        return uri_str.rsplit('/', 1)[-1]

    class_uris_to_exclude = {str(EX.Abstract),
                             str(EX.Topic),
                             str(EX.Session),
                             str(FOAF.Person)
                             }

    # First pass: Add all URIs and BNodes as NetworkX nodes with their attributes
    for node_uri in g.all_nodes():
        node_str = str(node_uri)
        if node_str in class_uris_to_exclude:
            continue

        if isinstance(node_uri, URIRef) or isinstance(node_uri, BNode):
            if node_str not in nx_graph:
                nx_graph.add_node(node_str)

                # Set label: prefer rdfs:label, then split URI
                label_val = g.value(node_uri, RDFS.label)
                if label_val:
                    nx_graph.nodes[node_str]['label'] = str(label_val)
                else:
                    nx_graph.nodes[node_str]['label'] = get_label_from_uri(node_str)

                # Set RDF type as a node attribute (useful for coloring/filtering in Gephi)
                rdf_type_val = g.value(node_uri, RDF.type)
                if rdf_type_val:
                    nx_graph.nodes[node_str]['rdf_type'] = get_label_from_uri(str(rdf_type_val))

                # Collect all literal properties for the node
                literal_props = {}
                for prop, val in g.predicate_objects(node_uri):
                    if isinstance(val, Literal):
                        prop_name = get_label_from_uri(str(prop))
                        # Avoid overwriting 'label' or 'rdf_type' if they were set above
                        if prop_name not in ['label', 'rdf_type']:  # 'rdf_type' is the key used above for RDF.type
                            if prop_name in literal_props:
                                literal_props[prop_name].append(str(val))
                            else:
                                literal_props[prop_name] = [str(val)]

                # Add collected literals to the NetworkX node, joining lists if multiple values exist
                for prop_name, values in literal_props.items():
                    if len(values) > 1:
                        nx_graph.nodes[node_str][prop_name] = ", ".join(values)
                    else:
                        nx_graph.nodes[node_str][prop_name] = values[0]

    # Add edges
    for s, p, o in g:
        s_str = str(s)
        p_str = str(p)
        o_str = str(o)

        if (isinstance(s, URIRef) or isinstance(s, BNode)) and \
                (isinstance(o, URIRef) or isinstance(o, BNode)):
            edge_label = get_label_from_uri(p_str)
            nx_graph.add_edge(s_str, o_str, relation=edge_label)

    nodes_to_remove = [node_id for node_id in nx_graph.nodes() if node_id in class_uris_to_exclude]
    nx_graph.remove_nodes_from(nodes_to_remove)

    return nx_graph


def open_graph(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path!r}")

    system = platform.system()
    if system == 'Windows':
        os.startfile(path)
    elif system == 'Darwin':
        subprocess.run(['open', path], check=True)
    else:
        subprocess.run(['xdg-open', path], check=True)


def knowledge_graph_pipeline():
    global df
    df = pd.read_parquet(config.FINAL_DATA_PATH)

    graph_original_data = create_rdf_graph_from_original_data(df)
    nx_graph_original_data = create_nx_graph_from_rdf_graph(graph_original_data)
    nx.write_gexf(nx_graph_original_data, config.GRAPH_ORIGINAL_DATA_PATH)

    graph_generated_data = create_rdf_graph_from_generated_data(df)
    nx_graph_generated_data = create_nx_graph_from_rdf_graph(graph_generated_data)
    nx.write_gexf(nx_graph_generated_data, config.GRAPH_GENERATED_DATA_PATH)


if __name__ == "__main__":
    knowledge_graph_pipeline()

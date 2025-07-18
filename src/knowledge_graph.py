from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import FOAF, RDF, RDFS, XSD
import networkx as nx
import pandas as pd

import config

def createRdfGraphFromDataFrame(df):
    g = Graph()

    g.bind("foaf", FOAF)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    DC = Namespace("http://purl.org/dc/elements/1.1/")
    g.bind("dc", DC)
    EX = Namespace("http://example.org/abstract_kg#")
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
            if isinstance(row['keywords'], list):
                for keyword_text in row['keywords']:
                    g.add((abstract_uri, EX.hasKeyword, Literal(keyword_text)))
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


def createNxGraphFromRdfGraph(g):
    nx_graph = nx.DiGraph()

    # Helper function to get a clean label from a URI
    def get_label_from_uri(uri_str):
        if '#' in uri_str:
            return uri_str.rsplit('#', 1)[-1]
        return uri_str.rsplit('/', 1)[-1]

    # First pass: Add all URIs and BNodes as NetworkX nodes with their attributes
    for node_uri in g.all_nodes():
        node_str = str(node_uri)
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

                # Add other literal properties as node attributes
                for prop, val in g.predicate_objects(node_uri):
                    if isinstance(val, Literal):
                        prop_name = get_label_from_uri(str(prop))
                        # Avoid overwriting 'label' or 'rdf_type' if they were set above
                        if prop_name not in ['label', 'type']: # 'type' is for RDF.type
                            nx_graph.nodes[node_str][prop_name] = str(val)

    # Add edges
    for s, p, o in g:
        s_str = str(s)
        p_str = str(p)
        o_str = str(o)

        if (isinstance(s, URIRef) or isinstance(s, BNode)) and \
           (isinstance(o, URIRef) or isinstance(o, BNode)):
            edge_label = get_label_from_uri(p_str)
            nx_graph.add_edge(s_str, o_str, relation=edge_label)

    return nx_graph

if __name__ == "__main__":

    df = pd.read_parquet(config.FINAL_DATA_PATH)

    g = createRdfGraphFromDataFrame(df)

    nx_graph = createNxGraphFromRdfGraph(g)

    nx.write_gexf(nx_graph, config.GRAPH_PATH)

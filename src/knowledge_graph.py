from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import FOAF, RDF, RDFS, XSD
import networkx as nx
import pandas as pd

import config
import os


g = Graph()

g.bind("foaf", FOAF)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)
g.bind("xsd", XSD)
DC = Namespace("http://purl.org/dc/elements/1.1/")
g.bind("dc", DC)
EX = Namespace("http://example.org/abstract_kg#")
g.bind("ex", EX)

if __name__ == "__main__":

    df = pd.read_parquet(config.FINAL_DATA_PATH)

    created_abstracts = set()
    created_authors = set()
    created_topics = set()
    created_sessions = set()
    created_session_topic_links = set()

    for index, row in df.iterrows():
        # --- Abstract Entity ---
        abstract_id = row['id']
        abstract_uri = EX[f"abstract_{abstract_id}"]

        if abstract_uri not in created_abstracts:
            # Only add abstract's core properties once for each unique abstract_id
            g.add((abstract_uri, RDF.type, EX.ResearchAbstract))
            g.add((abstract_uri, DC.title, Literal(row['title'], lang=row['language'])))
            g.add((abstract_uri, DC.language, Literal(row['language'])))
            g.add((abstract_uri, EX.hasAbstractText, Literal(row['content_raw'], lang=row['language'])))

            # Add keywords (assuming keywords are static per abstract ID)
            if isinstance(row['keywords'], list):
                for keyword_text in row['keywords']:
                    g.add((abstract_uri, EX.hasKeyword, Literal(keyword_text)))
            created_abstracts.add(abstract_uri)

        # --- Author Entity and Relationship ---
        author_id = row['author_id']
        author_uri = EX[f"author_{author_id}"]

        if author_uri not in created_authors:
            # Only add author details once for each unique author_id
            g.add((author_uri, RDF.type, FOAF.Person))
            # Assuming we can derive a name from author_id or another column if available
            # For this mock, we'll just use a generic name based on ID
            g.add((author_uri, FOAF.name, Literal(f"Author {author_id}"))) # You might have a separate author name column
            g.add((author_uri, EX.academicDegree, Literal(row['academicdegree'])))
            g.add((author_uri, EX.affiliationOrganization, Literal(row['affiliationorganisation'])))
            g.add((author_uri, EX.affiliationCity, Literal(row['affiliationcity'])))
            g.add((author_uri, EX.affiliationCountry, Literal(row['affiliationcountry'])))
            created_authors.add(author_uri)

        # Link abstract to author
        g.add((abstract_uri, DC.creator, author_uri))

        # --- Topic Entity and Relationship ---
        topic_id = row['topic_id']
        topic_title = row['topic_title']
        topic_uri = EX[f"topic_{topic_id}"]

        if topic_uri not in created_topics:
            g.add((topic_uri, RDF.type, EX.Topic))
            g.add((topic_uri, DC.title, Literal(topic_title, lang=row['language'])))
            created_topics.add(topic_uri)

        # Link abstract to topic
        g.add((abstract_uri, EX.dealsWithTopic, topic_uri))

        # --- Session Entity and Relationship ---
        session_id = row['session_id']
        session_title = row['session_title']
        session_uri = EX[f"session_{session_id}"]

        if session_uri not in created_sessions:
            g.add((session_uri, RDF.type, EX.Session))
            g.add((session_uri, DC.title, Literal(session_title, lang=row['language'])))
            created_sessions.add(session_uri)

        # Link abstract to session
        g.add((abstract_uri, EX.partOfSession, session_uri))

        # Link session to topic
        # A session can cover multiple topics, and a topic can be covered by multiple sessions.
        # We use a tuple (session_uri, topic_uri) to ensure we only add this link once.
        session_topic_link = (session_uri, topic_uri)
        if session_topic_link not in created_session_topic_links:
            g.add((session_uri, EX.coversTopic, topic_uri))
            created_session_topic_links.add(session_topic_link)

        # ------------------------------------------
        # Convert into format for gephi
        nx_graph = nx.DiGraph()
        nx_nodes_added = set()

    for s, p, o in g:
        # Convert URIs and Literals to strings for NetworkX node IDs
        # For better readability in Gephi, we'll try to use rdfs:label or a stripped URI part as the node 'label' attribute.
        s_str = str(s)
        o_str = str(o)
        p_str = str(p)
        # Add subject node
        if s_str not in nx_graph:
            nx_graph.add_node(s_str)
            nx_nodes_added.add(s_str)
            # Try to get a human-readable label for the node
            label_val = g.value(s, RDFS.label)
            if label_val:
                nx_graph.nodes[s_str]['label'] = str(label_val)
            else:
                # Fallback: use the last part of the URI as label
                    nx_graph.nodes[s_str]['label'] = s.split('#')[-1] if '#' in s_str else s.split('/')[-1]

            # Add RDF type as a node attribute (useful for coloring/filtering in Gephi)
            rdf_type_val = g.value(s, RDF.type)
            if rdf_type_val:
                nx_graph.nodes[s_str]['rdf_type'] = str(rdf_type_val).split('#')[-1] if '#' in str(
                    rdf_type_val) else str(rdf_type_val).split('/')[-1]

            # Add other common literal properties as node attributes
            # This part can be expanded based on which literal properties you want to see directly on nodes in Gephi
            if isinstance(s, URIRef):  # Only process URIRefs for properties
                for prop, val in g.predicate_objects(s):
                    if isinstance(val, Literal):
                        prop_name = str(prop).split('#')[-1] if '#' in str(prop) else str(prop).split('/')[-1]
                        # Avoid overwriting 'label' or 'rdf_type' if they were set above
                        if prop_name not in ['label', 'type']:  # 'type' is for RDF.type
                            nx_graph.nodes[s_str][prop_name] = str(val)

        # Add object node (if it's a URI or BNode)
        if isinstance(o, URIRef) or isinstance(o, BNode):
            if o_str not in nx_graph:
                nx_graph.add_node(o_str)
                nx_nodes_added.add(o_str)
                label_val = g.value(o, RDFS.label)
                if label_val:
                    nx_graph.nodes[o_str]['label'] = str(label_val)
                else:
                    nx_graph.nodes[o_str]['label'] = o.split('#')[-1] if '#' in o_str else o.split('/')[-1]

                rdf_type_val = g.value(o, RDF.type)
                if rdf_type_val:
                    nx_graph.nodes[o_str]['rdf_type'] = str(rdf_type_val).split('#')[-1] if '#' in str(
                        rdf_type_val) else str(rdf_type_val).split('/')[-1]

                if isinstance(o, URIRef):  # Only process URIRefs for properties
                    for prop, val in g.predicate_objects(o):
                        if isinstance(val, Literal):
                            prop_name = str(prop).split('#')[-1] if '#' in str(prop) else str(prop).split('/')[-1]
                            if prop_name not in ['label', 'type']:
                                nx_graph.nodes[o_str][prop_name] = str(val)
        elif isinstance(o, Literal):
            # If the object is a Literal, it typically becomes an attribute of the subject node
            # We've already handled this when adding subject node attributes above,
            # but if a literal is the object of a triple, it can't be a node itself in NetworkX for GEXF export
            # unless you create a special "literal node" which is usually not recommended for Gephi.
            pass  # Already handled or not applicable for direct edge creation

        # Add the edge (relationship)
        # The predicate (p) becomes the edge 'relation' attribute in NetworkX
        # We use the full predicate URI as the edge ID for uniqueness, but a stripped label for display
        edge_label = p.split('#')[-1] if '#' in p_str else p.split('/')[-1]
        nx_graph.add_edge(s_str, o_str, relation=edge_label)

    print(
        f"--- Converted to NetworkX Graph with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges ---")
    print("-" * 30)

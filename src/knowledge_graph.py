from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import FOAF, RDF, RDFS, XSD
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


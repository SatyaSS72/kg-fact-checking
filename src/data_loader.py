from rdflib import Graph, URIRef
from rdflib.namespace import RDF
from src.utils import get_logger

logger = get_logger("data_loader")


TRUTH_PROP = URIRef("http://swc2017.aksw.org/hasTruthValue")

def parse_statements(path):
    logger.info(f"        📥 Parsing reified statements from {path}")

    facts = []
    try:
        logger.info("        🧾 Loading RDF graph")
        g = Graph()
        g.parse(path, format="nt")

        logger.info("        🔍 Extracting RDF.Statement instances")
        for stmt in g.subjects(RDF.type, RDF.Statement):
            s = str(g.value(stmt, RDF.subject))
            p = str(g.value(stmt, RDF.predicate))
            o = str(g.value(stmt, RDF.object))
            t = g.value(stmt, TRUTH_PROP)
            facts.append((stmt, s, p, o, t))

    except Exception as e:
        logger.exception(f"        💥 Failed to parse statements from {path}")
        raise e

    logger.info(f"         ✔ Loaded {len(facts)} reified statements")
    return facts

def parse_triples(path):
    logger.info(f"        📥 Parsing triples from {path}")

    triples = []
    try:
        logger.info("        🧾 Loading RDF graph")
        g = Graph()
        g.parse(path, format="nt")

        logger.info("        🧹 Filtering URI-only triples")
        for s, p, o in g:
            if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef):
                triples.append((str(s), str(p), str(o)))

    except Exception as e:
        logger.exception(f"        💥 Failed to parse triples from {path}")
        raise e

    logger.info(f"        ✔ Loaded {len(triples)} triples")
    return triples

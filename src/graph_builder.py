import networkx as nx
from src.utils import get_logger

logger = get_logger("graph_builder")


def build_graph(triples):
    logger.info("   ▶ Initializing graph structure")

    try:
        G = nx.MultiDiGraph()
        logger.info("      🔗 Adding triples as graph edges")
        for s, p, o in triples:
            G.add_edge(s, o, predicate=p)
    except Exception as e:
        logger.exception("      💥 Graph construction failed")
        raise e

    logger.info(
        f"      🏗️ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )
    return G

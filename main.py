import logging
from src.utils import setup_logger, get_logger, silence_pykeens, filter_warnings

setup_logger(
    name="",               
    level=logging.INFO    
)
silence_pykeens()
filter_warnings()

logger = get_logger("fact-checker")

from src.utils import set_determinism, SEED
from src.data_loader import parse_statements, parse_triples
from src.graph_builder import build_graph
from src.embeddings import train_embedding
from src.features import extract_features, hard_negatives
from src.classifier import train_global_classifier
from src.models import train_models
from src.prediction import predict_fact

from pykeen.models import RotatE, ComplEx
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import XSD
from collections import defaultdict, Counter



def main():
    set_determinism()
    logger.info("🚀 Starting Knowledge Graph Fact Checking Pipeline")

    logger.info("📂 Loading datasets")
    logger.info("    ▶ Loading training dataset...")
    train_facts = parse_statements("data/KG-2022-train.nt")
    logger.info(f"      ✔ Training facts loaded: {len(train_facts)}")

    logger.info("    ▶ Loading testing dataset...")
    test_facts = parse_statements("data/KG-2022-test.nt")
    logger.info(f"      ✔ Test facts loaded: {len(test_facts)}")

    logger.info("    ▶ Loading reference KG...")
    ref_triples = parse_triples("data/reference-kg.nt")
    logger.info(f"      ✔ Reference KG triples loaded: {len(ref_triples)}")

    ref_kg_set = set(ref_triples)
    all_triples = [(s, p, o) for _, s, p, o, _ in train_facts + test_facts] + ref_triples

    logger.info("🕸️  Building knowledge graph")
    G = build_graph(all_triples)
    logger.info("     ✔ Graph construction complete")

    logger.info("🧠 Training knowledge graph embeddings (RotatE + ComplEx)")
    rot, tf = train_embedding(RotatE, all_triples, 200, 5, SEED)
    logger.info("     ✔ RotatE embedding trained")

    cx, _ = train_embedding(ComplEx, all_triples, 200, 5, SEED)
    logger.info("     ✔ ComplEx embedding trained")


    logger.info("🧩 Constructing predicate-specific training data")
    data = defaultdict(lambda: ([], []))

    for _, s, p, o, t in train_facts:
        if t is None:
            continue
        x = extract_features(G, s, p, o, rot, cx, tf, ref_kg_set)
        data[p][0].append(x)
        data[p][1].append(float(t))

        if float(t) == 1.0:
            for o_neg in hard_negatives(rot, tf, o):
                xn = extract_features(G, s, p, o_neg, rot, cx, tf, ref_kg_set)
                data[p][0].append(xn)
                data[p][1].append(0.0)


    logger.info("🌍 Training global classifier")
    g_scaler, g_classifier = train_global_classifier(data, SEED)
    logger.info("    ✔ Global classifier trained")

    logger.info("🛠️ Training predicate-specific models")
    models, scalers = train_models(data, SEED)
    logger.info(f"    ✔ Predicate models trained: {len(models)}")


    logger.info("🔮 Generating predictions for test facts")
    pred_freq = Counter(
        p for _, _, p, _, t in train_facts
        if t is not None and float(t) == 1.0
    )

    total_pos = sum(pred_freq.values())
    if total_pos == 0:
        total_pos = 1

    out = Graph()
    for stmt, s, p, o, _ in test_facts:
        f = extract_features(G, s, p, o, rot, cx, tf, ref_kg_set).reshape(1, -1)
        score = predict_fact(f, p, models, scalers, g_scaler, g_classifier, pred_freq, total_pos)
        out.add((URIRef(stmt),
                 URIRef("http://swc2017.aksw.org/hasTruthValue"),
                 Literal(float(score), datatype=XSD.double)))

    out.serialize("result.ttl", format="nt")
    logger.info("✅ Prediction complete — result.ttl generated")

if __name__ == "__main__":
    main()

import torch
import numpy as np
import networkx as nx
from src.utils import get_logger, cosine

logger = get_logger("features")


def relation_paths(G, s, p, o):
    same_pred = 0
    try:
        for path in nx.all_simple_edge_paths(G, s, o, cutoff=2):
            if all(G.edges[e]["predicate"] == p for e in path):
                same_pred += 1
    except nx.NetworkXNoPath:
        pass
    except Exception:
        logger.debug(f"        🧪 Path computation failed for ({s}, {p}, {o})")

    try:
        shortest = nx.shortest_path_length(G, s, o)
    except Exception:
        shortest = 10

    return same_pred, shortest

def model_score(model, tf, s, p, o):
    try:
        if s not in tf.entity_to_id or o not in tf.entity_to_id or p not in tf.relation_to_id:
            return 0.0

        ids = torch.tensor([[tf.entity_to_id[s],
                              tf.relation_to_id[p],
                              tf.entity_to_id[o]]])
        score = model.score_hrt(ids)
        return score.real.item() if torch.is_complex(score) else score.item()

    except Exception:
        logger.debug(f"        🧮 Embedding score failed for ({s}, {p}, {o})")
        return 0.0

def margin_score(model, tf, s, p, o):
    ent, rel = tf.entity_to_id, tf.relation_to_id
    if s not in ent or o not in ent or p not in rel:
        return 0.0

    try:
        E = model.entity_representations[0]().detach()
        R = model.relation_representations[0]().detach()

        if torch.is_complex(E): E = E.real
        if torch.is_complex(R): R = R.real

        return -torch.norm(E[ent[s]] + R[rel[p]] - E[ent[o]], p=1).item()

    except Exception:
        logger.debug(f"        🧮 Margin score failed for ({s}, {p}, {o})")
        return 0.0

def hard_negatives(model, tf, o, k=5):
    try:
        emb = model.entity_representations[0]().detach()
        if torch.is_complex(emb):
            emb = emb.real

        oid = tf.entity_to_id[o]
        sims = torch.cosine_similarity(emb[oid], emb)
        indices = torch.topk(sims, min(k + 1, sims.size(0))).indices[1:]

        id2ent = {v: k for k, v in tf.entity_to_id.items()}
        return [id2ent[i.item()] for i in indices]

    except Exception:
        logger.debug(f"    🎯 Hard negative sampling failed for entity {o}")
        return []

def extract_features(G, s, p, o, rot, cx, tf, ref_kg_set):
    try:
        out_deg = G.out_degree(s) if G.has_node(s) else 0
        in_deg = G.in_degree(o) if G.has_node(o) else 0

        rel_paths, shortest = relation_paths(G, s, p, o)

        rot_s = model_score(rot, tf, s, p, o)
        cx_s = model_score(cx, tf, s, p, o)

        rot_m = margin_score(rot, tf, s, p, o)
        cx_m = margin_score(cx, tf, s, p, o)

        ent = tf.entity_to_id
        if s in ent and o in ent:
            E = rot.entity_representations[0]().detach().real
            cos_so = cosine(E[ent[s]], E[ent[o]])
            l2_so = torch.norm(E[ent[s]] - E[ent[o]]).item()
        else:
            cos_so, l2_so = 0.0, 0.0

        in_ref = float((s, p, o) in ref_kg_set)
        rev_ref = float((o, p, s) in ref_kg_set)

        return np.array([
            out_deg, in_deg,
            rel_paths, shortest,
            rot_s, cx_s,
            rot_m, cx_m,
            cos_so, l2_so,
            in_ref, rev_ref
        ], dtype=float)

    except Exception as e:
        logger.exception("    💥 Feature extraction failed")
        raise e

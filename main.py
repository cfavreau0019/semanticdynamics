from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from app.embeddings import EmbeddingModel, chord_distance, arc_distance, semantic_volume, pairwise_distances

app = FastAPI(title="LLM Geometry API")
model = EmbeddingModel()


class Texts(BaseModel):
    texts: List[str]
    normalize: Optional[bool] = True


class DistanceRequest(BaseModel):
    a: Optional[List[float]] = None
    b: Optional[List[float]] = None
    texts: Optional[List[str]] = None
    metric: Optional[str] = "chord"


class SemanticVolumeRequest(BaseModel):
    texts: Optional[List[str]] = None
    embeddings: Optional[List[List[float]]] = None


@app.post("/embeddings")
def embeddings(req: Texts):
    embs = model.embed(req.texts, normalize=req.normalize)
    return {"embeddings": embs.tolist()}


@app.post("/distance")
def distance(req: DistanceRequest):
    if req.a is None or req.b is None:
        if req.texts and len(req.texts) == 2:
            embs = model.embed(req.texts)
            a, b = embs[0], embs[1]
        else:
            return {"error": "Provide either `a` and `b` embeddings, or `texts` (2 items)."}
    else:
        a = np.array(req.a, dtype=float)
        b = np.array(req.b, dtype=float)

    if req.metric == "chord":
        d = chord_distance(a, b)
    elif req.metric == "arc":
        d = arc_distance(a, b)
    elif req.metric == "cosine":
        d = float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    else:
        d = float(np.linalg.norm(a - b))
    return {"distance": d, "metric": req.metric}


@app.post("/geom/semantic_volume")
def semantic_volume_endpoint(req: SemanticVolumeRequest):
    if req.embeddings is not None:
        embs = np.array(req.embeddings, dtype=float)
    elif req.texts is not None:
        embs = model.embed(req.texts)
    else:
        return {"error": "Provide `embeddings` or `texts`."}
    vol = float(semantic_volume(embs))
    return {"semantic_volume": vol}


@app.post("/geom/pairwise")
def pairwise(req: SemanticVolumeRequest, metric: Optional[str] = "chord"):
    if req.embeddings is not None:
        embs = np.array(req.embeddings, dtype=float)
    elif req.texts is not None:
        embs = model.embed(req.texts)
    else:
        return {"error": "Provide `embeddings` or `texts`."}
    mat = pairwise_distances(embs, metric=metric)
    return {"pairwise": mat.tolist(), "metric": metric}


@app.post("/geom/hallucination")
def hallucination(source: Texts, generated: Texts):
    s_emb = model.embed(source.texts)
    g_emb = model.embed(generated.texts)
    # compute min distances from each generated to sources
    scores = []
    for ge in g_emb:
        dists = [chord_distance(ge, se) for se in s_emb]
        scores.append({"min_chord": min(dists), "mean_chord": float(np.mean(dists))})
    return {"scores": scores}

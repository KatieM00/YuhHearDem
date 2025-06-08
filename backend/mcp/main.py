#!/usr/bin/env python3
"""
Parliamentary Graph Query MCP Server with SSE Transport
------------------------------------------------------

A FastMCP tool-server that exposes search / traversal / export utilities
for a MongoDB-backed RDF-style graph of Barbados-Parliament data.

Cloud Run compatible version with CORS support and health endpoints.
"""

import os
import sys
import re
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Set, Any, Optional

from bson import ObjectId

# --- third-party -------------------------------------------------------------
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    from dotenv import load_dotenv
    from rdflib import Graph, URIRef, Literal, BNode, Namespace
    from rdflib.namespace import RDF, RDFS, OWL, FOAF, XSD
    from fastmcp import FastMCP
except ImportError as e:                        # pragma: no cover
    print(f"Missing package: {e}")
    print("pip install fastmcp pymongo python-dotenv rdflib")
    sys.exit(1)

# optional sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False

# --------------------------------------------------------------------------- #
#                               CONFIG / LOGGING                              #
# --------------------------------------------------------------------------- #
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#                               CORE  CLASS                                   #
# --------------------------------------------------------------------------- #
class GraphQuerier:
    """Graph-query helper for the MongoDB parliament DB."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        database_name: str = "parliamentary_graph",
    ):
        connection_string = (
            connection_string or os.getenv("MONGODB_CONNECTION_STRING")
        )
        if not connection_string:
            raise ValueError(
                "Set MONGODB_CONNECTION_STRING env var or pass connection_string."
            )

        try:
            self.client = MongoClient(connection_string)
            self.client.admin.command("ping")
            logger.info("✅  Connected to MongoDB")
        except ConnectionFailure as e:
            raise ConnectionFailure(f"Cannot connect to MongoDB: {e}") from e

        self.db = self.client[database_name]
        self.nodes = self.db.nodes
        self.edges = self.db.edges
        self.statements = self.db.statements
        self.videos = self.db.videos

        # embeddings model
        self.embedding_model = None
        if VECTOR_SEARCH_AVAILABLE:
            try:
                logger.info("🔄  Loading embedding model …")
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("✅  Vector search enabled")
            except Exception as exc:            # pragma: no cover
                logger.warning(f"Vector search disabled: {exc}")

    # -------------------------- utility helpers -------------------------- #
    def clean_mongodb_data(self, doc: Any) -> Any:
        """Convert ObjectId → str & strip embeddings for JSON serialisation."""
        if isinstance(doc, list):
            return [self.clean_mongodb_data(x) for x in doc]
        if isinstance(doc, dict):
            out = {}
            for k, v in doc.items():
                if k == "_id":
                    out[k] = str(v) if isinstance(v, ObjectId) else v
                elif k == "embedding":
                    continue
                else:
                    out[k] = self.clean_mongodb_data(v)
            return out
        if isinstance(doc, ObjectId):
            return str(doc)
        return doc

    # ------------------------------------------------------------------ #
    #                    SEARCH  (vector / text / hybrid)                #
    # ------------------------------------------------------------------ #
    def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        if not self.embedding_model:
            return None
        try:
            return self.embedding_model.encode(query).tolist()
        except Exception as exc:               # pragma: no cover
            logger.warning(f"Embedding failed: {exc}")
            return None

    def vector_search_nodes(self, query: str, limit: int = 8) -> List[Dict]:
        if not self.embedding_model:
            return []

        vec = self.generate_query_embedding(query)
        if vec is None:
            return []

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": vec,
                    "numCandidates": limit * 2,
                    "limit": limit,
                }
            },
            {"$addFields": {"similarity_score": {"$meta": "vectorSearchScore"}}},
        ]

        try:
            return list(self.nodes.aggregate(pipeline))
        except Exception as exc:               # pragma: no cover
            logger.warning(f"Vector search failed: {exc}")
            return []

    def text_search_nodes(self, query: str, limit: int = 8) -> List[Dict]:
        results: List[Dict] = []
        try:
            results = (
                self.nodes.find(
                    {"$text": {"$search": query}},
                    {"score": {"$meta": "textScore"}},
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(limit)
            )
        except Exception as exc:               # pragma: no cover
            logger.warning(f"Text search failed (index): {exc}")

        regex = re.compile(re.escape(query), re.IGNORECASE)
        regex_hits = list(
            self.nodes.find(
                {
                    "$or": [
                        {"label": {"$regex": regex}},
                        {"local_name": {"$regex": regex}},
                        {"searchable_text": {"$regex": regex}},
                        {"properties.http://schema.org/name": {"$regex": regex}},
                        {
                            "properties.http://www.w3.org/2000/01/rdf-schema#label": {
                                "$regex": regex
                            }
                        },
                        {"properties.http://xmlns.com/foaf/0.1/name": {"$regex": regex}},
                    ]
                }
            ).limit(limit)
        )

        # merge deduplicating by uri
        out: Dict[str, Dict] = {}
        for doc in list(results) + regex_hits:
            uri = doc["uri"]
            out.setdefault(uri, doc)
        return list(out.values())[:limit]

    def hybrid_search_nodes(
        self, query: str, limit: int = 8, vector_weight: float = 0.7
    ) -> List[Dict]:
        vec = self.vector_search_nodes(query, limit)
        txt = self.text_search_nodes(query, limit)

        combined: Dict[str, Dict] = {}
        for d in vec:
            uri = d["uri"]
            vs = d.get("similarity_score", 0.0)
            d.update(
                vector_score=vs,
                text_score=0.0,
                hybrid_score=vs * vector_weight,
                search_types=["vector"],
            )
            combined[uri] = d

        for d in txt:
            uri = d["uri"]
            ts = d.get("score", 0.5)
            if uri in combined:
                base = combined[uri]
                base["text_score"] = ts
                base["hybrid_score"] = base["vector_score"] * vector_weight + ts * (
                    1 - vector_weight
                )
                base["search_types"].append(d.get("search_type", "text"))
            else:
                d.update(
                    vector_score=0.0,
                    text_score=ts,
                    hybrid_score=ts * (1 - vector_weight),
                    search_types=[d.get("search_type", "text")],
                )
                combined[uri] = d

        return sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)[
            :limit
        ]

    def search_nodes(self, query: str, mode: str = "hybrid", limit: int = 8) -> List[Dict]:
        if mode == "vector":
            res = self.vector_search_nodes(query, limit)
        elif mode == "text":
            res = self.text_search_nodes(query, limit)
        else:
            res = self.hybrid_search_nodes(query, limit)

        for r in res:
            r.pop("embedding", None)
        return res

    # ------------------------------------------------------------------ #
    #                         GRAPH  TRAVERSAL                           #
    # ------------------------------------------------------------------ #
    def get_connected_nodes(self, uris: Set[str], hops: int = 2) -> Set[str]:
        current, seen = set(uris), set(uris)
        for _ in range(max(0, hops)):
            if not current:
                break
            edges = self.edges.find(
                {
                    "$or": [
                        {"subject": {"$in": list(current)}},
                        {"object": {"$in": list(current)}},
                    ]
                }
            )
            nxt = {e["subject"] for e in edges}.union({e["object"] for e in edges})
            current = nxt - seen
            seen.update(nxt)
        return seen

    def get_subgraph(self, uris: Set[str]) -> Dict[str, Any]:
        return {
            "nodes": list(
                self.nodes.find({"uri": {"$in": list(uris)}}, {"embedding": 0})
            ),
            "edges": list(
                self.edges.find({"subject": {"$in": list(uris)}, "object": {"$in": list(uris)}})
            ),
            "statements": list(
                self.statements.find(
                    {"$or": [{"subject": {"$in": list(uris)}}, {"object": {"$in": list(uris)}}]}
                )
            ),
        }

    def query_graph(
        self,
        query: str,
        hops: int = 2,
        search_mode: str = "hybrid",
        limit: int = 8,
    ) -> Dict[str, Any]:
        seeds = self.search_nodes(query, search_mode, limit)
        if not seeds:
            return {"nodes": [], "edges": [], "statements": [], "search_results": []}

        uris = {n["uri"] for n in seeds}
        all_uris = self.get_connected_nodes(uris, hops)
        sg = self.get_subgraph(all_uris)

        sg["search_results"] = self.clean_mongodb_data(seeds)
        sg["nodes"] = self.clean_mongodb_data(sg["nodes"])
        sg["edges"] = self.clean_mongodb_data(sg["edges"])
        sg["statements"] = self.clean_mongodb_data(sg["statements"])
        return sg

    # ------------------------------------------------------------------ #
    #                       RDF /  SERIALISATION                         #
    # ------------------------------------------------------------------ #
    def safe_string_to_rdf_term(self, value: str):
        """Convert a raw string coming from Mongo into a safe rdflib term."""
        # – obvious literals –
        if " " in value or len(value) > 100:
            return Literal(value)

        bad = {
            "ownership",
            "sector",
            "work",
            "national",
            "barbadian",
            "gardens",
            "tourism",
            "botanical",
            "development",
            "investment",
            "contribution",
        }
        if any(b in value.lower() for b in bad):
            return Literal(value)

        # URIs
        if value.startswith(("http://", "https://")):
            try:
                return URIRef(value)
            except Exception:                 # pragma: no cover
                return Literal(value)

        # blank nodes
        if value.startswith("_:"):
            return BNode(value[2:])

        # prefixed names that we store verbatim (bbp:, sess:, schema:)
        if re.match(r"^(bbp|sess|schema):[^\s]+$", value):
            return URIRef(value)

        # angle-bracketed URI
        if value.startswith("<") and value.endswith(">"):
            inside = value[1:-1]
            if " " not in inside:
                try:
                    return URIRef(inside)
                except Exception:
                    pass
            return Literal(value)

        # numbers
        if re.fullmatch(r"\d+", value):
            return Literal(value, datatype=XSD.integer)
        if re.fullmatch(r"\d+\.\d+", value):
            return Literal(value, datatype=XSD.decimal)

        # dates / years
        if re.fullmatch(r"\d{4}", value):
            return Literal(value, datatype=XSD.gYear)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return Literal(value, datatype=XSD.date)

        # default
        return Literal(value)

    # -------------------- subgraph → rdflib Graph ---------------------- #
    def subgraph_to_rdf_graph(self, sg: Dict[str, Any]) -> Graph:
        g = Graph()

        BBP = Namespace("http://example.com/barbados-parliament-ontology#")
        SESS = Namespace("http://example.com/barbados-parliament-session/")
        SCHEMA = Namespace("http://schema.org/")
        ORG = Namespace("http://www.w3.org/ns/org#")
        PROV = Namespace("http://www.w3.org/ns/prov#")

        for prefix, ns in [
            ("bbp", BBP),
            ("sess", SESS),
            ("schema", SCHEMA),
            ("org", ORG),
            ("prov", PROV),
            ("foaf", FOAF),
            ("owl", OWL),
            ("rdf", RDF),
            ("rdfs", RDFS),
            ("xsd", XSD),
        ]:
            g.bind(prefix, ns)

        # nodes
        for node in sg["nodes"]:
            try:
                n_uri = URIRef(node["uri"])
                for t in node.get("type", []):
                    g.add((n_uri, RDF.type, URIRef(t)))
                if "label" in node:
                    g.add((n_uri, RDFS.label, Literal(str(node["label"]))))
                for p_uri, vals in node.get("properties", {}).items():
                    if p_uri == str(RDF.type):
                        continue
                    p = URIRef(p_uri)
                    for v in vals if isinstance(vals, list) else [vals]:
                        g.add((n_uri, p, self.safe_string_to_rdf_term(str(v))))
            except Exception as exc:            # pragma: no cover
                logger.warning(f"Skip node {node.get('uri')}: {exc}")

        # edges
        for edge in sg["edges"]:
            try:
                g.add(
                    (
                        URIRef(edge["subject"]),
                        URIRef(edge["predicate"]),
                        self.safe_string_to_rdf_term(str(edge["object"])),
                    )
                )
            except Exception as exc:            # pragma: no cover
                logger.warning(f"Skip edge: {exc}")

        # reified statements with optional provenance
        for st in sg["statements"]:
            try:
                st_node = (
                    BNode(st["statement_uri"][2:])
                    if st.get("statement_uri", "").startswith("_:")
                    else BNode(f"stmt_{st['statement_id'][:8]}")
                )
                g.add((st_node, RDF.type, RDF.Statement))
                g.add((st_node, RDF.subject, URIRef(st["subject"])))
                g.add((st_node, RDF.predicate, URIRef(st["predicate"])))
                g.add((st_node, RDF.object, self.safe_string_to_rdf_term(str(st["object"]))))

                if any(k in st for k in ("from_video", "start_offset", "end_offset")):
                    seg = BNode(f"segment_{st['statement_id'][:8]}")
                    g.add((st_node, PROV.wasDerivedFrom, seg))
                    g.add((seg, RDF.type, BBP.TranscriptSegment))
                    if st.get("from_video"):
                        g.add((seg, BBP.fromVideo, URIRef(st["from_video"])))
                    if st.get("start_offset") is not None:
                        g.add(
                            (
                                seg,
                                BBP.startTimeOffset,
                                Literal(st["start_offset"], datatype=XSD.decimal),
                            )
                        )
                    if st.get("end_offset") is not None:
                        g.add(
                            (
                                seg,
                                BBP.endTimeOffset,
                                Literal(st["end_offset"], datatype=XSD.decimal),
                            )
                        )
            except Exception as exc:            # pragma: no cover
                logger.warning(f"Skip statement: {exc}")
        return g

    # ---------------------------  Turtle  ------------------------------ #
    def subgraph_to_turtle(self, sg: Dict[str, Any]) -> str:
        rdf = self.subgraph_to_rdf_graph(sg)
        header = (
            f"# Generated {datetime.now(timezone.utc).isoformat()}Z\n"
            f"# Nodes: {len(sg['nodes'])}, Edges: {len(sg['edges'])}, "
            f"Statements: {len(sg['statements'])}\n"
            f"# Total triples: {len(rdf)}\n\n"
        )
        try:
            return header + rdf.serialize(format="turtle")
        except Exception as exc:                # pragma: no cover
            logger.error(f"Turtle serialisation failed: {exc}")
            return header + "# <serialization-error>\n"


# --------------------------------------------------------------------------- #
#                       MCP TOOL  DEFINITIONS                                 #
# --------------------------------------------------------------------------- #
mcp = FastMCP("Parliamentary Graph Query Server")
_querier: Optional[GraphQuerier] = None


def get_querier() -> GraphQuerier:
    global _querier
    if _querier is None:
        _querier = GraphQuerier()
    return _querier


###############################################################################
# FastMCP TOOLS – TURTLE-ONLY QUERYING
###############################################################################

@mcp.tool()
def search_graph_turtle(
    query: str,
    seed_nodes: int = 8,
    hops: int = 2,
    search_mode: str = "hybrid",
) -> str:
    """
    search_graph_turtle
    -------------------
    Return an RDF/Turtle sub-graph for a *textual* query.

    Parameters
    ----------
    query : str
        Free-text search string (e.g. "Barbados Water Authority").
    seed_nodes : int, optional
        Number of top-ranked nodes to seed the traversal with (1-50, default 8).
    hops : int, optional
        How many relationship hops to follow from each seed (0-5, default 2).
    search_mode : {"hybrid","vector","text"}, optional
        Which underlying search engine to use (default "hybrid").

    Returns
    -------
    str
        Pure Turtle document.  No JSON metadata, no statistics.
    """
    seed_nodes = max(1, min(seed_nodes, 50))
    hops = max(0, min(hops, 5))
    if search_mode not in {"hybrid", "vector", "text"}:
        search_mode = "hybrid"

    q = get_querier()
    sg = q.query_graph(query=query, hops=hops, search_mode=search_mode, limit=seed_nodes)

    if not sg["nodes"]:
        return f"# No results found for query: {query}\n"

    return q.subgraph_to_turtle(sg)


@mcp.tool()
def expand_entity_turtle(
    entity_uri: str,
    hops: int = 1,
) -> str:
    """
    expand_entity_turtle
    --------------------
    Given a *specific* entity URI, expand outward and export the neighbourhood
    as Turtle.

    Parameters
    ----------
    entity_uri : str
        A full URI that is already present in the graph
        (e.g. "http://example.com/barbados-parliament-ontology#Org_GovernmentOfBarbados").
    hops : int, optional
        How many relationship hops to traverse away from the entity (0-5, default 1).

    Returns
    -------
    str
        Turtle serialisation of the resulting sub-graph.
    """
    hops = max(0, min(hops, 5))
    q = get_querier()

    # Collect neighbourhood URIs and build sub-graph
    uris = q.get_connected_nodes({entity_uri}, hops)
    sg = q.get_subgraph(uris)

    if not sg["nodes"]:
        return f"# No data found for entity: <{entity_uri}>\n"

    return q.subgraph_to_turtle(sg)

@mcp.tool()
def get_graph_statistics() -> dict:
    q = get_querier()
    stats = {
        "nodes": q.nodes.count_documents({}),
        "edges": q.edges.count_documents({}),
        "statements": q.statements.count_documents({}),
        "videos": q.videos.count_documents({}),
        "nodes_with_embeddings": q.nodes.count_documents({"embedding": {"$exists": True}}),
    }
    prov = q.statements.count_documents(
        {"start_offset": {"$ne": None}, "end_offset": {"$ne": None}}
    )
    return {
        "database_statistics": stats,
        "capabilities": {
            "vector_search": VECTOR_SEARCH_AVAILABLE and q.embedding_model is not None,
            "hybrid_search": VECTOR_SEARCH_AVAILABLE and q.embedding_model is not None,
            "text_search": True,
            "provenance_available": prov > 0,
        },
    }


@mcp.tool()
def health_check() -> dict:
    try:
        get_querier().client.admin.command("ping")
        return {"status": "healthy", "time": datetime.now(timezone.utc).isoformat()}
    except Exception as exc:                   # pragma: no cover
        return {"status": "unhealthy", "error": str(exc)}


# --------------------------------------------------------------------------- #
#                              ENTRY  POINT                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Cloud Run sets PORT environment variable
    port = int(os.getenv("PORT", "8080"))
    host = "0.0.0.0"

    logger.info("🚀  Starting Parliamentary Graph Query MCP Server for Cloud Run")
    logger.info(f"📍  http://{host}:{port}")

    try:
        # Test DB connection early
        _ = get_querier()
        
        # Run with SSE transport for public access
        mcp.run(
            transport="sse", 
            host=host, 
            port=port
        )
    except Exception as exc:                   # pragma: no cover
        logger.error(f"❌  Failed to start server: {exc}", exc_info=True)
        sys.exit(1)
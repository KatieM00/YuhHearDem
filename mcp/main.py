#!/usr/bin/env python3
"""
Parliamentary Graph Query MCP Server - Enhanced with Clean Output
----------------------------------------------------------------

Enhanced version with cleaned up output:
- De-duplicated labels and names
- Removed ranking information for concise results
- Provenance returned in Turtle format with minimal fields
"""

import os
import sys
import re
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Set, Any, Optional
import time
import numpy as np
import time

from bson import ObjectId

# --- third-party -------------------------------------------------------------
try:
    from pymongo import MongoClient, ASCENDING
    from pymongo.errors import ConnectionFailure
    from dotenv import load_dotenv
    from rdflib import Graph, URIRef, Literal, BNode, Namespace
    from rdflib.namespace import RDF, RDFS, OWL, FOAF, XSD
    from fastmcp import FastMCP
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    from fastapi import Request
except ImportError as e:
    print(f"Missing package: {e}")
    print("pip install fastmcp pymongo python-dotenv rdflib")
    sys.exit(1)

# mandatory sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Vector search is mandatory. Please install:")
    print("pip install sentence-transformers")
    sys.exit(1)

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
#                      ENHANCED GRAPH QUERIER WITH CLEAN OUTPUT              #
# --------------------------------------------------------------------------- #
class EnhancedGraphQuerier:
    """Enhanced graph querier with clean, concise output."""

    def __init__(self):
        self.client = None
        self.db = None
        self.nodes = None
        self.edges = None
        self.statements = None
        self.embedding_model = None
        self._initialize_database()
        self._initialize_embeddings()

    def _initialize_database(self):
        """Initialize database connection with proper error handling."""
        connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("MONGODB_CONNECTION_STRING environment variable not set")
        
        try:
            self.client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000,
                maxPoolSize=3,
                minPoolSize=1,
                retryWrites=True,
                w='majority'
            )
            
            # Test connection
            self.client.admin.command("ping", maxTimeMS=3000)
            
            # Initialize database references
            self.db = self.client["parliamentary_graph"]
            self.nodes = self.db.nodes
            self.edges = self.db.edges
            self.statements = self.db.statements
            
            # Create PageRank index if it doesn't exist
            try:
                self.nodes.create_index([("pagerank_score", ASCENDING)])
                self.nodes.create_index([("pagerank_rank", ASCENDING)])
            except:
                pass  # Index already exists
            
            logger.info("✅  Connected to MongoDB")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            if self.client:
                self.client.close()
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")

    def _initialize_embeddings(self):
        """Initialize embedding model - mandatory for operation."""
        try:
            logger.info("🔄  Loading embedding model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅  Vector search enabled")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise RuntimeError(f"Vector search is mandatory but failed to initialize: {e}")

    def _deduplicate_labels(self, labels: List[str]) -> str:
        """De-duplicate and clean labels, returning the best one."""
        if not labels:
            return None
        
        # Convert to set to remove exact duplicates
        unique_labels = list(set(labels))
        
        # If only one unique label, return it
        if len(unique_labels) == 1:
            return unique_labels[0]
        
        # Sort by length and quality - prefer longer, more descriptive labels
        # but avoid very long ones that might be concatenated
        def label_quality(label):
            # Prefer labels that are not too short, not too long, and descriptive
            length = len(label)
            if length < 3:
                return 0  # Too short
            if length > 100:
                return 1  # Too long
            if '(' in label and ')' in label:
                return 3  # Has parenthetical info
            return 2  # Good length
        
        # Sort by quality then by length
        sorted_labels = sorted(unique_labels, key=lambda x: (label_quality(x), len(x)), reverse=True)
        return sorted_labels[0]

    def search_nodes(self, query: str, limit: int = 8) -> List[Dict]:
        """Standard vector-only search for backward compatibility."""
        try:
            logger.info(f"🔍 Vector search for: {query}")
            vector_results = self._vector_search_nodes(query, limit)
            
            # Clean and simplify results
            cleaned_results = []
            for result in vector_results:
                cleaned = self._clean_node_result(result)
                if cleaned:
                    cleaned_results.append(cleaned)
            
            logger.info(f"Vector search found {len(cleaned_results)} results")
            return cleaned_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _clean_node_result(self, node: Dict) -> Dict:
        """Clean and simplify a node result, removing duplicates and unnecessary fields."""
        if not node:
            return None
        
        # Start with essential fields
        cleaned = {
            "uri": node.get("uri"),
            "type": node.get("type", [])
        }
        
        # Handle labels - deduplicate name and label fields
        all_labels = []
        if "label" in node and node["label"]:
            if isinstance(node["label"], list):
                all_labels.extend(node["label"])
            else:
                all_labels.append(node["label"])
        
        if "name" in node and node["name"]:
            if isinstance(node["name"], list):
                all_labels.extend(node["name"])
            else:
                all_labels.append(node["name"])
        
        # Deduplicate and set the best label - ONLY as 'label', remove 'name'
        best_label = self._deduplicate_labels(all_labels)
        if best_label:
            cleaned["label"] = best_label
        # Explicitly do NOT include 'name' field to avoid duplicates
        
        # Include essential content fields but not ranking or video info
        if "searchable_text" in node:
            cleaned["searchable_text"] = node["searchable_text"]
        
        # Video information removed - only available in provenance
        
        # Clean up _id
        if "_id" in node:
            cleaned["_id"] = str(node["_id"])
        
        return cleaned

    def hybrid_search(self, query: str, limit: int = 8, pagerank_weight: float = 0.3, 
                     min_pagerank_score: float = 0.00001) -> List[Dict]:
        """Hybrid search with clean output."""
        try:
            logger.info(f"🎯 Hybrid search for: '{query}' (PageRank weight: {pagerank_weight:.1%})")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # MongoDB vector search pipeline with PageRank integration
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 5,  # Get more candidates
                        "limit": limit * 3
                    }
                },
                {
                    "$match": {
                        "pagerank_score": {"$gte": min_pagerank_score}
                    }
                },
                {
                    "$addFields": {
                        "similarity_score": {"$meta": "vectorSearchScore"},
                        # Normalize PageRank score (assuming max around 0.01 for typical graphs)
                        "normalized_pagerank": {
                            "$divide": [
                                {"$ifNull": ["$pagerank_score", 0.00001]},
                                0.01  # Approximate max PageRank score
                            ]
                        }
                    }
                },
                {
                    "$addFields": {
                        "hybrid_score": {
                            "$add": [
                                {"$multiply": [pagerank_weight, "$normalized_pagerank"]},
                                {"$multiply": [(1 - pagerank_weight), "$similarity_score"]}
                            ]
                        }
                    }
                },
                {
                    "$sort": {"hybrid_score": -1}
                },
                {
                    "$limit": limit
                },
                {
                    "$project": {
                        "uri": 1,
                        "label": 1,
                        "name": 1,
                        "type": 1,
                        "searchable_text": 1
                        # Removed video fields - only in provenance
                    }
                }
            ]
            
            results = list(self.nodes.aggregate(pipeline))
            
            # Clean and simplify results
            cleaned_results = []
            for result in results:
                cleaned = self._clean_node_result(result)
                if cleaned:
                    cleaned_results.append(cleaned)
            
            logger.info(f"🎯 Hybrid search found {len(cleaned_results)} results")
            return cleaned_results
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            # Fall back to regular vector search
            return self.search_nodes(query, limit)

    def authority_search(self, query: str, limit: int = 8, min_pagerank_rank: int = 1000) -> List[Dict]:
        """Search for authoritative nodes with clean output."""
        try:
            logger.info(f"👑 Authority search for: '{query}' (max rank: {min_pagerank_rank})")
            
            # First get semantically relevant nodes with good PageRank
            candidates = self.hybrid_search(
                query, 
                limit=limit * 2, 
                pagerank_weight=0.7,  # Heavily favor PageRank
                min_pagerank_score=0.00001
            )
            
            # For authority search, we need to check PageRank rank in database
            # since we removed it from hybrid_search output
            enhanced_candidates = []
            for node in candidates:
                # Get PageRank rank from database
                db_node = self.nodes.find_one({"uri": node["uri"]}, {"pagerank_rank": 1})
                if db_node and db_node.get("pagerank_rank", float('inf')) <= min_pagerank_rank:
                    enhanced_candidates.append((node, db_node.get("pagerank_rank", float('inf'))))
            
            # Sort by PageRank rank (lower is better) and return just the nodes
            enhanced_candidates.sort(key=lambda x: x[1])
            authority_results = [node for node, rank in enhanced_candidates[:limit]]
            
            logger.info(f"👑 Found {len(authority_results)} authoritative nodes")
            return authority_results
            
        except Exception as e:
            logger.error(f"❌ Authority search failed: {e}")
            return []

    def topic_specific_search(self, query: str, limit: int = 8, topic_expansion: int = 50) -> List[Dict]:
        """Find nodes important within the specific topic domain of the query."""
        try:
            logger.info(f"🎯 Topic-specific search for: '{query}'")
            
            # Find semantically relevant nodes to define the topic
            topic_nodes = self.hybrid_search(
                query, 
                limit=topic_expansion,
                pagerank_weight=0.1,  # Favor similarity for topic definition
                min_pagerank_score=0.00001
            )
            
            if not topic_nodes:
                logger.warning("No topic nodes found")
                return []
            
            # Extract URIs of topic-relevant nodes
            topic_uris = {node['uri'] for node in topic_nodes}
            
            # Calculate mini-PageRank within this topic subgraph
            topic_pagerank_scores = self._calculate_topic_pagerank(topic_uris)
            
            # Combine topic PageRank with original relevance and return top results
            enhanced_results = []
            for node in topic_nodes:
                uri = node['uri']
                # Calculate hybrid score but don't include it in output
                if uri in topic_pagerank_scores:
                    topic_hybrid_score = (
                        0.6 * topic_pagerank_scores[uri] + 
                        0.4 * 0.5  # Assume reasonable similarity since these came from search
                    )
                else:
                    topic_hybrid_score = 0.2  # Low score for nodes without topic PageRank
                
                enhanced_results.append((node, topic_hybrid_score))
            
            # Sort by topic-specific hybrid score and return just the nodes
            enhanced_results.sort(key=lambda x: x[1], reverse=True)
            final_results = [node for node, score in enhanced_results[:limit]]
            
            logger.info(f"🎯 Topic-specific search found {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Topic-specific search failed: {e}")
            return self.hybrid_search(query, limit)

    def _calculate_topic_pagerank(self, topic_uris: Set[str], damping: float = 0.85, 
                                 max_iterations: int = 50) -> Dict[str, float]:
        """Calculate PageRank within a topic-specific subgraph."""
        try:
            # Get edges between topic nodes
            edges_cursor = self.edges.find({
                "subject": {"$in": list(topic_uris)},
                "object": {"$in": list(topic_uris)},
                "object_type": "uri"
            }, {"subject": 1, "object": 1})
            
            # Build adjacency structure
            uri_to_index = {uri: i for i, uri in enumerate(topic_uris)}
            index_to_uri = {i: uri for uri, i in uri_to_index.items()}
            num_nodes = len(topic_uris)
            
            if num_nodes == 0:
                return {}
            
            # Build edge list
            edges_list = []
            for edge in edges_cursor:
                subject_uri = edge["subject"]
                object_uri = edge["object"]
                if subject_uri in uri_to_index and object_uri in uri_to_index:
                    edges_list.append((uri_to_index[subject_uri], uri_to_index[object_uri]))
            
            if not edges_list:
                # No connections, return uniform scores
                uniform_score = 1.0 / num_nodes
                return {uri: uniform_score for uri in topic_uris}
            
            # Calculate PageRank using simplified algorithm
            pagerank_scores = self._simple_pagerank(edges_list, num_nodes, damping, max_iterations)
            
            # Map back to URIs
            return {index_to_uri[i]: float(score) for i, score in enumerate(pagerank_scores)}
            
        except Exception as e:
            logger.error(f"❌ Topic PageRank calculation failed: {e}")
            return {}

    def _simple_pagerank(self, edges_list: List[tuple], num_nodes: int, 
                        damping: float = 0.85, max_iterations: int = 50) -> np.ndarray:
        """Simplified PageRank calculation for topic subgraphs."""
        if num_nodes == 0:
            return np.array([])
        
        # Build adjacency information
        from collections import defaultdict
        out_links = defaultdict(list)
        out_degree = defaultdict(int)
        
        for from_node, to_node in edges_list:
            out_links[from_node].append(to_node)
            out_degree[from_node] += 1
        
        # Initialize PageRank scores
        pagerank = np.ones(num_nodes) / num_nodes
        
        # Power iteration
        for _ in range(max_iterations):
            new_pagerank = np.full(num_nodes, (1 - damping) / num_nodes)
            
            for node in range(num_nodes):
                if out_degree[node] > 0:
                    contribution = damping * pagerank[node] / out_degree[node]
                    for target in out_links[node]:
                        new_pagerank[target] += contribution
                else:
                    # Dangling node: distribute to all nodes
                    contribution = damping * pagerank[node] / num_nodes
                    new_pagerank += contribution
            
            # Check convergence
            if np.sum(np.abs(new_pagerank - pagerank)) < 1e-6:
                break
            
            pagerank = new_pagerank
        
        return pagerank

    def _vector_search_nodes(self, query: str, limit: int = 8) -> List[Dict]:
        """Vector search using sentence transformers."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # MongoDB vector search pipeline - simplified for clean output
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 3,
                        "limit": limit,
                    }
                },
                {
                    "$project": {
                        "uri": 1,
                        "label": 1,
                        "name": 1,
                        "type": 1,
                        "searchable_text": 1
                        # Removed video fields - only in provenance
                    }
                }
            ]

            results = list(self.nodes.aggregate(pipeline))
            logger.info(f"Vector search pipeline returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise RuntimeError(f"Vector search is mandatory but failed: {e}")

    def get_connected_nodes(self, uris: Set[str], hops: int = 1) -> Set[str]:
        """Get nodes connected to the given URIs."""
        try:
            current, seen = set(uris), set(uris)
            for hop in range(max(0, hops)):
                if not current or len(seen) > 500:
                    break
                    
                edges = self.edges.find({
                    "$or": [
                        {"subject": {"$in": list(current)}},
                        {"object": {"$in": list(current)}},
                    ]
                })
                
                nxt = set()
                for edge in edges:
                    nxt.add(edge["subject"])
                    nxt.add(edge["object"])
                
                current = nxt - seen
                seen.update(nxt)
                    
            return seen
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return uris

    def get_subgraph(self, uris: Set[str]) -> Dict[str, Any]:
        """Get subgraph for the given URIs with clean, deduplicated nodes."""
        try:
            if len(uris) > 500:
                uris = set(list(uris)[:500])
            
            # Get raw nodes without embedding, ranking info, and video info (keep name for deduplication)
            raw_nodes = list(self.nodes.find(
                {"uri": {"$in": list(uris)}}, 
                {
                    "embedding": 0, 
                    "pagerank_score": 0, 
                    "pagerank_rank": 0,
                    "similarity_score": 0,
                    "hybrid_score": 0,
                    "source_video": 0,
                    "video_title": 0
                    # Keep name and label for deduplication in _clean_node_result
                }
            ))
            
            # Clean and deduplicate each node
            cleaned_nodes = []
            for node in raw_nodes:
                cleaned = self._clean_node_result(node)
                if cleaned:
                    cleaned_nodes.append(cleaned)
            
            edges = list(self.edges.find({
                "subject": {"$in": list(uris)}, 
                "object": {"$in": list(uris)},
                "predicate": {"$ne": "http://schema.org/name"}  # Filter out schema:name edges
            }))
            
            # Clean MongoDB objects in edges
            for edge in edges:
                if "_id" in edge:
                    edge["_id"] = str(edge["_id"])
            
            return {"nodes": cleaned_nodes, "edges": edges}
            
        except Exception as e:
            logger.error(f"Subgraph retrieval failed: {e}")
            return {"nodes": [], "edges": []}

    def to_turtle(self, subgraph: Dict[str, Any]) -> str:
        """Convert subgraph to clean Turtle format without ranking information."""
        try:
            g = Graph()
            
            # Add prefixes
            g.bind("bbp", "http://example.com/barbados-parliament-ontology#")
            g.bind("sess", "http://example.com/barbados-parliament-session/")
            g.bind("rdfs", RDFS)
            g.bind("rdf", RDF)
            # Don't bind schema namespace to avoid automatic schema:name inference
            
            # Add nodes - clean and deduplicated
            for node in subgraph["nodes"]:
                try:
                    uri = URIRef(node["uri"])
                    
                    # Debug: Check what fields are actually in the cleaned node
                    logger.debug(f"Node fields: {list(node.keys())} for {uri}")
                    if "name" in node:
                        logger.warning(f"Found 'name' field in cleaned node: {node['name']}")
                    
                    # Add single deduplicated label only (no schema:name duplication)
                    if "label" in node and node["label"]:
                        g.add((uri, RDFS.label, Literal(str(node["label"]))))
                    
                    # Add types
                    for t in node.get("type", []):
                        g.add((uri, RDF.type, URIRef(t)))
                    
                    # Explicitly do NOT add any name or schema:name properties
                    
                except Exception as e:
                    logger.warning(f"Skipping node: {e}")
            
            # Add edges
            for edge in subgraph["edges"]:
                try:
                    g.add((
                        URIRef(edge["subject"]),
                        URIRef(edge["predicate"]),
                        URIRef(edge["object"]) if edge["object"].startswith("http") else Literal(edge["object"])
                    ))
                except Exception as e:
                    logger.warning(f"Skipping edge: {e}")
            
            header = f"# Generated {datetime.now(timezone.utc).isoformat()}Z\n"
            header += f"# Nodes: {len(subgraph['nodes'])}, Edges: {len(subgraph['edges'])}\n"
            header += f"# Clean output without ranking or video information\n\n"
            
            return header + g.serialize(format="turtle")
            
        except Exception as e:
            logger.error(f"Turtle serialization failed: {e}")
            return f"# Error: {str(e)}\n"

    def provenance_to_turtle(self, node_uris: List[str], include_transcript: bool = True) -> str:
        """Get provenance information and return as clean Turtle format."""
        try:
            logger.info(f"📚 Getting provenance for {len(node_uris)} nodes as Turtle")
            
            g = Graph()
            
            # Add prefixes
            g.bind("bbp", "http://example.com/barbados-parliament-ontology#")
            g.bind("prov", "http://www.w3.org/ns/prov#")
            g.bind("schema", "http://schema.org/")
            g.bind("xsd", "http://www.w3.org/2001/XMLSchema#")
            g.bind("rdfs", RDFS)
            
            PROV = Namespace("http://www.w3.org/ns/prov#")
            SCHEMA = Namespace("http://schema.org/")
            
            for uri in node_uris[:10]:  # Limit to prevent explosion
                try:
                    node_uri = URIRef(uri)
                    
                    # Get related statements with minimal fields
                    projection = {
                        "subject": 1,
                        "predicate": 1, 
                        "object": 1,
                        "source_video": 1,
                        "video_title": 1,
                        "start_offset": 1,
                        "end_offset": 1
                    }
                    
                    if include_transcript:
                        projection["transcript_text"] = 1
                    
                    statements = list(self.statements.find({
                        "$or": [
                            {"subject": uri},
                            {"predicate": uri}, 
                            {"object": uri}
                        ]
                    }, projection))
                    
                    # Process statements
                    for i, stmt in enumerate(statements[:5]):  # Limit statements per node
                        stmt_uri = URIRef(f"{uri}/statement/{i}")
                        
                        # Basic provenance
                        g.add((stmt_uri, RDF.type, PROV.Entity))
                        g.add((stmt_uri, PROV.wasDerivedFrom, node_uri))
                        g.add((stmt_uri, SCHEMA.about, node_uri))
                        
                        # Video information directly in statement
                        video_id = stmt.get("source_video")
                        video_title = stmt.get("video_title")
                        start_time = stmt.get("start_offset")
                        end_time = stmt.get("end_offset")
                        
                        if video_id:
                            # Create timestamped YouTube URL
                            if start_time is not None:
                                timestamped_url = f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s"
                            else:
                                timestamped_url = f"https://www.youtube.com/watch?v={video_id}"
                            
                            g.add((stmt_uri, SCHEMA.url, Literal(timestamped_url)))
                            
                            # Video title directly on statement
                            if video_title:
                                g.add((stmt_uri, SCHEMA.videoTitle, Literal(video_title)))
                        
                        # Timestamps as plain integers
                        if start_time is not None:
                            g.add((stmt_uri, SCHEMA.startTime, Literal(int(start_time))))
                        
                        if end_time is not None:
                            g.add((stmt_uri, SCHEMA.endTime, Literal(int(end_time))))
                        
                        # Transcript text if requested
                        if include_transcript and "transcript_text" in stmt:
                            transcript = stmt["transcript_text"]
                            if transcript and len(transcript.strip()) > 0:
                                # Truncate very long transcripts
                                if len(transcript) > 1000:
                                    transcript = transcript[:1000] + "..."
                                g.add((stmt_uri, SCHEMA.text, Literal(transcript)))
                        
                except Exception as e:
                    logger.warning(f"Skipping provenance for {uri}: {e}")
            
            header = f"# Provenance information generated {datetime.now(timezone.utc).isoformat()}Z\n"
            header += f"# Nodes: {len(node_uris)}, Include transcript: {include_transcript}\n"
            if include_transcript:
                header += f"# Transcript text included (truncated at 1000 chars)\n"
            header += "\n"
            
            return header + g.serialize(format="turtle")
            
        except Exception as e:
            logger.error(f"❌ Provenance turtle generation failed: {e}")
            return f"# Error: {str(e)}\n"

    def close(self):
        """Close database connection."""
        if hasattr(self, 'client') and self.client:
            self.client.close()

# --------------------------------------------------------------------------- #
#                              ENHANCED MCP SERVER                            #
# --------------------------------------------------------------------------- #

# Global instance
_querier: Optional[EnhancedGraphQuerier] = None

def get_querier() -> EnhancedGraphQuerier:
    global _querier
    if _querier is None:
        _querier = EnhancedGraphQuerier()
    return _querier

# Create MCP server
mcp = FastMCP(
    "Enhanced Parliamentary Graph Query Server - Clean Output",
    settings={
        "initialization_timeout": 60.0,
        "keep_alive_interval": 60.0,
        "max_request_size": 512 * 1024,
        "request_timeout": 120.0,
    }
)

# Add HTTP health check endpoint using FastMCP's custom_route
@mcp.custom_route("/health", methods=["GET"])
async def health_endpoint(request: Request) -> JSONResponse:
    """HTTP health check endpoint for Google Cloud Run and load balancers."""
    try:
        # Get database connection
        q = get_querier()
        
        # Test database connectivity with timeout
        start_time = time.time()
        q.client.admin.command("ping", maxTimeMS=3000)
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Quick database stats to verify collections are accessible
        node_count = q.nodes.estimated_document_count()
        edge_count = q.edges.estimated_document_count()
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database": {
                "connected": True,
                "response_time_ms": round(response_time, 2),
                "collections_accessible": True,
                "node_count": node_count,
                "edge_count": edge_count
            },
            "services": {
                "vector_search": True,  # Always true since it's mandatory
                "pagerank": node_count > 0,  # Assume PageRank available if nodes exist
                "hybrid_search": node_count > 0  # Always available when nodes exist
            },
            "version": "clean_output_v1.0"
        }
        
        return JSONResponse(
            status_code=200,
            content=health_data
        )
        
    except Exception as e:
        # Log the error but don't expose internal details in response
        logger.error(f"❌ Health check failed: {e}")
        
        error_data = {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database": {
                "connected": False,
                "error": "Database connectivity issue"
            },
            "message": "Service temporarily unavailable"
        }
        
        return JSONResponse(
            status_code=503,  # Service Unavailable
            content=error_data
        )

@mcp.tool()
def hybrid_search_turtle(query: str, hops: int = 1, limit: int = 5, 
                        pagerank_weight: float = 0.3) -> str:
    """
    Hybrid search combining PageRank importance with semantic similarity.
    Returns clean Turtle format without ranking information.
    
    Args:
        query: Free-text search string
        hops: Number of relationship hops (0-2, default 1)
        limit: Number of seed nodes (1-10, default 8)
        pagerank_weight: Weight of PageRank vs similarity (0-1, default 0.3)
    
    Returns:
        Clean Turtle-formatted RDF data with deduplicated labels
    """
    start_time = datetime.now()
    try:
        if not query or len(query.strip()) < 2:
            return "# Error: Query too short (minimum 2 characters)\n"
        
        hops = max(0, min(int(hops), 2))
        limit = max(1, min(int(limit), 10))
        pagerank_weight = max(0.0, min(float(pagerank_weight), 1.0))
        
        logger.info(f"🎯 Hybrid search: '{query}' (PageRank: {pagerank_weight:.1%}, hops: {hops})")
        
        q = get_querier()
        seeds = q.hybrid_search(query, limit, pagerank_weight)
        
        if not seeds:
            return f"# No results found for hybrid search: {query}\n"
        
        seed_uris = {node["uri"] for node in seeds if "uri" in node}
        all_uris = q.get_connected_nodes(seed_uris, hops)
        subgraph = q.get_subgraph(all_uris)
        result = q.to_turtle(subgraph)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Hybrid search completed in {elapsed:.2f}s")
        return result
        
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Hybrid search failed after {elapsed:.2f}s: {e}")
        return f"# Error: Hybrid search failed - {str(e)}\n"

@mcp.tool()
def authority_search_turtle(query: str, hops: int = 1, limit: int = 5, 
                           max_rank: int = 1000) -> str:
    """
    Search for authoritative nodes (high PageRank) related to the query.
    Returns clean Turtle format without ranking information.
    
    Args:
        query: Free-text search string
        hops: Number of relationship hops (0-2, default 1)
        limit: Number of seed nodes (1-10, default 5)
        max_rank: Maximum PageRank rank to consider (default 1000)
    
    Returns:
        Clean Turtle-formatted RDF data focusing on authoritative nodes
    """
    start_time = datetime.now()
    try:
        if not query or len(query.strip()) < 2:
            return "# Error: Query too short (minimum 2 characters)\n"
        
        hops = max(0, min(int(hops), 2))
        limit = max(1, min(int(limit), 10))
        max_rank = max(1, min(int(max_rank), 10000))
        
        logger.info(f"👑 Authority search: '{query}' (max rank: {max_rank}, hops: {hops})")
        
        q = get_querier()
        seeds = q.authority_search(query, limit, max_rank)
        
        if not seeds:
            return f"# No authoritative results found for: {query}\n"
        
        seed_uris = {node["uri"] for node in seeds if "uri" in node}
        all_uris = q.get_connected_nodes(seed_uris, hops)
        subgraph = q.get_subgraph(all_uris)
        result = q.to_turtle(subgraph)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Authority search completed in {elapsed:.2f}s")
        return result
        
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Authority search failed after {elapsed:.2f}s: {e}")
        return f"# Error: Authority search failed - {str(e)}\n"

@mcp.tool()
def topic_search_turtle(query: str, hops: int = 1, limit: int = 5) -> str:
    """
    Search for nodes important within the specific topic domain of the query.
    Returns clean Turtle format without ranking information.
    
    Args:
        query: Free-text search string defining the topic
        hops: Number of relationship hops (0-2, default 1)
        limit: Number of seed nodes (1-10, default 5)
    
    Returns:
        Clean Turtle-formatted RDF data with topic-specific importance rankings
    """
    start_time = datetime.now()
    try:
        if not query or len(query.strip()) < 2:
            return "# Error: Query too short (minimum 2 characters)\n"
        
        hops = max(0, min(int(hops), 2))
        limit = max(1, min(int(limit), 10))
        
        logger.info(f"🎯 Topic search: '{query}' (hops: {hops})")
        
        q = get_querier()
        seeds = q.topic_specific_search(query, limit)
        
        if not seeds:
            return f"# No topic-specific results found for: {query}\n"
        
        seed_uris = {node["uri"] for node in seeds if "uri" in node}
        all_uris = q.get_connected_nodes(seed_uris, hops)
        subgraph = q.get_subgraph(all_uris)
        result = q.to_turtle(subgraph)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Topic search completed in {elapsed:.2f}s")
        return result
        
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Topic search failed after {elapsed:.2f}s: {e}")
        return f"# Error: Topic search failed - {str(e)}\n"

@mcp.tool()
def search_graph_turtle(query: str, hops: int = 1, limit: int = 3) -> str:
    """
    Standard search (backward compatibility) - now uses hybrid search by default.
    Returns clean Turtle format without ranking information.
    
    Args:
        query: Free-text search string
        hops: Number of relationship hops (0-2, default 1) 
        limit: Number of seed nodes (1-10, default 3)
    
    Returns:
        Clean Turtle-formatted RDF data
    """
    # Use hybrid search as the new default with balanced weights
    return hybrid_search_turtle(query, hops, limit, pagerank_weight=0.4)

@mcp.tool()
def get_provenance_turtle(node_uris: str, include_transcript: bool = True) -> str:
    """
    Get provenance information for specific nodes in clean Turtle format.
    
    Args:
        node_uris: Comma-separated list of node URIs to get provenance for
        include_transcript: Whether to include transcript text (default True)
    
    Returns:
        Turtle-formatted provenance with video title, video ID, start time, and transcript
    """
    try:
        if not node_uris or not isinstance(node_uris, str):
            return "# Error: node_uris parameter required\n"
        
        # Parse URIs
        uris = [uri.strip() for uri in node_uris.split(",") if uri.strip()]
        if not uris:
            return "# Error: No valid URIs provided\n"
        
        # Limit to prevent token explosion
        if len(uris) > 10:
            uris = uris[:10]
        
        logger.info(f"📚 Getting provenance for {len(uris)} nodes as Turtle (transcript: {include_transcript})")
        
        q = get_querier()
        result = q.provenance_to_turtle(uris, include_transcript)
        
        logger.info(f"✅ Provenance turtle generated for {len(uris)} nodes")
        return result
        
    except Exception as e:
        logger.error(f"❌ Provenance turtle generation failed: {e}")
        return f"# Error: Provenance generation failed - {str(e)}\n"

@mcp.tool()
def health_check() -> str:
    """Check server and database health - can be called via MCP or used for monitoring."""
    try:
        logger.info("🏥 Running health check")
        q = get_querier()
        
        # Test database connectivity with timeout
        start_time = time.time()
        q.client.admin.command("ping", maxTimeMS=3000)
        response_time = (time.time() - start_time) * 1000
        
        # Quick database stats
        node_count = q.nodes.estimated_document_count()
        edge_count = q.edges.estimated_document_count()
        
        result = {
            "status": "healthy",
            "time": datetime.now(timezone.utc).isoformat(),
            "database": {
                "connected": True,
                "response_time_ms": round(response_time, 2),
                "node_count": node_count,
                "edge_count": edge_count
            },
            "services": {
                "vector_search": True,  # Always true since it's mandatory
                "pagerank": node_count > 0,
                "hybrid_search": node_count > 0
            },
            "version": "clean_output_v1.0"
        }
        
        logger.info("✅ Health check passed")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return json.dumps({
            "status": "unhealthy", 
            "error": str(e),
            "time": datetime.now(timezone.utc).isoformat(),
            "database": {"connected": False}
        })

# --------------------------------------------------------------------------- #
#                              MAIN ENTRY POINT                              #
# --------------------------------------------------------------------------- #

def cleanup():
    """Clean up resources."""
    global _querier
    if _querier:
        _querier.close()
        _querier = None

import signal
import atexit

atexit.register(cleanup)
signal.signal(signal.SIGTERM, lambda s, f: cleanup())

async def initialize_with_retry(max_retries: int = 3, delay: float = 1.0):
    """Initialize the database connection with retries."""
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Database initialization attempt {attempt + 1}/{max_retries}")
            querier = get_querier()
            logger.info("✅ Enhanced database connection established")
            return querier
        except Exception as e:
            logger.error(f"❌ Initialization attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"⏳ Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                raise

if __name__ == "__main__":
    # Get port from environment (Cloud Run compatibility)
    port = int(os.getenv("PORT", "8080"))
    host = "0.0.0.0"
    
    logger.info("🚀 Starting Enhanced Parliamentary Graph MCP Server - Clean Output")
    logger.info("✨ Features: Deduplicated labels, No ranking info, Turtle provenance")
    logger.info(f"📡 Server will run on {host}:{port}")
    logger.info(f"🏥 Health check available at http://{host}:{port}/health")
    
    try:
        # Initialize database connection with retries
        asyncio.run(initialize_with_retry())
        
        logger.info("🌐 Starting enhanced MCP server...")
        
        # Use FastMCP's built-in server with custom health route
        mcp.run(
            transport="sse",
            host=host,
            port=port,
            log_level="info",
        )
            
    except KeyboardInterrupt:
        logger.info("👋 Shutting down gracefully")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)
    finally:
        cleanup()

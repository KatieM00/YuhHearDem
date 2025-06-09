#!/usr/bin/env python3
"""
Parliamentary Graph Query MCP Server - Enhanced with Hybrid Search
----------------------------------------------------------------

Enhanced version that combines PageRank importance with semantic similarity
for better search results. Includes topic-specific PageRank and authority-aware queries.
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
from collections import defaultdict

from bson import ObjectId

# --- third-party -------------------------------------------------------------
try:
    from pymongo import MongoClient, ASCENDING
    from pymongo.errors import ConnectionFailure
    from dotenv import load_dotenv
    from rdflib import Graph, URIRef, Literal, BNode, Namespace
    from rdflib.namespace import RDF, RDFS, OWL, FOAF, XSD
    from fastmcp import FastMCP
except ImportError as e:
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
#                      ENHANCED GRAPH QUERIER WITH HYBRID SEARCH             #
# --------------------------------------------------------------------------- #
class EnhancedGraphQuerier:
    """Enhanced graph querier with hybrid PageRank + semantic search capabilities."""

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
        """Initialize embedding model with timeout protection."""
        if not VECTOR_SEARCH_AVAILABLE:
            logger.info("📄 Vector search not available")
            return
            
        try:
            logger.info("🔄  Loading embedding model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅  Vector search enabled")
        except Exception as e:
            logger.warning(f"📄 Vector search disabled: {e}")
            self.embedding_model = None

    def search_nodes(self, query: str, limit: int = 8) -> List[Dict]:
        """Standard vector-only search for backward compatibility."""
        try:
            logger.info(f"🔍 Vector search for: {query}")
            vector_results = self._vector_search_nodes(query, limit)
            
            # Clean results
            for result in vector_results:
                result.pop("embedding", None)
                if "_id" in result:
                    result["_id"] = str(result["_id"])
            
            logger.info(f"Vector search found {len(vector_results)} results")
            return vector_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def hybrid_search(self, query: str, limit: int = 8, pagerank_weight: float = 0.3, 
                     min_pagerank_score: float = 0.0001) -> List[Dict]:
        """
        Hybrid search combining PageRank importance with semantic similarity.
        
        Args:
            query: Search query text
            limit: Number of results to return
            pagerank_weight: Weight of PageRank in final score (0-1, default 0.3)
            min_pagerank_score: Minimum PageRank score to consider
            
        Returns:
            List of nodes ranked by hybrid score
        """
        if not self.embedding_model:
            logger.warning("📄 Embedding model not available, falling back to text search")
            return self._fallback_text_search(query, limit)
        
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
                                {"$ifNull": ["$pagerank_score", 0.0001]},
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
                        "type": 1,
                        "searchable_text": 1,
                        "pagerank_score": 1,
                        "pagerank_rank": 1,
                        "similarity_score": 1,
                        "hybrid_score": 1,
                        "source_video": 1,
                        "video_title": 1
                    }
                }
            ]
            
            results = list(self.nodes.aggregate(pipeline))
            
            # Clean results
            for result in results:
                if "_id" in result:
                    result["_id"] = str(result["_id"])
            
            logger.info(f"🎯 Hybrid search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            # Fall back to regular vector search
            return self.search_nodes(query, limit)

    def authority_search(self, query: str, limit: int = 8, min_pagerank_rank: int = 1000) -> List[Dict]:
        """
        Search for authoritative nodes (high PageRank) related to the query.
        
        Args:
            query: Search query text
            limit: Number of results to return
            min_pagerank_rank: Maximum rank to consider (lower numbers = higher authority)
            
        Returns:
            List of authoritative nodes related to the query
        """
        try:
            logger.info(f"👑 Authority search for: '{query}' (max rank: {min_pagerank_rank})")
            
            # First get semantically relevant nodes with good PageRank
            candidates = self.hybrid_search(
                query, 
                limit=limit * 2, 
                pagerank_weight=0.7,  # Heavily favor PageRank
                min_pagerank_score=0.0001
            )
            
            # Filter by rank and sort by authority
            authority_results = [
                node for node in candidates 
                if node.get("pagerank_rank", float('inf')) <= min_pagerank_rank
            ]
            
            # Sort by PageRank rank (lower is better)
            authority_results.sort(key=lambda x: x.get("pagerank_rank", float('inf')))
            
            logger.info(f"👑 Found {len(authority_results)} authoritative nodes")
            return authority_results[:limit]
            
        except Exception as e:
            logger.error(f"❌ Authority search failed: {e}")
            return []

    def topic_specific_search(self, query: str, limit: int = 8, topic_expansion: int = 50) -> List[Dict]:
        """
        Find nodes important within the specific topic domain of the query.
        
        Args:
            query: Search query defining the topic
            limit: Number of results to return
            topic_expansion: Number of topic-relevant nodes to consider for local PageRank
            
        Returns:
            List of nodes important within the query's topic domain
        """
        try:
            logger.info(f"🎯 Topic-specific search for: '{query}'")
            
            # Find semantically relevant nodes to define the topic
            topic_nodes = self.hybrid_search(
                query, 
                limit=topic_expansion,
                pagerank_weight=0.1,  # Favor similarity for topic definition
                min_pagerank_score=0.0
            )
            
            if not topic_nodes:
                logger.warning("No topic nodes found")
                return []
            
            # Extract URIs of topic-relevant nodes
            topic_uris = {node['uri'] for node in topic_nodes}
            
            # Calculate mini-PageRank within this topic subgraph
            topic_pagerank_scores = self._calculate_topic_pagerank(topic_uris)
            
            # Combine topic PageRank with original relevance
            enhanced_results = []
            for node in topic_nodes:
                enhanced_node = node.copy()
                uri = node['uri']
                if uri in topic_pagerank_scores:
                    enhanced_node['topic_pagerank'] = topic_pagerank_scores[uri]
                    # Hybrid score: topic importance + semantic relevance
                    enhanced_node['topic_hybrid_score'] = (
                        0.6 * topic_pagerank_scores[uri] + 
                        0.4 * node.get('similarity_score', 0)
                    )
                else:
                    enhanced_node['topic_pagerank'] = 0.0
                    enhanced_node['topic_hybrid_score'] = 0.4 * node.get('similarity_score', 0)
                
                enhanced_results.append(enhanced_node)
            
            # Sort by topic-specific hybrid score
            enhanced_results.sort(key=lambda x: x['topic_hybrid_score'], reverse=True)
            
            logger.info(f"🎯 Topic-specific search found {len(enhanced_results)} results")
            return enhanced_results[:limit]
            
        except Exception as e:
            logger.error(f"❌ Topic-specific search failed: {e}")
            return self.hybrid_search(query, limit)

    def _calculate_topic_pagerank(self, topic_uris: Set[str], damping: float = 0.85, 
                                 max_iterations: int = 50) -> Dict[str, float]:
        """
        Calculate PageRank within a topic-specific subgraph.
        
        Args:
            topic_uris: URIs defining the topic subgraph
            damping: PageRank damping factor
            max_iterations: Maximum iterations for convergence
            
        Returns:
            Dictionary mapping URIs to topic-specific PageRank scores
        """
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
        if not self.embedding_model:
            return self._fallback_text_search(query, limit)

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # MongoDB vector search pipeline
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
                {"$addFields": {"similarity_score": {"$meta": "vectorSearchScore"}}},
            ]

            results = list(self.nodes.aggregate(pipeline))
            logger.info(f"Vector search pipeline returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self._fallback_text_search(query, limit)

    def _fallback_text_search(self, query: str, limit: int = 8) -> List[Dict]:
        """Fallback text search when vector search is unavailable."""
        try:
            regex = re.compile(re.escape(query), re.IGNORECASE)
            results = list(
                self.nodes.find(
                    {
                        "$or": [
                            {"label": {"$regex": regex}},
                            {"local_name": {"$regex": regex}},
                            {"searchable_text": {"$regex": regex}},
                        ]
                    },
                    {"embedding": 0}
                ).limit(limit)
            )
            
            # Add PageRank boost to text search results
            for result in results:
                result["similarity_score"] = 1.0  # Dummy similarity for consistency
                if "_id" in result:
                    result["_id"] = str(result["_id"])
            
            # Sort by PageRank if available
            results.sort(key=lambda x: x.get("pagerank_score", 0), reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"Text search fallback failed: {e}")
            return []

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
        """Get subgraph for the given URIs."""
        try:
            if len(uris) > 500:
                uris = set(list(uris)[:500])
            
            nodes = list(self.nodes.find({"uri": {"$in": list(uris)}}, {"embedding": 0}))
            edges = list(self.edges.find({
                "subject": {"$in": list(uris)}, 
                "object": {"$in": list(uris)}
            }))
            
            # Clean MongoDB objects
            for node in nodes:
                if "_id" in node:
                    node["_id"] = str(node["_id"])
            for edge in edges:
                if "_id" in edge:
                    edge["_id"] = str(edge["_id"])
            
            return {"nodes": nodes, "edges": edges}
            
        except Exception as e:
            logger.error(f"Subgraph retrieval failed: {e}")
            return {"nodes": [], "edges": []}

    def to_turtle(self, subgraph: Dict[str, Any]) -> str:
        """Convert subgraph to Turtle format with PageRank annotations."""
        try:
            g = Graph()
            
            # Add prefixes
            g.bind("bbp", "http://example.com/barbados-parliament-ontology#")
            g.bind("sess", "http://example.com/barbados-parliament-session/")
            g.bind("rdfs", RDFS)
            g.bind("rdf", RDF)
            g.bind("pg", "http://example.com/pagerank/")  # PageRank namespace
            
            # Add nodes with PageRank information
            for node in subgraph["nodes"]:
                try:
                    uri = URIRef(node["uri"])
                    if "label" in node:
                        g.add((uri, RDFS.label, Literal(str(node["label"]))))
                    for t in node.get("type", []):
                        g.add((uri, RDF.type, URIRef(t)))
                    
                    # Add PageRank information if available
                    if "pagerank_score" in node and node["pagerank_score"] is not None:
                        g.add((uri, URIRef("http://example.com/pagerank/score"), 
                              Literal(float(node["pagerank_score"]))))
                    if "pagerank_rank" in node and node["pagerank_rank"] is not None:
                        g.add((uri, URIRef("http://example.com/pagerank/rank"), 
                              Literal(int(node["pagerank_rank"]))))
                    
                    # Add search scores if available
                    if "similarity_score" in node and node["similarity_score"] is not None:
                        g.add((uri, URIRef("http://example.com/pagerank/similarity"), 
                              Literal(float(node["similarity_score"]))))
                    if "hybrid_score" in node and node["hybrid_score"] is not None:
                        g.add((uri, URIRef("http://example.com/pagerank/hybrid_score"), 
                              Literal(float(node["hybrid_score"]))))
                    
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
            header += f"# Enhanced with PageRank and hybrid search scores\n\n"
            
            return header + g.serialize(format="turtle")
            
        except Exception as e:
            logger.error(f"Turtle serialization failed: {e}")
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
    "Enhanced Parliamentary Graph Query Server",
    settings={
        "initialization_timeout": 60.0,
        "keep_alive_interval": 60.0,
        "max_request_size": 512 * 1024,
        "request_timeout": 120.0,
    }
)

@mcp.tool()
def hybrid_search_turtle(query: str, hops: int = 1, limit: int = 5, 
                        pagerank_weight: float = 0.3) -> str:
    """
    Hybrid search combining PageRank importance with semantic similarity.
    
    Args:
        query: Free-text search string
        hops: Number of relationship hops (0-2, default 1)
        limit: Number of seed nodes (1-10, default 5)
        pagerank_weight: Weight of PageRank vs similarity (0-1, default 0.3)
    
    Returns:
        Turtle-formatted RDF data with nodes ranked by hybrid importance
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
    
    Args:
        query: Free-text search string
        hops: Number of relationship hops (0-2, default 1)
        limit: Number of seed nodes (1-10, default 5)
        max_rank: Maximum PageRank rank to consider (default 1000)
    
    Returns:
        Turtle-formatted RDF data focusing on authoritative/important nodes
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
    
    Args:
        query: Free-text search string defining the topic
        hops: Number of relationship hops (0-2, default 1)
        limit: Number of seed nodes (1-10, default 5)
    
    Returns:
        Turtle-formatted RDF data with topic-specific importance rankings
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
    
    Args:
        query: Free-text search string
        hops: Number of relationship hops (0-2, default 1) 
        limit: Number of seed nodes (1-10, default 3)
    
    Returns:
        Turtle-formatted RDF data
    """
    # Use hybrid search as the new default with balanced weights
    return hybrid_search_turtle(query, hops, limit, pagerank_weight=0.4)

@mcp.tool()
def get_provenance(node_uris: str, include_transcript: bool = False) -> str:
    """
    Get detailed provenance information for specific nodes.
    
    Args:
        node_uris: Comma-separated list of node URIs to get provenance for
        include_transcript: Whether to include full transcript text (default False)
    
    Returns:
        JSON with detailed source information, video segments, and timestamps
    """
    try:
        if not node_uris or not isinstance(node_uris, str):
            return json.dumps({"error": "node_uris parameter required"}, indent=2)
        
        # Parse URIs
        uris = [uri.strip() for uri in node_uris.split(",") if uri.strip()]
        if not uris:
            return json.dumps({"error": "No valid URIs provided"}, indent=2)
        
        # Limit to prevent token explosion
        if len(uris) > 10:
            uris = uris[:10]
        
        logger.info(f"📚 Getting provenance for {len(uris)} nodes")
        
        q = get_querier()
        provenance_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_count": len(uris),
            "include_transcript": include_transcript,
            "provenance": {}
        }
        
        for uri in uris:
            try:
                # Get node information
                node = q.nodes.find_one({"uri": uri}, {"embedding": 0})
                if not node:
                    provenance_data["provenance"][uri] = {"error": "Node not found"}
                    continue
                
                # Get related statements with provenance
                if include_transcript:
                    # Include all fields when transcript is requested
                    statements = list(q.statements.find({
                        "$or": [
                            {"subject": uri},
                            {"predicate": uri}, 
                            {"object": uri}
                        ]
                    }))
                else:
                    # Include specific fields but exclude transcript
                    statements = list(q.statements.find({
                        "$or": [
                            {"subject": uri},
                            {"predicate": uri}, 
                            {"object": uri}
                        ]
                    }, {
                        "statement_id": 1,
                        "subject": 1,
                        "predicate": 1,
                        "object": 1,
                        "source_video": 1,
                        "video_title": 1,
                        "from_video": 1,
                        "start_offset": 1,
                        "end_offset": 1,
                        "segment_type": 1,
                        "_id": 1
                    }))
                
                # Build compact provenance info
                node_provenance = {
                    "uri": uri,
                    "label": node.get("label", "Unknown"),
                    "type": node.get("type", []),
                    "source_videos": list(set(node.get("source_video", []))),
                    "video_title": node.get("video_title", "Unknown"),
                    "pagerank_rank": node.get("pagerank_rank"),
                    "statement_count": len(statements),
                    "video_segments": []
                }
                
                # Group statements by video segment for efficiency
                segments = {}
                for stmt in statements:
                    video_id = stmt.get("source_video", "unknown")
                    start_time = stmt.get("start_offset", 0)
                    end_time = stmt.get("end_offset", 0)
                    
                    # Create segment key (rounded to nearest 30 seconds for grouping)
                    segment_key = f"{video_id}_{int(start_time // 30) * 30}"
                    
                    if segment_key not in segments:
                        segments[segment_key] = {
                            "video_id": video_id,
                            "video_title": stmt.get("video_title", "Unknown"),
                            "start_time": start_time,
                            "end_time": end_time,
                            "statement_count": 0,
                            "roles": set()
                        }
                        
                        if include_transcript:
                            segments[segment_key]["transcript_preview"] = stmt.get("transcript_text", "")[:200]
                    
                    segments[segment_key]["statement_count"] += 1
                    segments[segment_key]["end_time"] = max(segments[segment_key]["end_time"], end_time)
                    
                    # Track roles if this is about a person
                    if "Person" in str(node.get("type", [])):
                        role_info = stmt.get("transcript_text", "")
                        if any(word in role_info.lower() for word in ["minister", "member", "mp", "speaker"]):
                            segments[segment_key]["roles"].add("parliamentary_member")
                
                # Convert to list and clean up
                for segment in segments.values():
                    segment["roles"] = list(segment["roles"])
                    segment["duration"] = round(segment["end_time"] - segment["start_time"], 1)
                    
                    # Format timestamps as MM:SS
                    def format_time(seconds):
                        mins = int(seconds // 60)
                        secs = int(seconds % 60)
                        return f"{mins}:{secs:02d}"
                    
                    segment["time_range"] = f"{format_time(segment['start_time'])}-{format_time(segment['end_time'])}"
                    
                    node_provenance["video_segments"].append(segment)
                
                # Sort segments by video and time
                node_provenance["video_segments"].sort(key=lambda x: (x["video_id"], x["start_time"]))
                
                # Add summary stats
                total_duration = sum(seg["duration"] for seg in node_provenance["video_segments"])
                unique_videos = len(set(seg["video_id"] for seg in node_provenance["video_segments"]))
                
                node_provenance["summary"] = {
                    "total_video_time": f"{int(total_duration // 60)}:{int(total_duration % 60):02d}",
                    "unique_videos": unique_videos,
                    "total_segments": len(node_provenance["video_segments"])
                }
                
                provenance_data["provenance"][uri] = node_provenance
                
            except Exception as e:
                logger.error(f"❌ Failed to get provenance for {uri}: {e}")
                provenance_data["provenance"][uri] = {"error": str(e)}
        
        logger.info(f"✅ Provenance retrieved for {len(provenance_data['provenance'])} nodes")
        return json.dumps(provenance_data, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Provenance retrieval failed: {e}")
        return json.dumps({
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, indent=2)

@mcp.tool()
def health_check() -> str:
    """Check server health."""
    try:
        logger.info("🏥 Running health check")
        q = get_querier()
        q.client.admin.command("ping", maxTimeMS=2000)
        
        result = {
            "status": "healthy",
            "time": datetime.now(timezone.utc).isoformat(),
            "database": "connected",
            "vector_search": q.embedding_model is not None,
            "hybrid_search_ready": q.embedding_model is not None
        }
        
        logger.info("✅ Health check passed")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return json.dumps({
            "status": "unhealthy", 
            "error": str(e),
            "time": datetime.now(timezone.utc).isoformat(),
            "database": "disconnected"
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
    port = int(os.getenv("PORT", "8080"))
    host = "0.0.0.0"
    
    logger.info("🚀 Starting Enhanced Parliamentary Graph MCP Server")
    logger.info("🎯 Features: Hybrid Search, Authority Ranking, Topic-Specific PageRank")
    logger.info(f"📡 Server will run on {host}:{port}")
    
    try:
        asyncio.run(initialize_with_retry())
        logger.info("🌐 Starting enhanced server...")
        
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
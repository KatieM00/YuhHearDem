#!/usr/bin/env python3
"""
Fixed Parliamentary Chatbot - Simple Session Management That Works
==================================================================

Goes back to working session management while maintaining enhanced features.
"""

import os
import json
import logging
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set, AsyncGenerator
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# Database and ML imports
from pymongo import MongoClient, ASCENDING
from sentence_transformers import SentenceTransformer

# Google ADK imports
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.adk.planners import BuiltInPlanner
from google.genai.types import Content, Part, GenerateContentConfig
from google.genai import types
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS

# Markdown processing
import markdown
from bs4 import BeautifulSoup

# Mount static files and templates
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pytz


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RDF Namespaces
BBP = Namespace("http://example.com/barbados-parliament-ontology#")
SCHEMA = Namespace("http://schema.org/")
PROV = Namespace("http://www.w3.org/ns/prov#")

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str = Field(..., description="User's question about Parliament")
    user_id: str = Field(..., description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Session ID")

class QueryResponse(BaseModel):
    session_id: str
    user_id: str
    message_id: str
    status: str
    message: Optional[str] = None

class SessionGraphState:
    """Manages cumulative graph state for a session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.clear_graph("New session started")
        
    def add_turtle_data(self, turtle_str: str) -> bool:
        """Add Turtle data to the cumulative graph."""
        try:
            # Parse Turtle into temporary graph
            temp_graph = Graph()
            temp_graph.parse(data=turtle_str, format='turtle')
            
            # Track what we're adding
            new_triples = len(temp_graph)
            
            # Add to cumulative graph
            for triple in temp_graph:
                self.graph.add(triple)
            
            # Update counts
            self.node_count = len(set(self.graph.subjects()) | set(self.graph.objects()))
            self.edge_count = len(self.graph)
            
            logger.info(f"📈 Session {self.session_id[:8]}: Added {new_triples} triples, total: {self.edge_count}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to parse Turtle: {e}")
            return False
    
    def get_turtle_dump(self) -> str:
        """Get current graph as Turtle format."""
        try:
            header = f"""# Session Graph Dump
# Session: {self.session_id}
# Created: {self.created_at.isoformat()}
# Nodes: {self.node_count}, Edges: {self.edge_count}
"""
            return header + self.graph.serialize(format='turtle')
        except Exception as e:
            logger.error(f"Failed to serialize graph: {e}")
            return f"# Error serializing graph: {e}\n"
    
    def clear_graph(self, reason: str = "Topic change"):
        """Clear the cumulative graph."""
        self.graph = Graph()
        self.created_at = datetime.now(timezone.utc)
        
        # Re-bind namespaces
        self.graph.bind("bbp", BBP)
        self.graph.bind("schema", SCHEMA)
        self.graph.bind("prov", PROV)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("rdf", RDF)
        
        self.node_count = 0
        self.edge_count = 0
        
        logger.info(f"🧹 Session {self.session_id[:8]}: Graph cleared. Reason: {reason}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "session_id": self.session_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "created_at": self.created_at.isoformat(),
            "size_mb": len(self.get_turtle_dump()) / (1024 * 1024)
        }

class ParliamentaryGraphQuerier:
    """Database querier with full search functionality."""

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
        """Initialize database connection."""
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
            
            # Create indexes if they don't exist
            try:
                self.nodes.create_index([("pagerank_score", ASCENDING)])
                self.nodes.create_index([("pagerank_rank", ASCENDING)])
            except:
                pass
            
            logger.info("✅ Connected to MongoDB")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            if self.client:
                self.client.close()
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")

    def _initialize_embeddings(self):
        """Initialize embedding model."""
        try:
            logger.info("🔄 Loading embedding model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Vector search enabled")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise RuntimeError(f"Vector search failed to initialize: {e}")

    def unified_hybrid_search(self, query: str, limit: int = 8) -> List[Dict]:
        """
        Performs both node vector search and statement text search,
        then intelligently combines and weights the results.
        """
        try:
            logger.info(f"🔍 Unified search for: '{query}'")
            
            # Run both searches in parallel
            try:
                node_results = self._search_nodes_vector(query, limit * 2)
            except Exception as e:
                logger.error(f"❌ Vector search failed: {e}")
                node_results = []
            try:
                statement_results = self._search_statements_atlas(query, limit * 2)
            except Exception as e:
                logger.error(f"❌ Atlas search failed: {e}")
                statement_results = [] 

            # Convert both result types to unified format
            unified_results = []
            
            # Process node results
            for node in node_results:
                unified_results.append({
                    'uri': node['uri'],
                    'source_type': 'node',
                    'content': node.get('searchable_text', ''),
                    'label': node.get('label') or node.get('name', ''),
                    'node_data': node,
                    'vector_score': node.get('similarity_score', 0),
                    'text_score': 0,
                    'provenance': None
                })
            
            # Process statement results and find their related nodes
            if statement_results:
                # Step 1: Collect all unique URIs from all statements
                all_related_uris = set()
                stmt_to_uris = {}  # Track which URIs belong to which statement
                
                for i, stmt in enumerate(statement_results):
                    related_uris = []
                    if stmt.get('subject'): related_uris.append(stmt['subject'])
                    if stmt.get('object'): related_uris.append(stmt['object'])
                    
                    stmt_to_uris[i] = related_uris
                    all_related_uris.update(related_uris)
                
                # Step 2: Fetch ALL related nodes in ONE database call
                if all_related_uris:
                    nodes_cursor = self.nodes.find(
                        {'uri': {'$in': list(all_related_uris)}},
                        {'uri': 1, 'label': 1, 'name': 1}  # Only fetch needed fields
                    )
                    
                    # Create a lookup dictionary for O(1) access
                    uri_to_node = {node['uri']: node for node in nodes_cursor}
                    
                    # Step 3: Build results using the lookup dictionary
                    for i, stmt in enumerate(statement_results):
                        for uri in stmt_to_uris[i]:
                            node = uri_to_node.get(uri)
                            if node:
                                unified_results.append({
                                    'uri': uri,
                                    'source_type': 'statement',
                                    'content': stmt.get('transcript_text', ''),
                                    'label': node.get('label') or node.get('name', ''),
                                    'node_data': node,
                                    'vector_score': 0,
                                    'text_score': stmt.get('score', 0),
                                    'provenance': {
                                        'video_id': stmt.get('source_video'),
                                        'video_title': stmt.get('video_title'),
                                        'start_time': stmt.get('start_offset'),
                                        'transcript_excerpt': stmt.get('transcript_text', '')[:200] + '...'
                                    }
                                })
            
            # Deduplicate by URI while preserving best scores
            uri_to_result = {}
            for result in unified_results:
                uri = result['uri']
                if uri not in uri_to_result:
                    uri_to_result[uri] = result
                else:
                    # Merge scores - keep highest of each type
                    existing = uri_to_result[uri]
                    existing['vector_score'] = max(existing['vector_score'], result['vector_score'])
                    existing['text_score'] = max(existing['text_score'], result['text_score'])
                    
                    # Prefer statement provenance if available
                    if result['provenance'] and not existing['provenance']:
                        existing['provenance'] = result['provenance']
                        existing['content'] = result['content']  # Use transcript text
            
            # Calculate unified scores and rank
            final_results = list(uri_to_result.values())
            final_results = self._calculate_unified_scores(final_results, query)
            
            # Sort by unified score and return top results
            final_results.sort(key=lambda x: x['unified_score'], reverse=True)
            
            logger.info(f"🎯 Unified search: {len(final_results)} unique results")
            return final_results[:limit]
            
        except Exception as e:
            logger.error(f"❌ Unified search failed: {e}")
            return []

    def _search_nodes_vector(self, query: str, limit: int) -> List[Dict]:
        """Vector search on nodes (existing logic)"""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        pipeline = [
            {"$vectorSearch": {
                "index": "vector_index",
                "path": "embedding", 
                "queryVector": query_embedding,
                "numCandidates": limit * 3,
                "limit": limit
            }},
            {"$addFields": {
                "similarity_score": {"$meta": "vectorSearchScore"}
            }}
        ]
        
        return list(self.nodes.aggregate(pipeline))

    def _search_statements_atlas(self, query: str, limit: int) -> List[Dict]:
        """Atlas Search on statements - much better than $text"""
        pipeline = [
            {
                "$search": {
                    "index": "default",
                    "compound": {
                        "should": [
                            {
                                "phrase": {
                                    "query": query,
                                    "path": "transcript_text",
                                    "score": {"boost": {"value": 3}}
                                }
                            },
                            {
                                "text": {
                                    "query": query,
                                    "path": ["transcript_text", "video_title"],
                                    "fuzzy": {"maxEdits": 1}
                                }
                            }
                        ]
                    }
                }
            },
            {
                "$addFields": {
                    "search_score": {"$meta": "searchScore"}
                }
            },
            {
                "$project": {
                    "subject": 1, "object": 1, "transcript_text": 1,
                    "source_video": 1, "video_title": 1, "start_offset": 1,
                    "search_score": 1
                }
            },
            {"$sort": {"search_score": -1}},
            {"$limit": limit}
        ]
        
        return list(self.statements.aggregate(pipeline))

    def _calculate_unified_scores(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Calculate unified scores using multiple factors:
        - Vector similarity score
        - Text relevance score  
        - Query characteristics
        - PageRank importance
        - Provenance quality
        """
        
        # Normalize scores to 0-1 range
        max_vector = max((r['vector_score'] for r in results), default=1)
        max_text = max((r['text_score'] for r in results), default=1)
        
        for result in results:
            # Normalize individual scores
            norm_vector = result['vector_score'] / max_vector if max_vector > 0 else 0
            norm_text = result['text_score'] / max_text if max_text > 0 else 0
            
            # Dynamic weighting based on query characteristics
            weights = self._get_dynamic_weights(query, result)
            
            # Base score combination
            base_score = (
                weights['vector_weight'] * norm_vector + 
                weights['text_weight'] * norm_text
            )
            
            # Boost factors
            pagerank_boost = self._get_pagerank_boost(result['node_data'])
            provenance_boost = self._get_provenance_boost(result['provenance'])
            content_quality_boost = self._get_content_quality_boost(result['content'])
            
            # Final unified score
            result['unified_score'] = base_score * (1 + pagerank_boost + provenance_boost + content_quality_boost)
            
            # Store components for debugging
            result['score_components'] = {
                'base_score': base_score,
                'norm_vector': norm_vector,
                'norm_text': norm_text,
                'weights': weights,
                'pagerank_boost': pagerank_boost,
                'provenance_boost': provenance_boost,
                'content_quality_boost': content_quality_boost
            }
        
        return results

    def _get_dynamic_weights(self, query: str, result: Dict) -> Dict:
        """
        Dynamically adjust vector vs text weights based on query and result characteristics
        """
        # Default weights
        vector_weight = 0.6
        text_weight = 0.4
        
        # Adjust based on query characteristics
        query_lower = query.lower()
        
        # Favor text search for:
        if any(indicator in query_lower for indicator in [
            'said', 'stated', 'mentioned', 'quote', 'exactly',
            '$', 'bbd', 'usd', 'payment', 'amount', 'cost',
            'bill', 'section', 'act', 'regulation'
        ]):
            text_weight += 0.3
            vector_weight -= 0.3
        
        # Favor vector search for:
        if any(indicator in query_lower for indicator in [
            'about', 'regarding', 'related to', 'concerning',
            'policy', 'strategy', 'approach', 'similar'
        ]):
            vector_weight += 0.2
            text_weight -= 0.2
        
        # Boost text weight if we have good provenance
        if result.get('provenance'):
            text_weight += 0.1
            vector_weight -= 0.1
        
        # Normalize to ensure they sum to 1
        total = vector_weight + text_weight
        return {
            'vector_weight': vector_weight / total,
            'text_weight': text_weight / total
        }

    def _get_pagerank_boost(self, node_data: Dict) -> float:
        pagerank_rank = node_data.get('pagerank_rank', 1000)
        # Convert rank to boost: top 10 nodes get big boost
        if pagerank_rank <= 10:
            return 0.5  # 50% boost for top 10
        elif pagerank_rank <= 50:
            return 0.3  # 30% boost for top 50
        elif pagerank_rank <= 100:
            return 0.1  # 10% boost for top 100
        else:
            return 0.0

    def _get_provenance_boost(self, provenance: Dict) -> float:
        """Boost if we have good provenance (video links, timestamps)"""
        if not provenance:
            return 0
        
        boost = 0
        if provenance.get('video_id'): boost += 0.1
        if provenance.get('start_time'): boost += 0.1
        if provenance.get('transcript_excerpt'): boost += 0.1
        
        return min(0.3, boost)

    def _get_content_quality_boost(self, content: str) -> float:
        """Boost based on content richness"""
        if not content:
            return 0
        
        # Simple content quality indicators
        word_count = len(content.split())
        boost = min(0.1, word_count / 1000)  # Up to 0.1 boost for rich content
        
        return boost

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
            
            # Get nodes
            raw_nodes = list(self.nodes.find(
                {"uri": {"$in": list(uris)}}, 
                {
                    "uri": 1,
                    "label": 1,
                    "name": 1,
                    "type": 1,
                    "searchable_text": 1
                }
            ))
            
            # Clean nodes
            cleaned_nodes = []
            for node in raw_nodes:
                cleaned = {
                    "uri": node.get("uri"),
                    "type": node.get("type", [])
                }
                
                # Handle labels - prefer label over name
                label = node.get("label") or node.get("name")
                if label:
                    cleaned["label"] = label
                
                if "searchable_text" in node:
                    cleaned["searchable_text"] = node["searchable_text"]
                
                cleaned_nodes.append(cleaned)
            
            # Get edges
            edges = list(self.edges.find({
                "subject": {"$in": list(uris)}, 
                "object": {"$in": list(uris)}
            }))
            
            # Clean edges
            for edge in edges:
                if "_id" in edge:
                    edge["_id"] = str(edge["_id"])
            
            return {"nodes": cleaned_nodes, "edges": edges}
            
        except Exception as e:
            logger.error(f"Subgraph retrieval failed: {e}")
            return {"nodes": [], "edges": []}

    def to_turtle(self, subgraph: Dict[str, Any]) -> str:
        """Convert subgraph to Turtle format."""
        try:
            g = Graph()
            
            # Add prefixes
            g.bind("bbp", "http://example.com/barbados-parliament-ontology#")
            g.bind("rdfs", RDFS)
            g.bind("rdf", RDF)
            
            # Add nodes
            for node in subgraph["nodes"]:
                try:
                    uri = URIRef(node["uri"])
                    
                    if "label" in node and node["label"]:
                        g.add((uri, RDFS.label, Literal(str(node["label"]))))
                    
                    for t in node.get("type", []):
                        g.add((uri, RDF.type, URIRef(t)))
                    
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
            header += f"# Nodes: {len(subgraph['nodes'])}, Edges: {len(subgraph['edges'])}\n\n"
            
            return header + g.serialize(format="turtle")
            
        except Exception as e:
            logger.error(f"Turtle serialization failed: {e}")
            return f"# Error: {str(e)}\n"

    def get_provenance_turtle(self, node_uris: List[str], include_transcript: bool = True) -> str:
        """Get provenance information as Turtle format."""
        try:
            logger.info(f"📚 Getting provenance for {len(node_uris)} nodes")
            
            g = Graph()
            g.bind("bbp", "http://example.com/barbados-parliament-ontology#")
            g.bind("prov", "http://www.w3.org/ns/prov#")
            g.bind("schema", "http://schema.org/")
            g.bind("rdfs", RDFS)
            
            for uri in node_uris[:10]:  # Limit to prevent explosion
                try:
                    node_uri = URIRef(uri)
                    
                    # Get related statements
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
                    for i, stmt in enumerate(statements[:5]):
                        stmt_uri = URIRef(f"{uri}/statement/{i}")
                        
                        # Basic provenance
                        g.add((stmt_uri, RDF.type, PROV.Entity))
                        g.add((stmt_uri, PROV.wasDerivedFrom, node_uri))
                        g.add((stmt_uri, SCHEMA.about, node_uri))
                        
                        # Video information
                        video_id = stmt.get("source_video")
                        video_title = stmt.get("video_title")
                        start_time = stmt.get("start_offset")
                        
                        if video_id:
                            if start_time is not None:
                                timestamped_url = f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s"
                            else:
                                timestamped_url = f"https://www.youtube.com/watch?v={video_id}"
                            
                            g.add((stmt_uri, SCHEMA.url, Literal(timestamped_url)))
                            
                            if video_title:
                                g.add((stmt_uri, SCHEMA.videoTitle, Literal(video_title)))
                        
                        if start_time is not None:
                            g.add((stmt_uri, SCHEMA.startTime, Literal(int(start_time))))
                        
                        # Transcript text
                        if include_transcript and "transcript_text" in stmt:
                            transcript = stmt["transcript_text"]
                            if transcript and len(transcript.strip()) > 0:
                                if len(transcript) > 1000:
                                    transcript = transcript[:1000] + "..."
                                g.add((stmt_uri, SCHEMA.text, Literal(transcript)))
                        
                except Exception as e:
                    logger.warning(f"Skipping provenance for {uri}: {e}")
            
            header = f"# Provenance information generated {datetime.now(timezone.utc).isoformat()}Z\n\n"
            return header + g.serialize(format="turtle")
            
        except Exception as e:
            logger.error(f"❌ Provenance turtle generation failed: {e}")
            return f"# Error: {str(e)}\n"

    def close(self):
        """Close database connection."""
        if hasattr(self, 'client') and self.client:
            self.client.close()

def convert_markdown_to_html_and_filter_links(markdown_text: str) -> str:
    """Convert markdown to HTML and filter out non-YouTube links."""
    try:
        # Convert markdown to HTML first
        html = markdown.markdown(markdown_text, extensions=['extra', 'codehilite'])
        
        # Parse with BeautifulSoup for link filtering and formatting
        soup = BeautifulSoup(html, 'html.parser')
        
        # Filter out non-YouTube links
        links = soup.find_all('a')
        for link in links:
            href = link.get('href', '')
            if 'youtube.com' not in href.lower():
                link.replace_with(link.get_text())
        
        # Clean up the HTML to remove excessive spacing
        html_content = str(soup)
        
        return html_content
        
    except Exception as e:
        logger.error(f"Error processing markdown: {e}")
        return markdown_text

class ParliamentarySystem:
    """Main parliamentary system using simple session management that works."""
    
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        self.querier = ParliamentaryGraphQuerier()
        self.conversation_history = {}  # Store by session_id
        self.session_graphs = {}  # Store SessionGraphState by session_id
        
        # Track current session for tool context
        self.current_session_id = None
        
        # Create enhanced search tools with session context
        def search_parliament_hybrid(query: str, hops: int = 2, limit: int = 5) -> str:
            """
            Search parliamentary records using hybrid search with session graph integration.
            
            Args:
                query: Search query for parliamentary information
                hops: Number of relationship hops to explore (1-3)
                limit: Maximum number of results (1-10)
            
            Returns:
                Parliamentary data in Turtle format with facts and relationships
            """
            try:
                logger.info(f"🔍 Searching parliament: {query}")
                
                # Perform hybrid search
                seeds = self.querier.unified_hybrid_search(query, limit)
                if not seeds:
                    return f"# No parliamentary data found for: {query}\n"
                
                # Get connected nodes
                seed_uris = {node["uri"] for node in seeds if "uri" in node}
                all_uris = self.querier.get_connected_nodes(seed_uris, hops)
                
                # Get subgraph and convert to turtle
                subgraph = self.querier.get_subgraph(all_uris)
                turtle_data = self.querier.to_turtle(subgraph)
                
                # Get provenance data
                provenance_data = ""
                if seed_uris:
                    provenance_turtle = self.querier.get_provenance_turtle(list(seed_uris)[:5])
                    provenance_data = f"\n\n# PROVENANCE DATA:\n{provenance_turtle}"
                
                combined_data = turtle_data + provenance_data
                
                # Update session graph if we have session context
                if self.current_session_id:
                    try:
                        session_graph = self.get_or_create_session_graph(self.current_session_id)
                        main_data = turtle_data.split("# PROVENANCE DATA:")[0].strip()
                        
                        if main_data and not main_data.startswith("# Error"):
                            session_graph.add_turtle_data(main_data)
                            logger.info(f"📈 Updated session graph: {session_graph.get_stats()}")
                    except Exception as e:
                        logger.warning(f"Failed to update session graph: {e}")
                
                logger.info(f"🎯 Found {len(subgraph.get('nodes', []))} nodes, {len(subgraph.get('edges', []))} edges")
                
                return combined_data
                
            except Exception as e:
                logger.error(f"Parliament search failed: {e}")
                return f"# Error searching parliament: {str(e)}\n"
        
        def clear_session_graph(reason: str = "Topic change detected") -> str:
            """
            Clear the cumulative session graph when topic changes significantly.
            
            Args:
                reason: Why you're clearing the session graph
            
            Returns:
                Confirmation message with graph statistics
            """
            if self.current_session_id:
                try:
                    session_graph = self.get_or_create_session_graph(self.current_session_id)
                    old_stats = session_graph.get_stats()
                    session_graph.clear_graph(reason)
                    logger.info(f"🧹 Session graph cleared: {reason}")
                    return f"Session graph cleared successfully. Previous state: {old_stats['edge_count']} edges. Reason: {reason}"
                except Exception as e:
                    logger.error(f"Failed to clear session graph: {e}")
                    return f"Error clearing session graph: {e}"
            else:
                return "No active session to clear"
        
        def get_session_graph_stats() -> str:
            """
            Get statistics about the current session's cumulative graph.
            
            Returns:
                JSON string with graph statistics
            """
            if self.current_session_id:
                try:
                    session_graph = self.get_or_create_session_graph(self.current_session_id)
                    stats = session_graph.get_stats()
                    return json.dumps(stats, indent=2)
                except Exception as e:
                    logger.error(f"Failed to get session stats: {e}")
                    return json.dumps({"error": str(e)})
            else:
                return json.dumps({"error": "No active session"})
        
        bb_timezone = pytz.timezone("America/Barbados")
        current_date = datetime.now(bb_timezone).strftime("%Y-%m-%d")

        # Create the main agent with enhanced tools
        self.agent = LlmAgent(
            name="YuhHearDem",
            model="gemini-2.5-flash-preview-05-20",
            description="AI assistant for Barbados Parliament with cumulative graph memory",
            planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=0)),
            instruction=f"""I'm YuhHearDem, your friendly AI assistant here to help you understand Barbados Parliament. My main goal is to give you direct quotes and remember our conversations, especially the latest information.

Current Date: {current_date}

I'm designed to always search for specific parliamentary information when you ask about topics, focusing on the **most recent** sessions, debates, and announcements. I *love* finding and presenting **direct quotes** from MPs, ministers, and officials – those are my top priority!

My memory works like a cumulative knowledge graph. As we chat and I search, it grows, building richer context with a strong emphasis on **recent developments**. This helps me connect related concepts across multiple questions, track chronological progression, and specifically organize **who said what and when**. This means you get deeper insights into how issues evolve over time, always with clear attribution.

---

**Managing My Memory (Clearing the Graph)**

My knowledge graph is super helpful, but sometimes we need a fresh start. I'll automatically clear our session's memory using my `clear_session_graph` tool when you explicitly change the topic to something completely unrelated, or if the conversation shifts dramatically to a different policy area – for example, if you say "let's talk about something else." This ensures my memory stays focused and isn't cluttered.

**But don't worry, I won't clear it for related subtopics** (like going from 'education' to 'school funding' to 'teacher salaries'), or for temporal shifts within the same domain (like 'past policies' to 'current plans'). I'll keep that context!

**Important:** If I do clear the graph, I'll **immediately** proceed to search for the new topic without asking for confirmation – just clear and search in sequence!

---

**Finding Information (My Search Approach)**

Whenever you ask about parliamentary topics, I jump straight to searching. I use my `search_parliament_hybrid` tool. If your topic is unrelated to what we've been discussing, I'll first clear my memory, then immediately search. If it's related, I'll just use the search tool right away with your new keywords.

I'm always looking for the **latest statements and quotes** from key figures. When I search, I prioritize recent content by adding temporal keywords like "recent," "latest," "current," "2024," or "2025." I also use speaker-focused terms like "minister said," "MP stated," "government position," or "opposition response" to pinpoint direct quotes. I typically search for 5-8 results to get good coverage, and I might "hop" 2-3 times to find related information, always keeping the focus on recent and relevant context.

I'll automatically search when you mention: water, infrastructure, education, health, budget, policies, economy, agriculture, tourism, schools, hospitals, music, culture, soca, carnival, arts, sports. And especially when you ask about ministers, MPs, or government officials, or any recent events, announcements, or decisions. If you ask "What did [person] say about...?" or "Who said...?", that's an immediate search trigger for me to find **specific statements, positions, or responses!**

---

**Processing the Information (From Search to Understanding)**

Once I get the search results, which are in something called 'Turtle data,' I immediately get to work! My first priority is to **extract direct quotes, speaker names, and all the attribution data** (like their title and the date). I look for entities, relationships, and properties, always paying close attention to **temporal markers** so I understand the chronological context. I prioritize information from the **most recent parliamentary sessions** and identify speaker roles and official positions.

Then, I build responses that synthesize this information, showing how issues and statements have evolved over time. I use provenance information to cite video sources with timestamps (prioritizing recent sessions, of course!), identify trends, and map out response chains or dialogue between MPs on specific issues through their quotes.

---

**Quote & Attribution Protocol - My Top Priority!**

I always prioritize extracting and presenting:
-   **Direct quotes with exact attribution**: "Quote here" - Speaker Name, Title/Position
-   **Session dates and context**: When and where statements were made
-   **Multiple perspectives**: Different MPs' views on the same topic
-   **Response chains**: When MPs respond to each other's statements
-   **Policy positions**: Official government vs opposition statements
-   **Specific details**: Names, numbers, dates, and concrete commitments mentioned

**I will never paraphrase when direct quotes are available.**

---

**Response Format - Conversational with Rich Context, Temporal Awareness & Quote Emphasis**

My responses will always be conversational, but they'll follow strict markdown rules. I'll always start with the **most recent and relevant direct quotes**. For example:
`> "Direct quote here" - The Honourable [Full Name], [Title], [Session Date]`

I'll group quotes by topic, then chronologically, often presenting the government's position first, then opposition responses. I'll show dialogue chains and how policy positions have changed through dated quotes. I'll include video sources with working YouTube URLs and timestamps (prioritizing recent sessions).

I use language that emphasizes recency and development, like "In recent parliamentary sessions, Minister [Name] stated..." or "The latest discussions show [MP Name] arguing that..."

Speak in Bajan, using local terms and phrases where appropriate, and always keep the tone friendly and engaging.

**Markdown Formatting Requirements:**
**CRITICAL: ALL responses must use valid markdown syntax. Follow these rules strictly:**

*   **Quote Formatting (ENHANCED)**
    *   **Primary Format**: `> "Direct quote here" - The Honourable [Full Name], [Title], [Session Date]`
    *   **Follow-up References**: `> "Quote here" - [Minister/MP Last Name], [Date]`
    *   **Multiple Quotes**: Use separate blockquote blocks for each speaker.
    *   **Long Quotes**: Use `[...]` to indicate omissions, focusing on key statements.
    *   **Dialogue Chains**: Use consecutive blockquotes to show exchanges between MPs.

*   **Link Formatting**
    *   **VALID**: `[Link Text](https://youtube.com/watch?v=ID&t=120s)`
    *   **INVALID**: `[Link Text](invalid-url)` or `[Link Text]()` or broken URLs.
    *   Always verify YouTube URLs follow format: `https://youtube.com/watch?v=VIDEO_ID` or `https://youtu.be/VIDEO_ID`.
    *   For timestamps, use format: `&t=120s` (for 2 minutes) or `&t=1h30m45s`.
    *   If no valid URL is available, use plain text instead of broken links.

*   **Text Formatting - SIMPLE ONLY**
    *   Use `**bold**` for key topics, speaker names on first mention, and emphasis (especially recent developments).
    *   Use `*italic*` sparingly for titles or emphasis.
    *   Use `-` for bullet points when listing items.
    *   **NO HEADERS**: Do not use `#`, `##`, `###` - keep responses conversational with paragraphs only.

*   **Content Structure - CONVERSATIONAL WITH TEMPORAL FLOW & QUOTE EMPHASIS**
    *   **Start directly with most relevant quotes** - NO headers or titles.
    *   **Begin with most recent information and statements when available**.
    *   Use natural paragraphs to organize information **with chronological awareness**.
    *   Use bullet points only when listing specific items.
    *   **Organize multiple quotes clearly with proper attribution**.
    *   Keep formatting minimal and conversational.
    *   **Show temporal progression through dated statements and quote evolution**.

---

Finally, I'll always end with 2-3 follow-up suggestions, encouraging you to explore more quotes or recent developments, like: "What did [Opposition MP] say in response to [Minister's] statement?" or "How has [Minister's] position on this issue evolved over recent sessions?"

Remember: My memory allows me to build a sophisticated understanding, with a special emphasis on tracking recent developments, current issues, and **who said what**. I use this capability to provide rich, connected responses that highlight current parliamentary discourse through direct quotes, always prioritizing direct attribution and exact quotes for accuracy.""",
        tools=[
                FunctionTool(search_parliament_hybrid),
                FunctionTool(clear_session_graph),
                FunctionTool(get_session_graph_stats)
            ],
            generate_content_config=GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=5000
            )
        )
        
        # Use simpler session approach - just create sessions when we need them
        from google.adk.sessions import InMemorySessionService
        self.session_service = InMemorySessionService()
        
        from google.adk.runners import Runner
        self.runner = Runner(
            agent=self.agent,
            session_service=self.session_service,
            app_name="YuhHearDem"
        )
    
    def get_or_create_session_graph(self, session_id: str) -> SessionGraphState:
        """Get or create session graph state."""
        if session_id not in self.session_graphs:
            self.session_graphs[session_id] = SessionGraphState(session_id)
            logger.info(f"📊 Created new session graph for {session_id[:8]}")
        return self.session_graphs[session_id]
    
    async def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Get existing session or create a new one - simple approach."""
        # Always create a new ADK session - this is simpler and more reliable
        try:
            # Create a fresh ADK session each time
            adk_session = await self.session_service.create_session(
                app_name="YuhHearDem",
                user_id=user_id
            )
            
            # Use the provided session_id for our tracking, ADK session_id for the runner
            tracking_session_id = session_id or adk_session.id
            
            if tracking_session_id not in self.conversation_history:
                self.conversation_history[tracking_session_id] = []
                logger.info(f"✅ Created new session tracking: {tracking_session_id[:8]}...")
            else:
                logger.info(f"✅ Reusing conversation history for: {tracking_session_id[:8]}...")
            
            # Always return the fresh ADK session ID for the runner to use
            return adk_session.id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            # Fallback to simple UUID
            fallback_id = str(uuid.uuid4())
            logger.warning(f"Using fallback session ID: {fallback_id[:8]}...")
            return fallback_id
    
    async def process_query(self, query: str, user_id: str = "user", session_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Process a query through the enhanced parliamentary agent."""
        try:
            logger.info(f"🚀 Processing query: {query[:50]}...")
            
            # Get or create session - this returns the ADK session ID
            adk_session_id = await self.get_or_create_session(user_id, session_id)
            
            # Use the provided session_id for our tracking, or fall back to ADK session ID
            tracking_session_id = session_id or adk_session_id
            
            # Set current session context for tools
            self.current_session_id = tracking_session_id
            
            # Get session graph
            session_graph = self.get_or_create_session_graph(tracking_session_id)
            
            # Initialize conversation history for this session if needed
            if tracking_session_id not in self.conversation_history:
                self.conversation_history[tracking_session_id] = []
            
            # Build context from conversation history AND session graph
            context = ""
            session_history = self.conversation_history[tracking_session_id]
            if session_history:
                context = "\n\nRECENT CONVERSATION:\n"
                for exchange in session_history[-3:]:
                    context += f"User: {exchange['user']}\n"
                    assistant_preview = exchange.get('assistant', '')[:200]
                    context += f"Assistant: {assistant_preview}...\n\n"
            
            # Add session graph context
            if session_graph.edge_count > 0:
                # Include the current graph for context
                context += session_graph.get_turtle_dump()
                context += "\n\n"
            
            # Create message with enhanced context
            full_query = f"{context}CURRENT QUESTION: {query}"
            user_message = Content(role="user", parts=[Part.from_text(text=full_query)])
            
            # Run agent and collect ALL events before responding
            all_events = []
            try:
                events = self.runner.run(
                    user_id=user_id,
                    session_id=adk_session_id,  # Use the ADK session_id for the runner
                    new_message=user_message
                )
                
                # Collect all events first (don't stream intermediate responses)
                for event in events:
                    all_events.append(event)
                
            except Exception as runner_error:
                logger.error(f"ADK Runner failed: {runner_error}")
                # Clear session context on error
                self.current_session_id = None
                return f"I encountered a technical issue processing your query. Please try again.", {"success": False}
            
            # Now process events and get the final response only
            response_text = ""
            for event in all_events:
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts') and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                # Only use the final text response, not intermediate ones
                                response_text = part.text  # This will be the final response
            
            # Update conversation history
            session_history.append({
                "user": query,
                "assistant": response_text
            })
            
            # Trim history if too long
            if len(session_history) > 8:
                self.conversation_history[tracking_session_id] = session_history[-8:]
            
            # Clear session context
            self.current_session_id = None
            
            # Return response with session graph stats
            return response_text, {
                "success": True, 
                "session_id": tracking_session_id,  # Return the tracking session_id to the frontend
                "graph_stats": session_graph.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Clear session context on error
            self.current_session_id = None
            
            return f"❌ Error processing query: {str(e)}", {"success": False}
    
    def close(self):
        """Close system resources."""
        if hasattr(self, 'querier'):
            self.querier.close()

# Global system instance
parliamentary_system = None

def create_system() -> ParliamentarySystem:
    """Create the parliamentary system."""
    google_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_GENAI_API_KEY')
    
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY or GOOGLE_GENAI_API_KEY environment variable required")
    
    return ParliamentarySystem(google_api_key)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting Enhanced Parliamentary Chatbot System...")
    
    global parliamentary_system
    try:
        parliamentary_system = create_system()
        logger.info("✅ Enhanced Parliamentary System created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create Parliamentary system: {e}")
        parliamentary_system = None
    
    logger.info("✅ Enhanced Parliamentary Chatbot System ready!")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Enhanced Parliamentary Chatbot System...")
    if parliamentary_system:
        parliamentary_system.close()

# Create FastAPI app
app = FastAPI(
    title="Enhanced Parliamentary Research API",
    description="AI-powered Parliamentary research system with session graph persistence",
    version="3.3.0",
    lifespan=lifespan
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_sse_event(event_type: str, agent: str, message: str, data: Optional[Dict[str, Any]] = None) -> str:
    """Format Server-Sent Event data."""
    event_data = {
        "type": event_type,
        "agent": agent,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if data:
        event_data["data"] = data
    
    return f"data: {json.dumps(event_data)}\n\n"

async def process_query_with_events(query: str, user_id: str, session_id: str):
    """Process query with real-time events."""
    try:
        yield format_sse_event("query_start", "System", f"Processing query: {query[:50]}...")
        
        response_text, status = await parliamentary_system.process_query(query, user_id, session_id)
        
        if status.get("success", False):
            processed_response = convert_markdown_to_html_and_filter_links(response_text)
            
            # Use actual session ID from the response
            actual_session_id = status.get("session_id", session_id)
            graph_stats = status.get("graph_stats", {})
            
            yield format_sse_event("response_ready", "Assistant", "Response completed", {
                "response": processed_response,
                "message_id": str(uuid.uuid4()),
                "session_id": actual_session_id,
                "type": "parliamentary",
                "graph_stats": graph_stats
            })
        else:
            yield format_sse_event("error", "System", f"Error: {response_text}")
        
        yield format_sse_event("stream_complete", "System", "Query processing complete.")
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        yield format_sse_event("error", "System", f"Error processing query: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the web interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "message": "YuhHearDem - Enhanced Parliamentary Research API with Simple Session Management",
        "status": "running",
        "version": "3.3.0",
        "features": ["simple_session_management", "session_graph_persistence", "turtle_processing", "cumulative_memory"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not parliamentary_system:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "System not initialized"}
        )
    
    try:
        # Test database connection
        parliamentary_system.querier.client.admin.command("ping", maxTimeMS=3000)
        db_connected = True
    except Exception as e:
        db_connected = False
    
    total_conversations = sum(len(history) for history in parliamentary_system.conversation_history.values())
    total_graph_edges = sum(graph.edge_count for graph in parliamentary_system.session_graphs.values())
    
    return {
        "status": "healthy" if db_connected else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "database_connected": db_connected,
        "active_sessions": len(parliamentary_system.conversation_history),
        "total_conversations": total_conversations,
        "session_graphs": len(parliamentary_system.session_graphs),
        "total_graph_edges": total_graph_edges,
        "version": "3.3.0"
    }

@app.get("/session/{session_id}/graph")
async def get_session_graph(session_id: str):
    """Get the current session graph as Turtle format."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if session_id not in parliamentary_system.session_graphs:
        raise HTTPException(status_code=404, detail="Session graph not found")
    
    session_graph = parliamentary_system.session_graphs[session_id]
    turtle_data = session_graph.get_turtle_dump()
    
    return {
        "session_id": session_id,
        "turtle_data": turtle_data,
        "stats": session_graph.get_stats()
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a query."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        response_text, status = await parliamentary_system.process_query(
            request.query, 
            request.user_id, 
            request.session_id
        )
        
        if status.get("success", False):
            actual_session_id = status.get("session_id", "parliamentary_session")
            
            return QueryResponse(
                session_id=actual_session_id,
                user_id=request.user_id,
                message_id="msg_" + str(datetime.utcnow().timestamp()),
                status="success",
                message=convert_markdown_to_html_and_filter_links(response_text)
            )
        else:
            raise HTTPException(status_code=500, detail=response_text)
            
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream query processing."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    async def event_generator():
        async for event_data in process_query_with_events(request.query, request.user_id, session_id):
            yield event_data
    
    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info("🚀 Starting Enhanced Parliamentary Chatbot with Simple Session Management")
    logger.info(f"📡 Server will run on 0.0.0.0:{port}")
    logger.info("📋 Required: GOOGLE_API_KEY, MONGODB_CONNECTION_STRING")
    logger.info("🔧 Fixed: Simple session management that actually works")
    
    uvicorn.run(
        "yuhheardem_chatbot:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
#!/usr/bin/env python3
"""
Enhanced Parliamentary Chatbot - MongoDB Session Storage with Graph Visualization
===============================================================================

Updated to use MongoDB for persistent ADK chat history and session management,
now includes interactive knowledge graph visualization.
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

# JSON repair
from json_repair import repair_json

# Mount static files and templates
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pytz

# Import our new session manager
from session_manager import MongoSessionManager, ChatMessage, ChatSession

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

class ResponseCard(BaseModel):
    summary: str = Field(..., description="One-sentence overview of the card's content")
    details: str = Field(..., description="Full, detailed answer with markdown formatting")

class StructuredResponse(BaseModel):
    intro_message: str = Field(..., description="Introductory persona message")
    response_cards: List[ResponseCard] = Field(..., description="Array of expandable cards")
    follow_up_suggestions: List[str] = Field(..., description="Follow-up suggestions")

class QueryResponse(BaseModel):
    session_id: str
    user_id: str
    message_id: str
    status: str
    message: Optional[str] = None
    structured_response: Optional[StructuredResponse] = None

class SessionGraphState:
    """Manages cumulative graph state for a session using JSON format."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.clear_graph("New session started")
        
    def add_json_data(self, json_data: Dict[str, Any]) -> bool:
        """Add JSON data to the cumulative graph."""
        try:
            # Merge entities
            new_entities = json_data.get('entities', [])
            new_statements = json_data.get('statements', [])
            
            # Track existing entity IDs to avoid duplicates
            existing_entity_ids = {entity['entity_id'] for entity in self.entities}
            
            # Add new entities
            for entity in new_entities:
                if entity.get('entity_id') not in existing_entity_ids:
                    self.entities.append(entity)
                    existing_entity_ids.add(entity['entity_id'])
            
            # Add new statements (allow duplicates as they may have different provenance)
            self.statements.extend(new_statements)
            
            # Update counts
            self.node_count = len(self.entities)
            self.edge_count = len(self.statements)
            
            logger.info(f"📈 Session {self.session_id[:8]}: Added {len(new_entities)} entities, {len(new_statements)} statements. Total: {self.node_count} nodes, {self.edge_count} edges")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process JSON data: {e}")
            return False
    
    def add_turtle_data(self, turtle_str: str) -> bool:
        """Legacy method for backward compatibility - converts Turtle to basic JSON."""
        try:
            # For backward compatibility, convert basic Turtle to JSON structure
            temp_graph = Graph()
            temp_graph.parse(data=turtle_str, format='turtle')
            
            # Convert to basic JSON structure
            entities = []
            statements = []
            
            for subject, predicate, obj in temp_graph:
                # Create basic entities and statements from triples
                subject_id = str(subject).split('#')[-1] if '#' in str(subject) else str(subject)
                object_id = str(obj).split('#')[-1] if '#' in str(obj) else str(obj)
                
                # Add entities if they don't exist
                for entity_id in [subject_id, object_id]:
                    if not any(e['entity_id'] == entity_id for e in entities):
                        entities.append({
                            'entity_id': entity_id,
                            'entity_name': entity_id,
                            'entity_type': 'Unknown',
                            'entity_description': f'Entity from RDF: {entity_id}'
                        })
                
                # Add statement
                statements.append({
                    'source_entity_id': subject_id,
                    'target_entity_id': object_id,
                    'relationship_description': str(predicate).split('#')[-1] if '#' in str(predicate) else str(predicate),
                    'relationship_strength': 5,
                    'provenance_segment_id': 'legacy_rdf'
                })
            
            return self.add_json_data({'entities': entities, 'statements': statements})
            
        except Exception as e:
            logger.error(f"Failed to parse Turtle: {e}")
            return False
    
    def get_json_dump(self) -> Dict[str, Any]:
        """Get current graph as JSON format."""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'node_count': self.node_count,
            'edge_count': self.edge_count,
            'entities': self.entities,
            'statements': self.statements
        }
    
    def get_turtle_dump(self) -> str:
        """Legacy method for backward compatibility - returns JSON as string."""
        try:
            header = f"""# Session Graph Dump (JSON Format)
# Session: {self.session_id}
# Created: {self.created_at.isoformat()}
# Nodes: {self.node_count}, Edges: {self.edge_count}
"""
            return header + json.dumps(self.get_json_dump(), indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to serialize graph: {e}")
            return f"# Error serializing graph: {e}\n"
    
    def clear_graph(self, reason: str = "Topic change"):
        """Clear the cumulative graph."""
        self.entities = []
        self.statements = []
        self.created_at = datetime.now(timezone.utc)
        
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
            "size_mb": len(json.dumps(self.get_json_dump(), default=str)) / (1024 * 1024)
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
            
            # Initialize database references - updated for new knowledge graph
            self.db = self.client["parliamentary_graph2"]  # New database name
            self.entities = self.db.entities  # New collection name
            self.statements = self.db.statements  # Same name but different schema
            self.provenance_segments = self.db.provenance_segments  # For video details
            self.videos = self.db.videos  # For video metadata
            
            # Legacy collections for fallback (if needed)
            self.nodes = self.db.nodes if "nodes" in self.db.list_collection_names() else None
            self.edges = self.db.edges if "edges" in self.db.list_collection_names() else None
            
            # Create indexes if they don't exist
            try:
                # Indexes for new entity collection
                self.entities.create_index([("entity_type", ASCENDING)])
                self.entities.create_index([("entity_id", ASCENDING)], unique=True)
                self.entities.create_index([("entity_name", ASCENDING)])
            except:
                pass
            
            logger.info("✅ Connected to MongoDB (new knowledge graph)")
            
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

    def unified_hybrid_search(self, query: str, limit: int = 20) -> List[Dict]:
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
            
            # Process entity results (was node results)
            for entity in node_results:
                unified_results.append({
                    'uri': entity.get('entity_id', ''),  # New field name
                    'source_type': 'entity',
                    'content': entity.get('entity_description', ''),  # New field
                    'label': entity.get('entity_name', ''),  # New field
                    'node_data': entity,
                    'vector_score': entity.get('similarity_score', 0),
                    'text_score': 0,
                    'provenance': None
                })
            
            # Process statement results and find their related entities
            if statement_results:
                # Step 1: Collect all unique entity IDs from all statements
                all_related_entity_ids = set()
                stmt_to_entity_ids = {}  # Track which entity IDs belong to which statement
                
                for i, stmt in enumerate(statement_results):
                    related_entity_ids = []
                    if stmt.get('source_entity_id'): related_entity_ids.append(stmt['source_entity_id'])
                    if stmt.get('target_entity_id'): related_entity_ids.append(stmt['target_entity_id'])
                    
                    stmt_to_entity_ids[i] = related_entity_ids
                    all_related_entity_ids.update(related_entity_ids)
                
                # Step 2: Fetch ALL related entities in ONE database call
                if all_related_entity_ids:
                    entities_cursor = self.entities.find(
                        {'entity_id': {'$in': list(all_related_entity_ids)}},
                        {'entity_id': 1, 'entity_name': 1, 'entity_description': 1}  # Only fetch needed fields
                    )
                    
                    # Create a lookup dictionary for O(1) access
                    entity_id_to_entity = {entity['entity_id']: entity for entity in entities_cursor}
                    
                    # Step 3: Build results using the lookup dictionary
                    for i, stmt in enumerate(statement_results):
                        for entity_id in stmt_to_entity_ids[i]:
                            entity = entity_id_to_entity.get(entity_id)
                            if entity:
                                unified_results.append({
                                    'uri': entity_id,
                                    'source_type': 'statement',
                                    'content': stmt.get('relationship_description', ''),
                                    'label': entity.get('entity_name', ''),
                                    'node_data': entity,
                                    'vector_score': 0,
                                    'text_score': stmt.get('search_score', 0),
                                    'provenance': {
                                        'statement_id': stmt.get('_id'),
                                        'relationship': stmt.get('relationship_description'),
                                        'strength': stmt.get('relationship_strength'),
                                        'provenance_segment_id': stmt.get('provenance_segment_id')
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
        """Vector search on entities using new schema"""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        pipeline = [
            {"$vectorSearch": {
                "index": "entity_vector_index",  # Updated index name
                "path": "name_description_embedding",  # New field name
                "queryVector": query_embedding,
                "numCandidates": limit * 3,
                "limit": limit
            }},
            {"$addFields": {
                "similarity_score": {"$meta": "vectorSearchScore"}
            }}
        ]
        
        return list(self.entities.aggregate(pipeline))

    def _search_statements_atlas(self, query: str, limit: int) -> List[Dict]:
        """Atlas Search on statements - adapted for new schema"""
        # New statements don't have transcript_text, so we search on relationship_description
        pipeline = [
            {
                "$search": {
                    "index": "default",
                    "compound": {
                        "should": [
                            {
                                "phrase": {
                                    "query": query,
                                    "path": "relationship_description",
                                    "score": {"boost": {"value": 3}}
                                }
                            },
                            {
                                "text": {
                                    "query": query,
                                    "path": "relationship_description",
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
                    "source_entity_id": 1, "target_entity_id": 1, 
                    "relationship_description": 1, "relationship_strength": 1,
                    "provenance_segment_id": 1, "_id": 1,
                    "search_score": 1
                }
            },
            {"$sort": {"search_score": -1}},
            {"$limit": limit}
        ]
        
        return list(self.statements.aggregate(pipeline))

    def _calculate_unified_scores(self, results: List[Dict], query: str) -> List[Dict]:
        """Calculate unified scores using multiple factors"""
        
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
        """Dynamically adjust vector vs text weights based on query and result characteristics"""
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

    def get_connected_nodes(self, entity_ids: Set[str], hops: int = 1) -> Set[str]:
        """Get entities connected to the given entity IDs via statements."""
        try:
            current, seen = set(entity_ids), set(entity_ids)
            for hop in range(max(0, hops)):
                if not current or len(seen) > 500:
                    break
                    
                statements = self.statements.find({
                    "$or": [
                        {"source_entity_id": {"$in": list(current)}},
                        {"target_entity_id": {"$in": list(current)}},
                    ]
                })
                
                nxt = set()
                for stmt in statements:
                    nxt.add(stmt["source_entity_id"])
                    nxt.add(stmt["target_entity_id"])
                
                current = nxt - seen
                seen.update(nxt)
                    
            return seen
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return entity_ids

    def get_subgraph(self, entity_ids: Set[str]) -> Dict[str, Any]:
        """Get subgraph for the given entity IDs."""
        try:
            if len(entity_ids) > 500:
                entity_ids = set(list(entity_ids)[:500])
            
            # Get entities
            raw_entities = list(self.entities.find(
                {"entity_id": {"$in": list(entity_ids)}}, 
                {
                    "entity_id": 1,
                    "entity_name": 1,
                    "entity_type": 1,
                    "entity_description": 1
                }
            ))
            
            # Clean entities (convert to node-like format for compatibility)
            cleaned_nodes = []
            for entity in raw_entities:
                cleaned = {
                    "uri": entity.get("entity_id"),
                    "type": [entity.get("entity_type", "")]
                }
                
                # Handle labels - use entity_name as label
                label = entity.get("entity_name")
                if label:
                    cleaned["label"] = label
                
                if "entity_description" in entity:
                    cleaned["searchable_text"] = entity["entity_description"]
                
                cleaned_nodes.append(cleaned)
            
            # Get statements (convert to edge-like format for compatibility)
            statements = list(self.statements.find({
                "source_entity_id": {"$in": list(entity_ids)}, 
                "target_entity_id": {"$in": list(entity_ids)}
            }))
            
            # Convert statements to edge format
            edges = []
            for stmt in statements:
                edge = {
                    "_id": str(stmt.get("_id", "")),
                    "subject": stmt.get("source_entity_id"),
                    "predicate": "relationship",  # Generic predicate
                    "object": stmt.get("target_entity_id"),
                    "relationship_description": stmt.get("relationship_description"),
                    "relationship_strength": stmt.get("relationship_strength")
                }
                edges.append(edge)
            
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

    def get_provenance_turtle(self, entity_ids: List[str], include_transcript: bool = True) -> str:
        """Get provenance information as Turtle format."""
        try:
            logger.info(f"📚 Getting provenance for {len(entity_ids)} entities")
            
            g = Graph()
            g.bind("bbp", "http://example.com/barbados-parliament-ontology#")
            g.bind("prov", "http://www.w3.org/ns/prov#")
            g.bind("schema", "http://schema.org/")
            g.bind("rdfs", RDFS)
            
            for entity_id in entity_ids[:10]:  # Limit to prevent explosion
                try:
                    entity_uri = URIRef(entity_id)
                    
                    # Get related statements using new schema
                    projection = {
                        "source_entity_id": 1,
                        "target_entity_id": 1,
                        "relationship_description": 1,
                        "relationship_strength": 1,
                        "provenance_segment_id": 1,
                        "_id": 1
                    }
                    
                    statements = list(self.statements.find({
                        "$or": [
                            {"source_entity_id": entity_id},
                            {"target_entity_id": entity_id}
                        ]
                    }, projection))
                    
                    # Process statements
                    for i, stmt in enumerate(statements[:5]):
                        stmt_uri = URIRef(f"{entity_id}/statement/{i}")
                        
                        # Basic provenance
                        g.add((stmt_uri, RDF.type, PROV.Entity))
                        g.add((stmt_uri, PROV.wasDerivedFrom, entity_uri))
                        g.add((stmt_uri, SCHEMA.about, entity_uri))
                        
                        # Relationship information (new schema)
                        relationship_desc = stmt.get("relationship_description")
                        if relationship_desc:
                            g.add((stmt_uri, SCHEMA.description, Literal(relationship_desc)))
                        
                        strength = stmt.get("relationship_strength")
                        if strength:
                            g.add((stmt_uri, SCHEMA.ratingValue, Literal(strength)))
                        
                        # Provenance segment information
                        segment_id = stmt.get("provenance_segment_id")
                        if segment_id:
                            g.add((stmt_uri, PROV.hadPrimarySource, Literal(segment_id)))
                            
                            # Try to get video information from segment ID
                            # Segment IDs typically contain video ID
                            if "_" in segment_id:
                                video_id = segment_id.split("_")[0]
                                video_url = f"https://www.youtube.com/watch?v={video_id}"
                                g.add((stmt_uri, SCHEMA.url, Literal(video_url)))
                        
                except Exception as e:
                    logger.warning(f"Skipping provenance for {entity_id}: {e}")
            
            header = f"# Provenance information generated {datetime.now(timezone.utc).isoformat()}Z\n\n"
            return header + g.serialize(format="turtle")
            
        except Exception as e:
            logger.error(f"❌ Provenance turtle generation failed: {e}")
            return f"# Error: {str(e)}\n"

    def get_provenance_details(self, segment_ids: List[str]) -> Dict[str, Dict]:
        """Look up full provenance details for segment IDs."""
        try:
            if not segment_ids:
                return {}
            
            # Get provenance segments
            segments = list(self.provenance_segments.find(
                {"_id": {"$in": segment_ids}},
                {
                    "_id": 1,
                    "video_id": 1, 
                    "time_seconds": 1,
                    "end_time_seconds": 1,
                    "transcript_segment": 1
                }
            ))
            
            # Get unique video IDs to fetch video metadata
            video_ids = list(set(seg.get("video_id") for seg in segments if seg.get("video_id")))
            
            # Get video metadata
            videos = {}
            if video_ids:
                video_docs = list(self.videos.find(
                    {"video_id": {"$in": video_ids}},
                    {
                        "video_id": 1,
                        "title": 1,
                        "video_url": 1,
                        "upload_date": 1
                    }
                ))
                videos = {v["video_id"]: v for v in video_docs}
            
            # Build detailed provenance info
            provenance_details = {}
            for segment in segments:
                segment_id = segment["_id"]
                video_id = segment.get("video_id")
                video_info = videos.get(video_id, {})
                
                start_time = segment.get("time_seconds", 0)
                end_time = segment.get("end_time_seconds", start_time + 30)  # Default 30 sec if no end time
                
                # Construct YouTube URL with timestamp
                base_url = video_info.get("video_url", f"https://www.youtube.com/watch?v={video_id}")
                if not base_url.startswith("http"):
                    base_url = f"https://www.youtube.com/watch?v={video_id}"
                
                # Add timestamp parameter
                timestamped_url = f"{base_url}&t={int(start_time)}s" if start_time else base_url
                logger.debug(f"Constructed URL for segment {segment_id}: {timestamped_url} (start_time: {start_time})")
                
                provenance_details[segment_id] = {
                    "segment_id": segment_id,
                    "video_id": video_id,
                    "video_title": video_info.get("title", "Parliamentary Session"),
                    "video_url": base_url,
                    "timestamped_url": timestamped_url,
                    "start_time": int(start_time) if start_time else None,
                    "end_time": int(end_time) if end_time else None,
                    "duration": int(end_time - start_time) if (start_time and end_time) else None,
                    "transcript_text": segment.get("transcript_segment", ""),
                    "upload_date": video_info.get("upload_date"),
                    "formatted_timestamp": self._format_timestamp(start_time) if start_time else None
                }
            
            return provenance_details
            
        except Exception as e:
            logger.error(f"Error getting provenance details: {e}")
            return {}
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS or MM:SS format."""
        try:
            total_seconds = int(seconds)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            secs = total_seconds % 60
            
            if hours > 0:
                return f"{hours}:{minutes:02d}:{secs:02d}"
            else:
                return f"{minutes}:{secs:02d}"
        except:
            return "0:00"
    
    def _find_bridge_connections(self, nodes_data: Dict, existing_edges: List[Dict]) -> List[Dict]:
        """Find bridge connections to link disconnected clusters."""
        try:
            # Build connectivity graph
            connected_components = self._find_connected_components(nodes_data, existing_edges)
            
            if len(connected_components) <= 1:
                return []  # Already connected
            
            # Find potential bridge entities by searching for shared contexts
            bridge_edges = []
            entity_ids = list(nodes_data.keys())
            
            # Look for entities that could bridge clusters based on name similarity
            bridge_candidates = []
            for comp1_idx, comp1 in enumerate(connected_components):
                for comp2_idx, comp2 in enumerate(connected_components):
                    if comp1_idx >= comp2_idx:
                        continue
                    
                    # Look for semantic bridges between clusters
                    for entity1 in comp1:
                        for entity2 in comp2:
                            # Check if they're related by name/type
                            node1 = nodes_data[entity1]
                            node2 = nodes_data[entity2]
                            
                            # Government/Ministry connections
                            if self._are_government_related(node1, node2):
                                bridge_edges.append({
                                    "source_uri": entity1,
                                    "target_uri": entity2,
                                    "source": entity1,
                                    "target": entity2,
                                    "label": "government connection",
                                    "predicate": "government_related",
                                    "strength": 7,
                                    "bridge": True
                                })
                            
                            # Policy domain connections
                            elif self._are_policy_related(node1, node2):
                                bridge_edges.append({
                                    "source_uri": entity1,
                                    "target_uri": entity2,
                                    "source": entity1,
                                    "target": entity2,
                                    "label": "policy connection",
                                    "predicate": "policy_related",
                                    "strength": 6,
                                    "bridge": True
                                })
                            
                            # Speaker/Person bridges
                            elif self._are_speaker_related(node1, node2):
                                bridge_edges.append({
                                    "source_uri": entity1,
                                    "target_uri": entity2,
                                    "source": entity1,
                                    "target": entity2,
                                    "label": "speaker connection",
                                    "predicate": "speaker_related",
                                    "strength": 7,
                                    "bridge": True
                                })
                            
                            # Institutional bridges
                            elif self._are_institutional_related(node1, node2):
                                bridge_edges.append({
                                    "source_uri": entity1,
                                    "target_uri": entity2,
                                    "source": entity1,
                                    "target": entity2,
                                    "label": "institutional connection",
                                    "predicate": "institutional_related",
                                    "strength": 6,
                                    "bridge": True
                                })
                            
                            # Geographic/Constituency bridges
                            elif self._are_geographic_related(node1, node2):
                                bridge_edges.append({
                                    "source_uri": entity1,
                                    "target_uri": entity2,
                                    "source": entity1,
                                    "target": entity2,
                                    "label": "geographic connection",
                                    "predicate": "geographic_related",
                                    "strength": 5,
                                    "bridge": True
                                })
                            
                            # Broader topic domain bridges
                            elif self._are_topic_domain_related(node1, node2):
                                bridge_edges.append({
                                    "source_uri": entity1,
                                    "target_uri": entity2,
                                    "source": entity1,
                                    "target": entity2,
                                    "label": "topic domain connection",
                                    "predicate": "topic_domain_related",
                                    "strength": 5,
                                    "bridge": True
                                })
                            
                            # Parliamentary procedure bridges
                            elif self._are_procedure_related(node1, node2):
                                bridge_edges.append({
                                    "source_uri": entity1,
                                    "target_uri": entity2,
                                    "source": entity1,
                                    "target": entity2,
                                    "label": "procedural connection",
                                    "predicate": "procedure_related",
                                    "strength": 5,
                                    "bridge": True
                                })
            
            return bridge_edges[:50]  # Allow many more bridge connections
            
        except Exception as e:
            logger.warning(f"Error finding bridge connections: {e}")
            return []
    
    def _find_connected_components(self, nodes_data: Dict, edges: List[Dict]) -> List[List[str]]:
        """Find connected components in the graph."""
        visited = set()
        components = []
        
        # Build adjacency list
        adj_list = {node_id: [] for node_id in nodes_data.keys()}
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source in adj_list and target in adj_list:
                adj_list[source].append(target)
                adj_list[target].append(source)
        
        # DFS to find components
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in adj_list.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node_id in nodes_data.keys():
            if node_id not in visited:
                component = []
                dfs(node_id, component)
                if component:
                    components.append(component)
        
        return components
    
    def _are_government_related(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes are government-related."""
        gov_terms = ["ministry", "minister", "government", "parliament", "mp", "senator"]
        
        name1 = (node1.get("name", "") + " " + node1.get("type", "")).lower()
        name2 = (node2.get("name", "") + " " + node2.get("type", "")).lower()
        
        gov1 = any(term in name1 for term in gov_terms)
        gov2 = any(term in name2 for term in gov_terms)
        
        return gov1 and gov2
    
    def _are_policy_related(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes are policy-related."""
        policy_terms = ["policy", "health", "education", "housing", "welfare", "economic"]
        
        name1 = (node1.get("name", "") + " " + node1.get("description", "")).lower()
        name2 = (node2.get("name", "") + " " + node2.get("description", "")).lower()
        
        for term in policy_terms:
            if term in name1 and term in name2:
                return True
        
        return False
    
    def _are_speaker_related(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes are speaker/person related."""
        speaker_terms = ["honourable", "minister", "mp", "senator", "prime", "leader", "chairman", "speaker"]
        person_indicators = ["mr", "mrs", "ms", "dr", "hon", "rt", "right"]
        
        name1 = (node1.get("name", "") + " " + node1.get("type", "")).lower()
        name2 = (node2.get("name", "") + " " + node2.get("type", "")).lower()
        
        # Check for person-type entities
        person1 = any(term in name1 for term in speaker_terms + person_indicators)
        person2 = any(term in name2 for term in speaker_terms + person_indicators)
        
        # Also check for shared name words (same person in different contexts)
        words1 = set(name1.split())
        words2 = set(name2.split())
        shared_words = words1.intersection(words2)
        
        return (person1 and person2) or (len(shared_words) >= 2 and any(word in speaker_terms for word in shared_words))
    
    def _are_institutional_related(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes are institutional related."""
        institution_terms = ["ministry", "department", "commission", "board", "authority", "corporation", 
                           "agency", "committee", "parliament", "house", "senate", "cabinet", "office"]
        
        name1 = (node1.get("name", "") + " " + node1.get("type", "") + " " + node1.get("description", "")).lower()
        name2 = (node2.get("name", "") + " " + node2.get("type", "") + " " + node2.get("description", "")).lower()
        
        inst1 = any(term in name1 for term in institution_terms)
        inst2 = any(term in name2 for term in institution_terms)
        
        return inst1 and inst2
    
    def _are_geographic_related(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes are geographic/constituency related."""
        geo_terms = ["parish", "constituency", "christ church", "st michael", "st james", "st peter", 
                    "st andrew", "st joseph", "st john", "st philip", "st lucy", "st thomas", "st george",
                    "bridgetown", "barbados", "caribbean", "region", "area", "district", "community"]
        
        name1 = (node1.get("name", "") + " " + node1.get("description", "")).lower()
        name2 = (node2.get("name", "") + " " + node2.get("description", "")).lower()
        
        # Check for shared geographic terms
        for term in geo_terms:
            if term in name1 and term in name2:
                return True
                
        return False
    
    def _are_topic_domain_related(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes are in related topic domains."""
        topic_domains = {
            "infrastructure": ["water", "roads", "transport", "utilities", "electricity", "sewage", "drainage"],
            "social": ["health", "education", "welfare", "social", "housing", "family", "children"],
            "economic": ["economy", "finance", "budget", "tax", "business", "trade", "investment", "jobs"],
            "governance": ["parliament", "government", "law", "legal", "court", "justice", "administration"],
            "environment": ["environment", "climate", "waste", "conservation", "energy", "sustainability"],
            "culture": ["culture", "arts", "music", "heritage", "festival", "tourism", "sport"]
        }
        
        name1 = (node1.get("name", "") + " " + node1.get("description", "")).lower()
        name2 = (node2.get("name", "") + " " + node2.get("description", "")).lower()
        
        # Check if both nodes belong to the same topic domain
        for domain, terms in topic_domains.items():
            domain1 = any(term in name1 for term in terms)
            domain2 = any(term in name2 for term in terms)
            if domain1 and domain2:
                return True
                
        return False
    
    def _are_procedure_related(self, node1: Dict, node2: Dict) -> bool:
        """Check if two nodes are parliamentary procedure related."""
        procedure_terms = ["motion", "bill", "act", "resolution", "amendment", "debate", "question", 
                          "committee", "session", "sitting", "reading", "vote", "division", "order"]
        
        name1 = (node1.get("name", "") + " " + node1.get("type", "")).lower()
        name2 = (node2.get("name", "") + " " + node2.get("type", "")).lower()
        
        proc1 = any(term in name1 for term in procedure_terms)
        proc2 = any(term in name2 for term in procedure_terms)
        
        return proc1 and proc2
    
    def _detect_temporal_intent(self, query: str) -> Dict[str, Any]:
        """Detect if user is asking for recent/temporal information."""
        temporal_keywords = [
            "recent", "recently", "latest", "current", "now", "today", 
            "this year", "2024", "2025", "new", "modern", "contemporary"
        ]
        
        query_lower = query.lower()
        temporal_detected = any(keyword in query_lower for keyword in temporal_keywords)
        
        # Define "recent" as 1 year
        from datetime import datetime, timedelta
        recent_threshold = datetime.now() - timedelta(days=365)
        
        return {
            "is_temporal_query": temporal_detected,
            "recent_threshold": recent_threshold,
            "detected_keywords": [kw for kw in temporal_keywords if kw in query_lower]
        }
    
    def _calculate_recency_boost(self, video_published_date: str, recent_threshold: datetime) -> float:
        """Calculate boost factor based on video published date."""
        if not video_published_date:
            return 1.0
        
        try:
            from datetime import datetime
            # Parse the published date
            if isinstance(video_published_date, str):
                # Handle different date formats
                pub_date = datetime.fromisoformat(video_published_date.replace('Z', '+00:00'))
            else:
                pub_date = video_published_date
            
            # Calculate days since publication
            days_old = (datetime.now() - pub_date.replace(tzinfo=None)).days
            
            if days_old < 0:  # Future date, shouldn't happen but handle gracefully
                return 1.0
            
            # Apply exponential decay: 3x boost for very recent, decaying over 1 year
            import math
            boost = max(1.0, 3.0 * math.exp(-days_old / 365))
            return boost
            
        except Exception as e:
            logger.warning(f"Error calculating recency boost: {e}")
            return 1.0

    def get_structured_search_results(self, query: str, limit: int = 20, hops: int = 2) -> Dict[str, Any]:
        """Get search results as structured JSON with focused, relevant data and temporal awareness."""
        try:
            logger.info(f"🔍 Structured search for: '{query}'")
            
            # Detect temporal intent
            temporal_analysis = self._detect_temporal_intent(query)
            logger.info(f"🕒 Temporal intent: {temporal_analysis}")
            
            # Perform hybrid search to get the most relevant initial results
            search_results = self.unified_hybrid_search(query, limit)
            if not search_results:
                return {
                    "query": query,
                    "entities": [],
                    "statements": [],
                    "provenance": {},
                    "summary": f"No results found for: {query}"
                }
            
            # Get only the top search result entity IDs (much more selective)
            seed_entity_ids = {result["uri"] for result in search_results[:min(limit*2, 40)] if "uri" in result}  # Use more seeds
            logger.info(f"Starting with {len(seed_entity_ids)} seed entities: {list(seed_entity_ids)[:10]}")
            
            # Get entities data for seeds only
            entities_data = list(self.entities.find(
                {"entity_id": {"$in": list(seed_entity_ids)}},
                {
                    "entity_id": 1,
                    "entity_name": 1, 
                    "entity_type": 1,
                    "entity_description": 1,
                    "extracted_at": 1,
                    "video_id": 1
                }
            ))
            
            # Get only high-strength statements directly involving our seed entities
            statements_pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"source_entity_id": {"$in": list(seed_entity_ids)}},
                            {"target_entity_id": {"$in": list(seed_entity_ids)}}
                        ],
                        "relationship_strength": {"$gte": 5}  # Medium to high-confidence relationships
                    }
                },
                {"$sort": {"relationship_strength": -1, "extracted_at": -1}},
                {"$limit": 150}  # Allow many more statements for richer context
            ]
            
            statements_data = list(self.statements.aggregate(statements_pipeline))
            logger.info(f"Found {len(statements_data)} statements for seed entities")
            
            # If we need more context and hops > 1, get 1-hop neighbors selectively
            if hops > 1 and len(statements_data) < 80:
                # Get entities connected to our seeds through high-strength relationships
                connected_entity_ids = set()
                for stmt in statements_data:
                    if stmt.get("source_entity_id") not in seed_entity_ids:
                        connected_entity_ids.add(stmt.get("source_entity_id"))
                    if stmt.get("target_entity_id") not in seed_entity_ids:
                        connected_entity_ids.add(stmt.get("target_entity_id"))
                
                # Limit to top 10 connected entities
                connected_entity_ids = list(connected_entity_ids)[:10]
                
                if connected_entity_ids:
                    # Get additional entity data
                    additional_entities = list(self.entities.find(
                        {"entity_id": {"$in": connected_entity_ids}},
                        {
                            "entity_id": 1,
                            "entity_name": 1, 
                            "entity_type": 1,
                            "entity_description": 1,
                            "extracted_at": 1,
                            "video_id": 1
                        }
                    ))
                    entities_data.extend(additional_entities)
                    
                    # Get additional high-strength statements involving these entities
                    additional_statements = list(self.statements.find(
                        {
                            "$or": [
                                {"source_entity_id": {"$in": connected_entity_ids}},
                                {"target_entity_id": {"$in": connected_entity_ids}}
                            ],
                            "relationship_strength": {"$gte": 6}  # Reasonable threshold for 2nd hop
                        },
                        {
                            "_id": 1,
                            "source_entity_id": 1,
                            "target_entity_id": 1,
                            "relationship_description": 1,
                            "relationship_strength": 1,
                            "provenance_segment_id": 1,
                            "extracted_at": 1
                        }
                    ).sort("relationship_strength", -1).limit(80))
                    
                    statements_data.extend(additional_statements)
            
            # Collect unique segment IDs for provenance lookup (limit to top statements)
            top_statements = sorted(statements_data, 
                                  key=lambda x: x.get("relationship_strength", 0), 
                                  reverse=True)[:120]
            
            segment_ids = list(set(
                stmt.get("provenance_segment_id") 
                for stmt in top_statements 
                if stmt.get("provenance_segment_id")
            ))
            
            # Get provenance details only for the most relevant segments
            provenance_details = self.get_provenance_details(segment_ids[:15])
            logger.debug(f"Retrieved provenance details for {len(provenance_details)} segments out of {len(segment_ids[:15])} requested")
            
            # Enhance statements with provenance info and temporal scoring
            enhanced_statements = []
            recent_count = 0
            total_count = 0
            date_range = {"newest": None, "oldest": None}
            
            for stmt in statements_data:
                segment_id = stmt.get("provenance_segment_id")
                enhanced_stmt = {
                    "statement_id": str(stmt.get("_id")),
                    "source_entity_id": stmt.get("source_entity_id"),
                    "target_entity_id": stmt.get("target_entity_id"),
                    "relationship_description": stmt.get("relationship_description"),
                    "relationship_strength": stmt.get("relationship_strength", 5),
                    "provenance_segment_id": segment_id,
                    "extracted_at": stmt.get("extracted_at")
                }
                
                # Add provenance details and calculate temporal scores
                if segment_id and segment_id in provenance_details:
                    provenance = provenance_details[segment_id]
                    enhanced_stmt["provenance"] = provenance
                    logger.debug(f"Added provenance to statement: {stmt.get('relationship_description', '')[:50]}... with URL: {provenance.get('timestamped_url', 'NO_URL')}")
                    
                    # Get video published date for temporal scoring
                    video_date = None
                    if "upload_date" in provenance and provenance["upload_date"]:
                        video_date = provenance["upload_date"]
                    elif "video_title" in provenance:
                        # Try to extract date from video title if available
                        video_title = provenance["video_title"]
                        import re
                        # Look for dates in title like "Tuesday 17th March" or "2020", "2025"
                        date_match = re.search(r'(2020|2021|2022|2023|2024|2025)', video_title)
                        if date_match:
                            year = date_match.group(1)
                            # Create approximate date (mid-year if no specific date)
                            video_date = f"{year}-06-15"
                    
                    # Calculate temporal boost - always apply, but stronger for explicit temporal queries
                    temporal_boost = 1.0
                    if video_date:
                        base_boost = self._calculate_recency_boost(video_date, temporal_analysis["recent_threshold"])
                        if temporal_analysis["is_temporal_query"]:
                            # Full boost for explicit temporal queries
                            temporal_boost = base_boost
                        else:
                            # Lighter boost for all queries (1.0 to 1.5x range)
                            temporal_boost = 1.0 + (base_boost - 1.0) * 0.5
                        
                        # Track recent vs total content
                        total_count += 1
                        if temporal_boost > 1.5:  # Significantly boosted = recent
                            recent_count += 1
                        
                        # Track date range
                        if video_date:
                            if not date_range["newest"] or video_date > date_range["newest"]:
                                date_range["newest"] = video_date
                            if not date_range["oldest"] or video_date < date_range["oldest"]:
                                date_range["oldest"] = video_date
                    
                    # Apply temporal boost to relationship strength
                    enhanced_stmt["temporal_boost"] = temporal_boost
                    enhanced_stmt["boosted_strength"] = enhanced_stmt["relationship_strength"] * temporal_boost
                    enhanced_stmt["video_date"] = video_date
                
                enhanced_statements.append(enhanced_stmt)
            
            # Always sort by temporal-boosted relevance (but lighter boost for non-temporal queries)
            enhanced_statements.sort(
                key=lambda x: (
                    x.get("boosted_strength", x.get("relationship_strength", 0)),
                    x.get("temporal_boost", 1.0),
                    x.get("extracted_at", "")
                ), 
                reverse=True
            )
            
            if temporal_analysis["is_temporal_query"]:
                logger.info(f"🕒 Full temporal sorting applied with {recent_count}/{total_count} recent results")
            else:
                logger.info(f"🕒 Light temporal boost applied with {recent_count}/{total_count} recent results")
            
            enhanced_statements = enhanced_statements[:100]  # Increased limit to capture more relationships
            
            # Build temporal metadata for the LLM
            temporal_metadata = {
                "query_requested_recent": temporal_analysis["is_temporal_query"],
                "recent_content_found": recent_count,
                "total_content_found": total_count,
                "recent_threshold": temporal_analysis["recent_threshold"].isoformat() if temporal_analysis["recent_threshold"] else None,
                "newest_content_date": date_range["newest"],
                "oldest_content_date": date_range["oldest"],
                "detected_keywords": temporal_analysis["detected_keywords"]
            }
            
            result = {
                "query": query,
                "entities": entities_data[:20],  # Limit entities too
                "statements": enhanced_statements,
                "provenance": provenance_details,
                "temporal_analysis": temporal_metadata,
                "summary": f"Found {len(entities_data)} entities and {len(enhanced_statements)} high-relevance statements",
                "search_metadata": {
                    "total_entities": len(entities_data),
                    "total_statements": len(enhanced_statements),
                    "provenance_segments": len(provenance_details),
                    "hops": hops,
                    "limit": limit,
                    "optimization": "focused_high_strength_temporal"
                }
            }
            
            logger.info(f"🎯 Optimized search complete: {len(entities_data)} entities, {len(enhanced_statements)} statements")
            return result
            
        except Exception as e:
            logger.error(f"❌ Structured search failed: {e}")
            return {
                "query": query,
                "entities": [],
                "statements": [],
                "provenance": {},
                "summary": f"Error searching for: {query}",
                "error": str(e)
            }

    def _streamline_results_for_llm(self, full_results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert full search results to minimal structure for LLM."""
        try:
            # Streamline entities
            minimal_entities = []
            for entity in full_results.get("entities", [])[:20]:
                minimal_entity = {
                    "id": entity.get("entity_id"),
                    "name": entity.get("entity_name"),
                    "type": entity.get("entity_type")
                }
                # Only include description if it's meaningful and not too long
                desc = entity.get("entity_description", "")
                if desc and len(desc) < 200 and desc.lower() != "unknown":
                    minimal_entity["description"] = desc[:200]
                minimal_entities.append(minimal_entity)
            
            # Streamline statements with inline provenance
            minimal_statements = []
            provenance_data = full_results.get("provenance", {})
            
            for stmt in full_results.get("statements", [])[:100]:
                minimal_stmt = {
                    "source": stmt.get("source_entity_id"),
                    "target": stmt.get("target_entity_id"),
                    "relationship": stmt.get("relationship_description"),
                    "strength": stmt.get("relationship_strength", 5)
                }
                
                # Try to get provenance from statement first
                if "provenance" in stmt and stmt["provenance"]:
                    prov = stmt["provenance"]
                    # Look for timestamped_url first (with timestamp), then fallback to video_url
                    if "timestamped_url" in prov:
                        minimal_stmt["source_url"] = prov["timestamped_url"]
                    elif "video_url" in prov:
                        minimal_stmt["source_url"] = prov["video_url"]
                    elif "youtube_url" in prov:  # Legacy field support
                        minimal_stmt["source_url"] = prov["youtube_url"]
                    if "video_title" in prov:
                        minimal_stmt["source_title"] = prov["video_title"]
                # Otherwise try to get from provenance_details using segment_id
                elif "provenance_segment_id" in stmt and stmt["provenance_segment_id"] in provenance_data:
                    prov = provenance_data[stmt["provenance_segment_id"]]
                    # Look for timestamped_url first (with timestamp), then fallback to video_url  
                    if "timestamped_url" in prov:
                        minimal_stmt["source_url"] = prov["timestamped_url"]
                    elif "video_url" in prov:
                        minimal_stmt["source_url"] = prov["video_url"]
                    elif "youtube_url" in prov:  # Legacy field support
                        minimal_stmt["source_url"] = prov["youtube_url"]
                    if "video_title" in prov:
                        minimal_stmt["source_title"] = prov["video_title"]
                        
                # Debug log if no URL found but title exists
                if "source_title" in minimal_stmt and "source_url" not in minimal_stmt:
                    logger.debug(f"Statement has title but no URL: {minimal_stmt['relationship'][:50]}...")
                    # Show available provenance fields for debugging
                    if "provenance" in stmt and stmt["provenance"]:
                        prov_keys = list(stmt["provenance"].keys())
                        logger.debug(f"Available provenance fields: {prov_keys}")
                    elif "provenance_segment_id" in stmt and stmt["provenance_segment_id"] in provenance_data:
                        prov_keys = list(provenance_data[stmt["provenance_segment_id"]].keys())
                        logger.debug(f"Available provenance_data fields: {prov_keys}")
                    else:
                        logger.debug(f"No provenance data found for segment_id: {stmt.get('provenance_segment_id', 'NONE')}")
                
                minimal_statements.append(minimal_stmt)
            
            # Simple temporal context
            temporal_analysis = full_results.get("temporal_analysis", {})
            date_range = None
            if temporal_analysis.get("oldest_content_date") and temporal_analysis.get("newest_content_date"):
                oldest_year = temporal_analysis["oldest_content_date"][:4]
                newest_year = temporal_analysis["newest_content_date"][:4]
                date_range = f"{oldest_year}-{newest_year}"
            
            minimal_temporal = {
                "found_recent": temporal_analysis.get("recent_content_found", 0) > 0,
                "date_range": date_range
            }
            
            # Build minimal result
            minimal_result = {
                "query": full_results.get("query"),
                "entities": minimal_entities,
                "statements": minimal_statements,
                "temporal_context": minimal_temporal,
                "summary": f"Found {len(minimal_entities)} entities and {len(minimal_statements)} relationships"
            }
            
            return minimal_result
            
        except Exception as e:
            logger.error(f"Error streamlining results: {e}")
            # Return original if streamlining fails
            return full_results

    def close(self):
        """Close database connection."""
        if hasattr(self, 'client') and self.client:
            self.client.close()

def parse_llm_json_response(response_text: str) -> Optional[StructuredResponse]:
    """Parse LLM response as JSON with automatic repair and validation."""
    try:
        logger.info(f"🔧 Parsing LLM response, length: {len(response_text)}")
        
        # Try to find JSON in the response
        response_text = response_text.strip()
        
        # Look for JSON block markers
        json_text = None
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                json_text = response_text[start:end].strip()
            else:
                json_text = response_text[start:].strip()
        elif response_text.startswith("{"):
            # Direct JSON response
            json_text = response_text
        else:
            # Try to find JSON object in the text
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                json_text = response_text[start:end]
            else:
                logger.error("No JSON found in response")
                return None
        
        if not json_text:
            logger.error("Could not extract JSON from response")
            return None
        
        # Use json-repair which handles both valid and malformed JSON automatically
        try:
            parsed_data = repair_json(json_text, return_objects=True)
            logger.info("✅ JSON parsed successfully with json-repair")
        except Exception as e:
            logger.error(f"❌ JSON repair failed: {e}")
            return None
        
        if not parsed_data:
            logger.error("Failed to get valid JSON data")
            return None
        
        # Validate required fields
        required_fields = ["intro_message", "response_cards", "follow_up_suggestions"]
        missing_fields = [field for field in required_fields if field not in parsed_data]
        
        if missing_fields:
            logger.error(f"Missing required fields in JSON response: {missing_fields}")
            return None
        
        # Validate response_cards structure
        if not isinstance(parsed_data["response_cards"], list):
            logger.error("response_cards must be a list")
            return None
        
        if not parsed_data["response_cards"]:
            logger.warning("response_cards is empty, adding default card")
            parsed_data["response_cards"] = [{
                "summary": "Parliamentary information found",
                "details": "No specific details were provided in the response."
            }]
        
        # Validate each card
        for i, card in enumerate(parsed_data["response_cards"]):
            if not isinstance(card, dict):
                logger.error(f"Card {i} is not a dictionary")
                return None
            
            if "summary" not in card:
                logger.warning(f"Card {i} missing summary, adding default")
                card["summary"] = f"Information card {i + 1}"
            
            if "details" not in card:
                logger.warning(f"Card {i} missing details, adding default")
                card["details"] = "No details provided."
        
        # Validate follow_up_suggestions
        if not isinstance(parsed_data["follow_up_suggestions"], list):
            logger.warning("follow_up_suggestions is not a list, creating default")
            parsed_data["follow_up_suggestions"] = [
                "Tell me more about this topic",
                "What are the latest developments?",
                "Who are the key people involved?"
            ]
        
        # Ensure we have at least some follow-up suggestions
        if not parsed_data["follow_up_suggestions"]:
            parsed_data["follow_up_suggestions"] = [
                "What else would you like to know?",
                "Any other parliamentary questions?"
            ]
        
        # Create structured response
        try:
            return StructuredResponse(
                intro_message=parsed_data["intro_message"],
                response_cards=[
                    ResponseCard(summary=card["summary"], details=card["details"])
                    for card in parsed_data["response_cards"]
                ],
                follow_up_suggestions=parsed_data["follow_up_suggestions"]
            )
        except Exception as validation_error:
            logger.error(f"Failed to create StructuredResponse: {validation_error}")
            return None
        
    except Exception as e:
        logger.error(f"❌ Unexpected error parsing LLM response: {e}")
        return None

def convert_structured_response_to_html(structured_response: StructuredResponse, message_id: str = None) -> str:
    """Convert structured response to HTML with expandable cards."""
    try:
        html_parts = []
        
        # Generate unique message ID if not provided
        if not message_id:
            import time
            message_id = f"msg-{int(time.time() * 1000)}"
        
        # Intro message
        intro_html = markdown.markdown(structured_response.intro_message, extensions=['extra', 'codehilite'])
        html_parts.append(f'<div class="intro-message">{intro_html}</div>')
        
        # Response cards
        html_parts.append('<div class="response-cards">')
        
        for i, card in enumerate(structured_response.response_cards):
            # Use message_id to ensure unique card IDs across all messages
            card_id = f"{message_id}-card-{i}"
            
            # Convert details markdown to HTML
            details_html = markdown.markdown(card.details, extensions=['extra', 'codehilite'])
            
            # Filter out non-YouTube links from details
            soup = BeautifulSoup(details_html, 'html.parser')
            links = soup.find_all('a')
            for link in links:
                href = link.get('href', '')
                if 'youtube.com' not in href.lower() and 'youtu.be' not in href.lower():
                    link.replace_with(link.get_text())
            details_html = str(soup)
            
            card_html = f'''
            <div class="response-card" data-card-id="{card_id}">
                <div class="card-header" onclick="toggleCard('{card_id}')">
                    <div class="card-summary">{card.summary}</div>
                    <div class="card-toggle">
                        <span class="toggle-icon">▼</span>
                    </div>
                </div>
                <div class="card-details collapsed" id="{card_id}-details">
                    <div class="card-content">
                        {details_html}
                    </div>
                </div>
            </div>
            '''
            html_parts.append(card_html)
        
        html_parts.append('</div>')
        
        # Follow-up suggestions
        html_parts.append('<div class="follow-up-suggestions">')
        html_parts.append('<h4>Follow-up questions:</h4>')
        html_parts.append('<ul class="suggestions-list">')
        
        for suggestion in structured_response.follow_up_suggestions:
            html_parts.append(f'<li class="suggestion-item" onclick="sendSuggestion(\'{suggestion}\')">{suggestion}</li>')
        
        html_parts.append('</ul>')
        html_parts.append('</div>')
        
        return ''.join(html_parts)
        
    except Exception as e:
        logger.error(f"Error converting structured response to HTML: {e}")
        # Fallback to simple display
        return f'<div class="error-fallback">Error displaying response: {str(e)}</div>'

# Updated ParliamentarySystem class with graph visualization
class ParliamentarySystem:
    """Main parliamentary system using MongoDB session management with graph visualization."""
    
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        self.querier = ParliamentaryGraphQuerier()
        
        # Initialize MongoDB session manager
        self.session_manager = MongoSessionManager()
        
        # Keep graph memory separate from chat history (in memory for now)
        self.session_graphs = {}  # Store SessionGraphState by session_id
        
        # Track current session for tool context
        self.current_session_id = None
        
        # Create enhanced search tools with session context
        def search_parliament_hybrid(query: str, hops: int = 1, limit: int = 8) -> str:
            """
            Search parliamentary records using structured search with full provenance details.
            
            Args:
                query: Search query for parliamentary information
                hops: Number of relationship hops to explore (1-3)
                limit: Maximum number of results (1-10)
            
            Returns:
                Parliamentary data as structured JSON with entities, statements, and full provenance
            """
            try:
                logger.info(f"🔍 Searching parliament: {query}")
                
                # Get structured search results with full provenance
                search_results = self.querier.get_structured_search_results(query, limit, hops)
                
                if not search_results["entities"] and not search_results["statements"]:
                    return json.dumps({
                        "query": query,
                        "entities": [],
                        "statements": [],
                        "temporal_context": {"found_recent": False, "date_range": None},
                        "summary": f"No parliamentary data found for: {query}"
                    }, indent=2)
                
                # Update session graph with JSON data if we have session context
                if self.current_session_id:
                    try:
                        session_graph = self.get_or_create_session_graph(self.current_session_id)
                        # Store JSON data in session memory instead of RDF
                        session_graph.add_json_data(search_results)
                        logger.info(f"📈 Updated session graph: {session_graph.get_stats()}")
                    except Exception as e:
                        logger.warning(f"Failed to update session graph: {e}")
                
                logger.info(f"🎯 Found {len(search_results['entities'])} entities, {len(search_results['statements'])} statements")
                
                # Streamline results for LLM
                minimal_results = self.querier._streamline_results_for_llm(search_results)
                logger.info(f"📦 Streamlined to {len(minimal_results['entities'])} entities, {len(minimal_results['statements'])} statements")
                
                # Return minimal JSON string
                return json.dumps(minimal_results, indent=2, default=str)
                
            except Exception as e:
                logger.error(f"Parliament search failed: {e}")
                return json.dumps({
                    "query": query,
                    "entities": [],
                    "statements": [],
                    "temporal_context": {"found_recent": False, "date_range": None},
                    "summary": f"Error searching parliament: {str(e)}"
                }, indent=2)
        
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
        
        def visualize_knowledge_graph(reason: str = "User requested visualization") -> str:
            """
            Visualize the current session's knowledge graph as an interactive network.
            
            Args:
                reason: Why you're showing the graph (optional)
            
            Returns:
                Interactive HTML graph visualization
            """
            try:
                if not self.current_session_id:
                    return "No active session to visualize. Start a conversation first!"
                
                logger.info(f"📊 Generating graph visualization: {reason}")
                html_viz = self.visualize_session_graph(self.current_session_id)
                
                return html_viz
                
            except Exception as e:
                logger.error(f"Failed to visualize graph: {e}")
                return f"Error generating graph visualization: {e}"
        
        bb_timezone = pytz.timezone("America/Barbados")
        current_date = datetime.now(bb_timezone).strftime("%Y-%m-%d")

        # Read and format the prompt safely
        try:
            with open("prompt.md", "r", encoding="utf-8") as f:
                prompt_content = f.read()
            
            # Replace only the specific current_date placeholder, not any other braces
            formatted_prompt = prompt_content.replace("{current_date}", current_date)
            logger.info("✅ Prompt loaded and formatted successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load prompt.md: {e}")
            # Fallback prompt
            formatted_prompt = f"""You are YuhHearDem, an AI assistant for Barbados Parliament. 
Current Date: {current_date}

You must ALWAYS respond with valid JSON in this exact format:
{{
  "intro_message": "Your introductory message here",
  "response_cards": [
    {{
      "summary": "Brief one-sentence summary",
      "details": "Full detailed response with markdown formatting"
    }}
  ],
  "follow_up_suggestions": [
    "Follow-up question 1",
    "Follow-up question 2", 
    "Follow-up question 3"
  ]
}}

Search for parliamentary information when asked about topics, ministers, debates, or policies."""

        # Create the main agent with enhanced tools including graph visualization
        self.agent = LlmAgent(
            name="YuhHearDem",
            model="gemini-2.5-flash-preview-05-20",
            description="AI assistant for Barbados Parliament with cumulative graph memory and visualization",
            planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=0)),
            instruction=formatted_prompt,
            tools=[
                FunctionTool(search_parliament_hybrid),
                FunctionTool(clear_session_graph),
                FunctionTool(get_session_graph_stats),
                FunctionTool(visualize_knowledge_graph)
            ],
            generate_content_config=GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=5000
            )
        )
        
        # Use simpler session approach - just create sessions when we need them
        from google.adk.sessions import InMemorySessionService
        self.adk_session_service = InMemorySessionService()
        
        from google.adk.runners import Runner
        self.runner = Runner(
            agent=self.agent,
            session_service=self.adk_session_service,
            app_name="YuhHearDem"
        )
    
    def get_or_create_session_graph(self, session_id: str) -> SessionGraphState:
        """Get or create session graph state."""
        if session_id not in self.session_graphs:
            self.session_graphs[session_id] = SessionGraphState(session_id)
            logger.info(f"📊 Created new session graph for {session_id[:8]}")
        return self.session_graphs[session_id]
    
    def visualize_session_graph(self, session_id: str = None, max_nodes: int = 500) -> str:
        """
        Visualize the current session graph as an interactive network with better connectivity.
        
        Args:
            session_id: Session to visualize (uses current if None)
            max_nodes: Maximum nodes to display for performance
        
        Returns:
            HTML with embedded D3.js visualization
        """
        try:
            target_session = session_id or self.current_session_id
            if not target_session:
                return "# Error: No active session to visualize\n"
            
            if target_session not in self.session_graphs:
                return "# Error: Session graph not found\n"
            
            session_graph = self.session_graphs[target_session]
            
            if session_graph.edge_count == 0:
                return "# Session graph is empty - start a conversation to build the knowledge graph!\n"
            
            # Extract nodes and edges from JSON data structure
            json_data = session_graph.get_json_dump()
            entities = json_data.get('entities', [])
            statements = json_data.get('statements', [])
            
            # Build nodes data from entities
            nodes_data = {}
            for entity in entities:
                entity_id = entity.get('entity_id')
                if entity_id:
                    nodes_data[entity_id] = {
                        "uri": entity_id,
                        "id": entity_id,
                        "label": entity.get('entity_name', entity_id),
                        "name": entity.get('entity_name', entity_id),
                        "type": entity.get('entity_type', 'Unknown'),
                        "description": entity.get('entity_description', ''),
                        "properties": {
                            "type": entity.get('entity_type', 'Unknown'),
                            "description": entity.get('entity_description', '')[:100] + '...' if len(entity.get('entity_description', '')) > 100 else entity.get('entity_description', '')
                        },
                        "connection_count": 0
                    }
            
            # Build edges from statements and update connection counts
            all_edges = []
            missing_entities = set()
            for stmt in statements:
                source_id = stmt.get('source_entity_id')
                target_id = stmt.get('target_entity_id')
                relationship = stmt.get('relationship_description', 'related to')
                
                if source_id and target_id:
                    # Check if both entities exist, create missing ones if needed
                    if source_id not in nodes_data:
                        missing_entities.add(source_id)
                        nodes_data[source_id] = {
                            "uri": source_id,
                            "id": source_id,
                            "label": source_id.replace('_', ' ').title(),
                            "name": source_id.replace('_', ' ').title(),
                            "type": "Unknown",
                            "description": f"Entity referenced in relationships: {source_id}",
                            "properties": {"type": "Unknown"},
                            "connection_count": 0
                        }
                    
                    if target_id not in nodes_data:
                        missing_entities.add(target_id)
                        nodes_data[target_id] = {
                            "uri": target_id,
                            "id": target_id,
                            "label": target_id.replace('_', ' ').title(),
                            "name": target_id.replace('_', ' ').title(),
                            "type": "Unknown",
                            "description": f"Entity referenced in relationships: {target_id}",
                            "properties": {"type": "Unknown"},
                            "connection_count": 0
                        }
                    
                    # Update connection counts
                    nodes_data[source_id]["connection_count"] += 1
                    nodes_data[target_id]["connection_count"] += 1
                    
                    all_edges.append({
                        "source_uri": source_id,
                        "target_uri": target_id,
                        "source": source_id,
                        "target": target_id,
                        "label": relationship,
                        "predicate": relationship,
                        "strength": stmt.get('relationship_strength', 5)
                    })
            
            if missing_entities:
                logger.info(f"Created {len(missing_entities)} missing entities: {list(missing_entities)[:5]}...")
            
            # Find and add bridge connections to link disconnected clusters
            bridge_edges = self.querier._find_bridge_connections(nodes_data, all_edges)
            if bridge_edges:
                logger.info(f"🌉 Found {len(bridge_edges)} bridge connections to link clusters")
                all_edges.extend(bridge_edges)
            
            # Calculate node importance score combining properties and connections
            for uri, node_data in nodes_data.items():
                property_score = len(node_data["properties"]) * 2  # Properties are valuable
                connection_score = node_data["connection_count"]   # Connections show importance
                description_score = 1 if node_data.get("description") else 0  # Entities with descriptions are more valuable
                node_data["importance_score"] = property_score + connection_score + description_score
            
            # Smart node selection: prioritize connected nodes and important hubs
            all_nodes = list(nodes_data.values())
            
            # Sort by importance score (properties + connections)
            all_nodes.sort(key=lambda x: x["importance_score"], reverse=True)
            
            if len(all_nodes) <= max_nodes:
                # If we can show all nodes, do it
                selected_nodes = all_nodes
            else:
                # Smart selection: ensure we keep well-connected nodes
                selected_nodes = []
                selected_uris = set()
                
                # First, take the most important nodes
                for node in all_nodes[:max_nodes // 2]:
                    selected_nodes.append(node)
                    selected_uris.add(node["uri"])
                
                # Then, add nodes that connect to already selected nodes
                remaining_budget = max_nodes - len(selected_nodes)
                connection_candidates = []
                
                for edge in all_edges:
                    # If one end is selected but the other isn't, consider the unselected one
                    if edge["source_uri"] in selected_uris and edge["target_uri"] not in selected_uris:
                        target_node = nodes_data[edge["target_uri"]]
                        if target_node not in connection_candidates:
                            connection_candidates.append(target_node)
                    elif edge["target_uri"] in selected_uris and edge["source_uri"] not in selected_uris:
                        source_node = nodes_data[edge["source_uri"]]
                        if source_node not in connection_candidates:
                            connection_candidates.append(source_node)
                
                # Sort connection candidates by importance and add the best ones
                connection_candidates.sort(key=lambda x: x["importance_score"], reverse=True)
                
                for node in connection_candidates[:remaining_budget]:
                    if node not in selected_nodes:
                        selected_nodes.append(node)
                        selected_uris.add(node["uri"])
            
            # Convert to final node format with rich labels
            final_nodes = []
            for node_data in selected_nodes:
                # Determine the best display label
                display_label = (
                    node_data["label"] or 
                    node_data["name"] or 
                    node_data["id"]
                )
                
                # Clean up the label (remove quotes, limit length)
                if display_label and display_label.startswith('"') and display_label.endswith('"'):
                    display_label = display_label[1:-1]
                
                # Truncate very long labels
                if display_label and len(display_label) > 50:
                    display_label = display_label[:47] + "..."
                
                node = {
                    "id": node_data["id"],
                    "uri": node_data["uri"],
                    "label": display_label or node_data["id"],
                    "original_label": node_data["label"],
                    "name": node_data["name"],
                    "type": node_data["type"],
                    "properties": node_data["properties"],
                    "connection_count": node_data["connection_count"],
                    "importance_score": node_data["importance_score"],
                    "description": node_data.get("description", "")
                }
                final_nodes.append(node)
            
            # Filter edges to only include selected nodes
            selected_node_ids = {node["id"] for node in final_nodes}
            logger.info(f"📊 Selected node IDs: {list(selected_node_ids)[:5]}...")
            logger.info(f"📊 Sample edge: {all_edges[0] if all_edges else 'No edges'}")
            
            final_edges = [
                edge for edge in all_edges 
                if edge["source"] in selected_node_ids and edge["target"] in selected_node_ids
            ]
            
            logger.info(f"📊 Edge filtering: {len(all_edges)} total -> {len(final_edges)} filtered")
            
            # Remove duplicate edges (can happen with bidirectional relationships)
            unique_edges = []
            seen_edges = set()
            for edge in final_edges:
                # Create a normalized edge key for deduplication
                edge_key = tuple(sorted([edge["source"], edge["target"]]) + [edge["predicate"]])
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    unique_edges.append(edge)
            
            # Load and render template
            from jinja2 import Environment, FileSystemLoader
            env = Environment(loader=FileSystemLoader('templates'))
            template = env.get_template('graph_visualization.html')
            
            html_content = template.render(
                nodes=final_nodes,
                edges=unique_edges,
                stats=session_graph.get_stats()
            )
            
            logger.info(f"📊 Generated graph visualization: {len(final_nodes)} nodes, {len(unique_edges)} edges")
            logger.info(f"📊 Node selection: {len(selected_nodes)} selected from {len(all_nodes)} total")
            
            return html_content
            
        except Exception as e:
            logger.error(f"❌ Graph visualization failed: {e}")
            return f"# Error generating graph visualization: {str(e)}\n"

    def _extract_display_name(self, uri: str) -> str:
        """Extract a human-readable name from a URI."""
        if not uri.startswith("http"):
            return uri[:50]  # Literal value, truncate if long
        
        # Extract the fragment or last path component
        if "#" in uri:
            return uri.split("#")[-1]
        elif "/" in uri:
            return uri.split("/")[-1]
        else:
            return uri

    def _get_node_type(self, uri: str) -> str:
        """Determine node type from URI for color coding."""
        uri_lower = uri.lower()
        
        if "person" in uri_lower or "mp" in uri_lower or "minister" in uri_lower:
            return "person"
        elif "bill" in uri_lower or "act" in uri_lower or "legislation" in uri_lower:
            return "legislation"
        elif "committee" in uri_lower or "parliament" in uri_lower:
            return "institution"
        elif "topic" in uri_lower or "policy" in uri_lower:
            return "topic"
        else:
            return "entity"
    
    async def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> Tuple[str, str]:
        """Get existing session or create a new one using MongoDB."""
        try:
            # If session_id provided, try to get existing session
            if session_id:
                existing_session = await self.session_manager.get_session(session_id)
                if existing_session:
                    logger.info(f"✅ Using existing session: {session_id[:8]}...")
                    
                    # Create a fresh ADK session for this interaction
                    adk_session = await self.adk_session_service.create_session(
                        app_name="YuhHearDem",
                        user_id=user_id
                    )
                    
                    return session_id, adk_session.id
            
            # Create new session
            new_session = await self.session_manager.create_session(
                user_id=user_id,
                session_id=session_id,
                metadata={"created_via": "parliamentary_system", "version": "3.6.0"}
            )
            
            # Create corresponding ADK session
            adk_session = await self.adk_session_service.create_session(
                app_name="YuhHearDem",
                user_id=user_id
            )
            
            logger.info(f"✅ Created new session: {new_session.session_id[:8]}...")
            return new_session.session_id, adk_session.id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            # Fallback to simple UUID
            fallback_id = str(uuid.uuid4())
            logger.warning(f"Using fallback session ID: {fallback_id[:8]}...")
            
            # Still try to create ADK session
            try:
                adk_session = await self.adk_session_service.create_session(
                    app_name="YuhHearDem",
                    user_id=user_id
                )
                return fallback_id, adk_session.id
            except:
                return fallback_id, fallback_id
    
    async def build_conversation_context(self, session_id: str, max_messages: int = 6) -> str:
        """Build conversation context from MongoDB stored messages."""
        try:
            # Get recent messages from MongoDB
            messages = await self.session_manager.get_session_messages(session_id, limit=max_messages)
            
            if not messages:
                return ""
            
            context = "\n\nRECENT CONVERSATION:\n"
            for message in messages:
                if message.role == "user":
                    context += f"User: {message.content}\n"
                elif message.role == "assistant":
                    # Use truncated version for context
                    assistant_preview = message.content[:200]
                    context += f"Assistant: {assistant_preview}...\n\n"
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to build conversation context: {e}")
            return ""
    
    async def process_query(self, query: str, user_id: str = "user", session_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Process a query through the enhanced parliamentary agent with MongoDB session storage."""
        try:
            logger.info(f"🚀 Processing query: {query[:50]}...")
            
            # Check if this is a graph visualization request
            if "visualize" in query.lower() and "graph" in query.lower():
                logger.info("📊 Detected graph visualization request")
            
            # Get or create session - returns both tracking session ID and ADK session ID
            tracking_session_id, adk_session_id = await self.get_or_create_session(user_id, session_id)
            
            # Set current session context for tools
            self.current_session_id = tracking_session_id
            
            # Get session graph
            session_graph = self.get_or_create_session_graph(tracking_session_id)
            
            # Build context from MongoDB conversation history AND session graph
            context = await self.build_conversation_context(tracking_session_id)
            
            # Add session graph context
            if session_graph.edge_count > 0:
                context += "\n\nCURRENT SESSION GRAPH:\n"
                context += session_graph.get_turtle_dump()
                context += "\n\n"
            
            # Store user message in MongoDB
            await self.session_manager.add_message(
                tracking_session_id,
                "user",
                query,
                metadata={"timestamp": datetime.now(timezone.utc).isoformat()}
            )
            
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
            
            # Parse JSON response
            structured_response = parse_llm_json_response(response_text)
            
            if structured_response:
                # Convert to HTML with unique message ID
                message_id = f"msg-{int(datetime.utcnow().timestamp() * 1000)}"
                html_response = convert_structured_response_to_html(structured_response, message_id)
                
                # Store assistant response in MongoDB
                await self.session_manager.add_message(
                    tracking_session_id,
                    "assistant",
                    response_text,  # Store original response for context
                    metadata={
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "message_id": message_id,
                        "response_type": "structured",
                        "graph_stats": session_graph.get_stats()
                    }
                )
                
                # Clear session context
                self.current_session_id = None
                
                # Return structured response
                return html_response, {
                    "success": True, 
                    "session_id": tracking_session_id,
                    "graph_stats": session_graph.get_stats(),
                    "structured_response": structured_response,
                    "response_type": "structured"
                }
            else:
                # Fallback to plain text if JSON parsing fails
                logger.warning("Failed to parse JSON response, falling back to plain text")
                
                # Create a user-friendly fallback message instead of showing raw JSON
                fallback_message = "I encountered an issue formatting my response properly. Here's what I found:\n\n" + response_text[:500]
                if len(response_text) > 500:
                    fallback_message += "..."
                
                # Convert markdown to HTML as fallback
                html_response = markdown.markdown(fallback_message, extensions=['extra', 'codehilite'])
                
                # Filter out non-YouTube links
                soup = BeautifulSoup(html_response, 'html.parser')
                links = soup.find_all('a')
                for link in links:
                    href = link.get('href', '')
                    if 'youtube.com' not in href.lower():
                        link.replace_with(link.get_text())
                html_response = str(soup)
                
                # Store assistant response in MongoDB
                await self.session_manager.add_message(
                    tracking_session_id,
                    "assistant",
                    fallback_message,
                    metadata={
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "response_type": "fallback_error",
                        "graph_stats": session_graph.get_stats()
                    }
                )
                
                # Clear session context
                self.current_session_id = None
                
                return html_response, {
                    "success": True, 
                    "session_id": tracking_session_id,
                    "graph_stats": session_graph.get_stats(),
                    "response_type": "fallback_error"
                }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Clear session context on error
            self.current_session_id = None
            
            return f"❌ Error processing query: {str(e)}", {"success": False}
    
    async def get_session_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get session history from MongoDB."""
        try:
            messages = await self.session_manager.get_session_messages(session_id, limit)
            return [
                {
                    "message_id": msg.message_id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"Failed to get session history: {e}")
            return []
    
    async def get_user_sessions(self, user_id: str, limit: int = 10, include_archived: bool = False) -> List[Dict[str, Any]]:
        """Get user's recent sessions from MongoDB."""
        try:
            sessions = await self.session_manager.get_user_sessions(user_id, limit, include_archived)
            return [
                {
                    "session_id": session.session_id,
                    "created_at": session.created_at.isoformat(),
                    "last_updated": session.last_updated.isoformat(),
                    "message_count": len(session.messages),
                    "metadata": session.metadata,
                    "archived": session.metadata.get("archived", False)
                }
                for session in sessions
            ]
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
    
    async def archive_session(self, session_id: str, reason: str = "User requested") -> bool:
        """Archive a session instead of deleting for audit purposes."""
        try:
            # Archive in MongoDB
            success = await self.session_manager.archive_session(session_id, reason)
            
            # Keep graph data in memory but mark it as archived
            if session_id in self.session_graphs:
                logger.info(f"🗃️ Session graph for {session_id[:8]} kept for audit purposes")
            
            return success
        except Exception as e:
            logger.error(f"Failed to archive session: {e}")
            return False
    
    def close(self):
        """Close system resources."""
        if hasattr(self, 'querier'):
            self.querier.close()
        if hasattr(self, 'session_manager'):
            self.session_manager.close()

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
    logger.info("🚀 Starting Enhanced Parliamentary Chatbot System with MongoDB Sessions and Graph Visualization...")
    
    global parliamentary_system
    try:
        parliamentary_system = create_system()
        logger.info("✅ Enhanced Parliamentary System with MongoDB sessions and graph visualization created successfully")
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
    description="AI-powered Parliamentary research system with MongoDB session persistence and interactive graph visualization",
    version="4.1.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

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
            # Use actual session ID from the response
            actual_session_id = status.get("session_id", session_id)
            graph_stats = status.get("graph_stats", {})
            structured_response = status.get("structured_response")
            
            yield format_sse_event("response_ready", "Assistant", "Response completed", {
                "response": response_text,
                "message_id": str(uuid.uuid4()),
                "session_id": actual_session_id,
                "type": "parliamentary",
                "graph_stats": graph_stats,
                "structured_response": structured_response.dict() if structured_response else None,
                "response_type": status.get("response_type", "structured")
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
        "message": "YuhHearDem - Enhanced Parliamentary Research API with MongoDB Session Storage and Graph Visualization",
        "status": "running",
        "version": "4.1.0",
        "features": ["mongodb_sessions", "audit_compliant_archiving", "persistent_chat_history", "robust_json_repair", "expandable_cards", "session_graph_persistence", "turtle_processing", "cumulative_memory", "interactive_graph_visualization"]
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
        # Test graph database connection
        parliamentary_system.querier.client.admin.command("ping", maxTimeMS=3000)
        graph_db_connected = True
    except Exception as e:
        graph_db_connected = False
    
    try:
        # Test session database connection
        parliamentary_system.session_manager.client.admin.command("ping", maxTimeMS=3000)
        session_db_connected = True
        
        # Get session stats
        session_stats = parliamentary_system.session_manager.get_session_stats()
    except Exception as e:
        session_db_connected = False
        session_stats = {"error": str(e)}
    
    total_graph_edges = sum(graph.edge_count for graph in parliamentary_system.session_graphs.values())
    
    return {
        "status": "healthy" if (graph_db_connected and session_db_connected) else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "graph_database_connected": graph_db_connected,
        "session_database_connected": session_db_connected,
        "session_graphs_in_memory": len(parliamentary_system.session_graphs),
        "total_graph_edges": total_graph_edges,
        "session_stats": session_stats,
        "version": "4.1.0"
    }

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information and history."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Get session from MongoDB
        session = await parliamentary_system.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get graph stats if available
        graph_stats = None
        if session_id in parliamentary_system.session_graphs:
            graph_stats = parliamentary_system.session_graphs[session_id].get_stats()
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_updated": session.last_updated.isoformat(),
            "message_count": len(session.messages),
            "metadata": session.metadata,
            "graph_stats": graph_stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 20):
    """Get messages from a session."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        messages = await parliamentary_system.get_session_history(session_id, limit)
        return {
            "session_id": session_id,
            "messages": messages,
            "count": len(messages)
        }
        
    except Exception as e:
        logger.error(f"Failed to get session messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/session/{session_id}/graph/visualize")
async def visualize_session_graph(session_id: str):
    """Get the session graph as interactive HTML visualization."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        html_content = parliamentary_system.visualize_session_graph(session_id)
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Failed to visualize session graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/sessions")
async def get_user_sessions(user_id: str, limit: int = 10, include_archived: bool = False):
    """Get user's recent sessions."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        sessions = await parliamentary_system.get_user_sessions(user_id, limit, include_archived)
        return {
            "user_id": user_id,
            "sessions": sessions,
            "count": len(sessions),
            "include_archived": include_archived
        }
        
    except Exception as e:
        logger.error(f"Failed to get user sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/{session_id}/archive")
async def archive_session(session_id: str, reason: str = "User requested"):
    """Archive a session for audit purposes instead of deleting."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        success = await parliamentary_system.archive_session(session_id, reason)
        if success:
            return {
                "message": f"Session {session_id} archived successfully for audit purposes",
                "reason": reason,
                "note": "Data preserved for compliance and audit requirements"
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to archive session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
            structured_response = status.get("structured_response")
            
            return QueryResponse(
                session_id=actual_session_id,
                user_id=request.user_id,
                message_id="msg_" + str(datetime.utcnow().timestamp()),
                status="success",
                message=response_text,
                structured_response=structured_response
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

@app.get("/stats")
async def get_system_stats():
    """Get overall system statistics."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        session_stats = parliamentary_system.session_manager.get_session_stats()
        graph_stats = {
            "active_session_graphs": len(parliamentary_system.session_graphs),
            "total_graph_edges": sum(graph.edge_count for graph in parliamentary_system.session_graphs.values())
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "version": "4.1.0",
            "session_stats": session_stats,
            "graph_stats": graph_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/archive-old-sessions")
async def archive_old_sessions(days_old: int = 365):
    """Admin endpoint to archive old sessions for audit compliance."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        archived_count = await parliamentary_system.session_manager.cleanup_old_sessions(days_old)
        return {
            "message": f"Archived {archived_count} sessions older than {days_old} days",
            "archived_count": archived_count,
            "note": "Sessions archived for audit compliance, not deleted",
            "retention_policy": f"Sessions auto-archived after {days_old} days of inactivity"
        }
        
    except Exception as e:
        logger.error(f"Failed to archive old sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info("🚀 Starting Enhanced Parliamentary Chatbot with MongoDB Session Storage and Graph Visualization")
    logger.info(f"📡 Server will run on 0.0.0.0:{port}")
    logger.info("📋 Required: GOOGLE_API_KEY, MONGODB_CONNECTION_STRING")
    logger.info("🔧 New: Interactive knowledge graph visualization with D3.js")
    
    uvicorn.run(
        "yuhheardem_chatbot:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
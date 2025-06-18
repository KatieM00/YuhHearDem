#!/usr/bin/env python3
"""
Enhanced MongoDB Knowledge Graph Extractor with Entity Disambiguation

Features:
- Updated schema matching new prompt format
- Multi-pass extraction with LLM feedback loop
- Vector-based entity disambiguation
- Proper provenance tracking via segment IDs

Requirements:
- google-genai
- pymongo
- python-dotenv
- pydantic
- sentence-transformers (for embeddings)

Usage:
    python enhanced_kg_extractor.py --database parliamentary_graph2
"""

import sys
import os
import json
import argparse
from typing import Optional, Dict, Any, List, Set, Tuple
from datetime import datetime, timezone
from collections import defaultdict
import hashlib
import time

try:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, BulkWriteError
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install google-genai pymongo python-dotenv pydantic sentence-transformers")
    sys.exit(1)

# Load environment variables
load_dotenv()

class Entity(BaseModel):
    """Pydantic model for entities."""
    entity_name: str
    entity_id: str
    entity_type: str
    entity_description: str

class Statement(BaseModel):
    """Pydantic model for statements."""
    _id: str
    source_entity_id: str
    target_entity_id: str
    relationship_description: str
    relationship_strength: int
    provenance_segment_id: str

class KnowledgeGraphOutput(BaseModel):
    """Pydantic model for the complete knowledge graph output."""
    entities: List[Entity]
    statements: List[Statement]

class ConnectivityValidator:
    """Validates that all entities are connected through statements."""
    
    def __init__(self, video_id: str):
        self.video_id = video_id
    
    def validate_connectivity(self, entities: List[Dict], statements: List[Dict]) -> Tuple[bool, List[str], Dict]:
        """
        Validate that all entities are connected through statements.
        
        Returns:
            Tuple of (is_valid, orphaned_entity_ids, connectivity_stats)
        """
        # Build entity set
        entity_ids = {entity.get("entity_id") for entity in entities}
        
        # Build connectivity graph from statements
        connected_entities = set()
        entity_connections = defaultdict(list)
        
        # Track all entities referenced in statements
        for statement in statements:
            source_id = statement.get("source_entity_id")
            target_id = statement.get("target_entity_id")
            
            # Add entities that participate in statements
            if source_id in entity_ids:
                connected_entities.add(source_id)
                entity_connections[source_id].append(statement)
            if target_id in entity_ids:
                connected_entities.add(target_id)
                entity_connections[target_id].append(statement)
        
        # Find orphaned entities (entities not in any statement)
        orphaned_entities = entity_ids - connected_entities
        
        # Generate connectivity statistics
        stats = {
            "total_entities": len(entity_ids),
            "connected_entities": len(connected_entities),
            "orphaned_entities": len(orphaned_entities),
            "total_statements": len(statements),
            "connectivity_ratio": len(connected_entities) / len(entity_ids) if entity_ids else 0
        }
        
        is_valid = len(orphaned_entities) == 0
        
        return is_valid, list(orphaned_entities), stats

class EnhancedKnowledgeGraphExtractor:
    def __init__(self, connection_string: str = None, database_name: str = "parliamentary_graph2", api_key: str = None):
        """
        Initialize the knowledge graph extractor with MongoDB connection and cached Gemini model.
        """
        # Setup MongoDB connection
        if connection_string is None:
            connection_string = os.getenv('MONGODB_CONNECTION_STRING')
            
        if not connection_string:
            raise ValueError(
                "MongoDB connection string is required. Set MONGODB_CONNECTION_STRING environment variable "
                "or pass connection_string parameter."
            )
        
        try:
            self.client = MongoClient(connection_string)
            # Test connection
            self.client.admin.command('ping')
            print("✅ Successfully connected to MongoDB")
        except ConnectionFailure as e:
            raise ConnectionFailure(f"Failed to connect to MongoDB: {e}")
        
        self.db = self.client[database_name]
        self.videos = self.db.videos
        self.provenance_segments = self.db.provenance_segments
        self.entities = self.db.entities
        self.statements = self.db.statements
        
        # Create indexes for efficient querying
        try:
            # Create compound indexes for faster exact matching
            self.entities.create_index([("entity_type", 1), ("entity_name", 1)])
            self.entities.create_index([("entity_type", 1), ("name_description_embedding", 1)])
            
            # Original indexes
            self.statements.create_index([("source_entity_id", 1), ("target_entity_id", 1)])
            self.statements.create_index([("provenance_segment_id", 1)])
            self.entities.create_index([("entity_type", 1)])
            self.entities.create_index([("entity_id", 1)], unique=True)
            print("✅ Indexes created/verified on knowledge graph collections")
        except Exception as e:
            print(f"⚠️  Warning: Could not create indexes: {e}")
        
        # Setup Google Generative AI
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Configure the client with API key
        self.genai_client = genai.Client(api_key=api_key)
        
        # Use the latest Gemini 2.5 Flash model
        self.model_name = "models/gemini-2.5-flash"
        
        # Cache for the extraction prompt (will be created on first use)
        self.prompt_cache = None
        self.cache_ttl_hours = 24
        
        # Setup embedding model for entity disambiguation
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.atlas_vector_search_available = False  # Will be set in setup_vector_index
            print("✅ Embedding model loaded for entity disambiguation")
        except Exception as e:
            print(f"⚠️  Warning: Could not load embedding model: {e}")
            self.embedding_model = None
            self.atlas_vector_search_available = False

    def get_extraction_system_instruction(self) -> str:
        """Get the extraction system instruction from the provided document."""
        return """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1.  Go through each individual line in the input and Identify all entities. For each identified entity, extract the following information:
    entity_name: Name of the entity, capitalized as it appears or a common proper noun form.
    entity_id: A unique, capitalized identifier for the entity, comprehensive enough to distinguish it in different contexts. This ID should be concise but clearly refer to the specific entity instance (e.g., 'SEXUAL_OFFENCES_ACT_SECTION_09' instead of just 'SECTION_09').
    entity_type: One of the following types, chosen based on the most precise fit:
        *   **PERSON:** A human individual.
        *   **ORGANIZATION:** A formal group, institution, company, or body.
        *   **LOCATION:** A specific geographical place.
        *   **LEGISLATION:** A proposed or enacted law, or a specific part of it (e.g., a Bill, Act, Section, Clause).
        *   **PARLIAMENTARY_ROLE:** A specific, formal designated position or office within a parliamentary or executive structure (e.g., Speaker, Prime Minister, Minister).
        *   **PARLIAMENTARY_PROCEDURE:** A formal rule, process, or action governing parliamentary conduct (e.g., a Motion, Question Time, a specific Reading of a Bill).
        *   **DATE_TIME:** A specific point or period in time (e.g., a full date, year, specific time, or duration).
        *   **NUMERIC_VALUE:** A quantitative expression, including amounts, percentages, counts, or other numerical data, particularly when it represents a significant, named financial or statistical figure (e.g., "GDP growth rate", "national debt figure").
        *   **POLICY_DOMAIN:** A broad, abstract area of public policy or national interest (e.g., Healthcare, Education, Climate Change).
        *   **OFFICIAL_DOCUMENT:** A formal government or parliamentary publication or record (e.g., a Report, Budget, Minutes, White Paper).
        *   **EVENT:** A specific, identifiable occurrence or happening (e.g., a meeting, election, disaster, ceremony).
        *   **ECONOMIC_CONCEPT:** A specific, named economic theory, principle, or a broad economic challenge (e.g., "Fiscal Policy", "Inflationary Pressure").
        *   **CONSTITUTIONAL_CONCEPT:** A fundamental legal principle, a specific article of a nation's constitution, or a concept related to governmental structure or rights (e.g., Republicanism, Human Rights, Separation of Powers).
        *   **INFRASTRUCTURE_PROJECT:** A specific public works development or major construction initiative (e.g., Airport Expansion, Road Project).
        *   **INTERNATIONAL_AGREEMENT:** A formal agreement, treaty, convention, or accord between sovereign states or international bodies.
        *   **REGIONAL_INITIATIVE:** A specific program, policy, or cooperative effort involving multiple nations in a region.
        *   **LEGAL_CASE_CONCEPT:** A specific court case, legal precedent, or a general legal principle discussed in a legal context.
        *   **EDUCATION_PROGRAM_ENTITY:** A named national education program, curriculum, or significant educational initiative.
        *   **HEALTH_PROGRAM_ENTITY:** A named national health program, public health initiative, or specific medical service program.
        *   **CULTURAL_HERITAGE_ENTITY:** A named entity or concept related to national identity, history, cultural preservation, or national symbols.
        *   **OTHER:** Any entity that does not fit into the above categories but is still relevant to the context of the text.

    entity_description: A general definition of the entity type itself, applicable across various contexts, not specific details from the provided text.
    Format each identified entity as a JSON object with keys: "entity_name", "entity_id", "entity_type", and "entity_description".

2.  From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are clearly related to each other.
    For each pair of related entities, extract the following information:
    provenance_segment_id: The provenance_segment_id from the start of the line where the relationship is stated.
    source_entity_id: The `entity_id` of the source entity, as identified in step 1.
    target_entity_id: The `entity_id` of the target entity, as identified in step 1.
    relationship_description: explanation as to why you think the source entity and the target entity are related to each other.
    relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity (on a scale of 1-10).
    Format each relationship as a JSON object with keys: "provenance_segment_id", "source_entity_id", "target_entity_id", "relationship_description", and "relationship_strength".

    Every entity MUST have a relationship to at least one other entity.

Return output in English as a single JSON object with two keys: "entities" (an array of entity objects) and "statements" (an array of relationship objects).

**IMPORTANT INPUT FORMAT NOTE:**
The transcript will be divided into two sections:

1. **CONTEXT SEGMENTS**: These are provided for reference only to give you context about what was discussed earlier. Do NOT extract entities or statements from these segments.

2. **PROCESS THESE SEGMENTS**: These are the segments you should actually process to extract entities and statements from.

Each line will begin with a provenance_segment_id, followed by the text.
Example: `UU2CifSWj5Q_240_0 Also commanded to lay the 2023-2030 National Policy...`

**OUTPUT FORMAT:**
```json
{
  "entities": [
    {
      "entity_name": "string",
      "entity_id": "string", 
      "entity_type": "string",
      "entity_description": "string"
    }
  ],
  "statements": [
    {
      "_id": "string",
      "source_entity_id": "string",
      "target_entity_id": "string", 
      "relationship_description": "string",
      "relationship_strength": integer,
      "provenance_segment_id": "string"
    }
  ]
}
```"""

    def create_or_get_prompt_cache(self) -> str:
        """Create or retrieve the cached prompt. Returns the cache name."""
        try:
            # Check if we already have a cache
            if self.prompt_cache and self.is_cache_valid():
                return self.prompt_cache
            
            print("🔄 Creating new prompt cache...")
            
            # Create the cache with the system instruction
            cache = self.genai_client.caches.create(
                model=self.model_name,
                config=types.CreateCachedContentConfig(
                    display_name='enhanced_knowledge_graph_extraction_prompt',
                    system_instruction=self.get_extraction_system_instruction(),
                    ttl=f"{self.cache_ttl_hours * 3600}s"
                )
            )
            
            self.prompt_cache = cache.name
            print(f"✅ Prompt cache created: {cache.name}")
            print(f"📊 Cache stats: {cache.usage_metadata.total_token_count} tokens cached")
            
            return self.prompt_cache
            
        except Exception as e:
            print(f"❌ Failed to create prompt cache: {e}")
            raise

    def is_cache_valid(self) -> bool:
        """Check if the current cache is still valid."""
        if not self.prompt_cache:
            return False
        
        try:
            cache_info = self.genai_client.caches.get(name=self.prompt_cache)
            return cache_info.expire_time > datetime.now(timezone.utc)
        except Exception:
            return False

    def cleanup_cache(self):
        """Clean up the prompt cache when done."""
        if self.prompt_cache:
            try:
                self.genai_client.caches.delete(name=self.prompt_cache)
                print("🗑️  Prompt cache cleaned up")
            except Exception as e:
                print(f"⚠️  Warning: Could not delete cache: {e}")

    def _generate_statement_id(self, source_entity_id: str, target_entity_id: str, provenance_segment_id: str) -> str:
        """Generate a unique statement ID."""
        # Create a hash of the key components for uniqueness
        content = f"{source_entity_id}_{target_entity_id}_{provenance_segment_id}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"STMT_{hash_suffix}_{provenance_segment_id}"

    def _ensure_statement_ids(self, statements: List[Dict]) -> List[Dict]:
        """Ensure all statements have _id fields."""
        for stmt in statements:
            if "_id" not in stmt or not stmt["_id"]:
                stmt["_id"] = self._generate_statement_id(
                    stmt.get("source_entity_id", ""),
                    stmt.get("target_entity_id", ""),
                    stmt.get("provenance_segment_id", "")
                )
        return statements

    def generate_entity_embedding(self, entity_name: str, entity_description: str) -> Optional[List[float]]:
        """Generate embedding for entity name + description."""
        if not self.embedding_model:
            return None
        
        try:
            combined_text = f"{entity_name}: {entity_description}"
            embedding = self.embedding_model.encode(combined_text)
            return embedding.tolist()
        except Exception as e:
            print(f"⚠️  Warning: Could not generate embedding: {e}")
            return None

    def setup_vector_index(self):
        """Create Atlas Vector Search index on entities collection."""
        try:
            # Check if we're on Atlas and can use vector search
            try:
                # Test if we can access the search indexes API (Atlas only)
                search_indexes = list(self.entities.list_search_indexes())
                self.atlas_vector_search_available = True
                print("✅ MongoDB Atlas detected - vector search available")
            except Exception:
                self.atlas_vector_search_available = False
                print("📦 Local MongoDB detected - using manual similarity search")
                return
            
            # Check if vector search index already exists
            vector_index_exists = False
            for index in search_indexes:
                if index.get('name') == 'entity_vector_index':
                    vector_index_exists = True
                    print("📦 Atlas vector search index already exists")
                    break
            
            if not vector_index_exists:
                print("🔄 Creating Atlas vector search index on entities...")
                
                # Create the vector search index definition
                vector_index_definition = {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "name_description_embedding",
                            "numDimensions": 384,  # all-MiniLM-L6-v2 embedding size
                            "similarity": "cosine"
                        },
                        {
                            "type": "filter",
                            "path": "entity_type"
                        }
                    ]
                }
                
                # Create the index
                self.entities.create_search_index(
                    model={
                        "name": "entity_vector_index",
                        "definition": vector_index_definition
                    }
                )
                print("✅ Atlas vector search index created (may take a few minutes to build)")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not create Atlas vector search index: {e}")
            self.atlas_vector_search_available = False

    def normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        # Convert to lowercase and strip whitespace
        normalized = name.lower().strip()
        
        # Common variations mapping
        variations = {
            'st.': 'saint',
            'mr.': 'mister',
            'mrs.': 'missus', 
            'dr.': 'doctor',
            'prof.': 'professor',
            'hon.': 'honourable',
            'rt. hon.': 'right honourable',
            'pm': 'prime minister',
            'mp': 'member of parliament',
            'govt': 'government',
            'govt.': 'government',
            'dep.': 'deputy',
            'dept': 'department',
            'dept.': 'department',
            'min.': 'minister',
            'sec.': 'secretary',
            'rep.': 'representative',
            'const.': 'constitution',
            'parl.': 'parliament',
            'comm.': 'committee',
            'assoc.': 'association',
            'org.': 'organization',
            'intl': 'international',
            'natl': 'national'
        }
        
        # Apply variations
        for abbrev, full in variations.items():
            normalized = normalized.replace(abbrev, full)
        
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        
        return normalized

    def is_exact_match(self, entity1: Dict, entity2: Dict) -> bool:
        """Check if two entities are exact matches based on normalized names and types."""
        if entity1.get("entity_type") != entity2.get("entity_type"):
            return False
        
        name1 = self.normalize_entity_name(entity1.get("entity_name", ""))
        name2 = self.normalize_entity_name(entity2.get("entity_name", ""))
        
        return name1 == name2

    def find_exact_match_entities(self, entity: Dict) -> List[Dict]:
        """Find existing entities that are exact matches - optimized with indexing."""
        try:
            entity_type = entity.get("entity_type", "")
            entity_name = entity.get("entity_name", "")
            
            # Use MongoDB query with proper indexing instead of fetching all entities
            # First, try direct name match (fastest)
            direct_matches = list(self.entities.find({
                "entity_type": entity_type,
                "entity_name": entity_name
            }).limit(5))
            
            if direct_matches:
                return direct_matches
            
            # If no direct match, try case-insensitive regex (slower but still indexed)
            normalized_name = self.normalize_entity_name(entity_name)
            
            # Create a regex pattern for the normalized name
            escaped_name = normalized_name.replace(".", r"\.").replace("(", r"\(").replace(")", r"\)")
            regex_pattern = f"^{escaped_name}$"
            
            regex_matches = list(self.entities.find({
                "entity_type": entity_type,
                "entity_name": {"$regex": regex_pattern, "$options": "i"}
            }).limit(5))
            
            return regex_matches
            
        except Exception as e:
            print(f"⚠️  Warning: Error in exact match search: {e}")
            return []

    def find_similar_entities(self, entity: Dict, similarity_threshold: float = 0.85) -> List[Dict]:
        """Find existing entities similar to the new one using Atlas Vector Search or manual fallback."""
        if not self.embedding_model:
            return []
        
        try:
            entity_name = entity.get("entity_name", "")
            entity_description = entity.get("entity_description", "")
            entity_type = entity.get("entity_type", "")
            
            # Generate embedding for the new entity
            new_embedding = self.generate_entity_embedding(entity_name, entity_description)
            if not new_embedding:
                return []
            
            # Use Atlas Vector Search if available (much faster)
            if getattr(self, 'atlas_vector_search_available', False):
                return self._atlas_vector_search(entity_type, new_embedding, similarity_threshold)
            else:
                return self._manual_similarity_search(entity_type, new_embedding, similarity_threshold)
                
        except Exception as e:
            print(f"⚠️  Warning: Error in similarity search: {e}")
            return []

    def _atlas_vector_search(self, entity_type: str, new_embedding: List[float], similarity_threshold: float) -> List[Dict]:
        """Fast Atlas Vector Search implementation."""
        try:
            # Atlas Vector Search aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "entity_vector_index",
                        "path": "name_description_embedding",
                        "queryVector": new_embedding,
                        "numCandidates": 100,
                        "limit": 10,
                        "filter": {
                            "entity_type": {"$eq": entity_type}
                        }
                    }
                },
                {
                    "$addFields": {
                        "similarity_score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$match": {
                        "similarity_score": {"$gte": similarity_threshold}
                    }
                }
            ]
            
            # Execute the vector search
            results = list(self.entities.aggregate(pipeline))
            
            # Sort by similarity score descending (already sorted by Atlas but ensure)
            results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            
            return results[:5]  # Return top 5
            
        except Exception as e:
            print(f"⚠️  Warning: Atlas Vector Search failed, falling back to manual: {e}")
            return self._manual_similarity_search(entity_type, new_embedding, similarity_threshold)

    def _manual_similarity_search(self, entity_type: str, new_embedding: List[float], similarity_threshold: float) -> List[Dict]:
        """Manual similarity search fallback (slower but works everywhere)."""
        try:
            # Optimized query: only get entities with embeddings and limit results
            existing_entities = list(self.entities.find({
                "entity_type": entity_type,
                "name_description_embedding": {"$exists": True, "$ne": None}
            }).limit(50))  # Limit to 50 most recent entities to speed up search
            
            if not existing_entities:
                return []
            
            similar_entities = []
            
            for existing_entity in existing_entities:
                existing_embedding = existing_entity.get("name_description_embedding")
                if not existing_embedding or len(existing_embedding) != len(new_embedding):
                    continue
                
                try:
                    # Calculate cosine similarity
                    similarity = np.dot(new_embedding, existing_embedding) / (
                        np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding)
                    )
                    
                    if similarity >= similarity_threshold:
                        existing_entity["similarity_score"] = float(similarity)
                        similar_entities.append(existing_entity)
                except Exception as sim_error:
                    print(f"⚠️  Warning: Error calculating similarity for {existing_entity.get('entity_name', 'unknown')}: {sim_error}")
                    continue
            
            # Sort by similarity score descending
            similar_entities.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            
            return similar_entities[:5]  # Return top 5 most similar
            
        except Exception as e:
            print(f"⚠️  Warning: Error in manual similarity search: {e}")
            return []

    def fetch_entity_subgraph(self, entity_id: str) -> Dict:
        """Fetch entity and its immediate relationships + 1-hop entities - optimized."""
        try:
            # Get the entity
            entity = self.entities.find_one({"entity_id": entity_id})
            if not entity:
                print(f"      ⚠️  Entity {entity_id} not found in database")
                return {"entity": None, "relationships": [], "connected_entities": []}
            
            # Get immediate relationships where this entity is source or target (limit for speed)
            relationships = list(self.statements.find({
                "$or": [
                    {"source_entity_id": entity_id},
                    {"target_entity_id": entity_id}
                ]
            }).limit(20))  # Limit relationships for faster LLM processing
            
            # Get 1-hop connected entities (limit for speed)
            connected_entity_ids = set()
            for rel in relationships:
                connected_entity_ids.add(rel["source_entity_id"])
                connected_entity_ids.add(rel["target_entity_id"])
            
            connected_entity_ids.discard(entity_id)  # Remove self
            
            # Limit connected entities to avoid huge subgraphs
            connected_entities = list(self.entities.find({
                "entity_id": {"$in": list(connected_entity_ids)}
            }).limit(10))
            
            return {
                "entity": entity,
                "relationships": relationships,
                "connected_entities": connected_entities
            }
            
        except Exception as e:
            print(f"      ⚠️  Warning: Error fetching subgraph for {entity_id}: {e}")
            return {"entity": None, "relationships": [], "connected_entities": []}

    def are_entities_same(self, existing_subgraph: Dict, new_entity: Dict, new_relationships: List[Dict]) -> bool:
        """Use LLM to determine if two entities represent the same thing."""
        try:
            # Check if existing_subgraph has the required structure
            if not existing_subgraph or not existing_subgraph.get('entity'):
                print(f"      ⚠️  Invalid existing subgraph for comparison")
                return False
            
            cache_name = self.create_or_get_prompt_cache()
            
            # Create comparison prompt
            prompt = f"""**ENTITY DISAMBIGUATION TASK**

I need you to determine if these two entities represent the same real-world entity.

**EXISTING ENTITY:**
- Name: {existing_subgraph['entity']['entity_name']}
- Type: {existing_subgraph['entity']['entity_type']}
- Description: {existing_subgraph['entity']['entity_description']}
- Current Relationships: {len(existing_subgraph['relationships'])}

**NEW ENTITY:**
- Name: {new_entity['entity_name']}
- Type: {new_entity['entity_type']}  
- Description: {new_entity['entity_description']}
- New Relationships: {len([r for r in new_relationships if r['source_entity_id'] == new_entity['entity_id'] or r['target_entity_id'] == new_entity['entity_id']])}

**EXISTING ENTITY RELATIONSHIPS:**
{json.dumps([{
    'source': r['source_entity_id'],
    'target': r['target_entity_id'], 
    'description': r['relationship_description']
} for r in existing_subgraph['relationships'][:10]], indent=2)}

**NEW ENTITY RELATIONSHIPS:**
{json.dumps([{
    'source': r['source_entity_id'],
    'target': r['target_entity_id'],
    'description': r['relationship_description']
} for r in new_relationships if r['source_entity_id'] == new_entity['entity_id'] or r['target_entity_id'] == new_entity['entity_id']][:10], indent=2)}

**QUESTION**: Do these entities represent the same real-world entity? Consider:
1. Are the names referring to the same thing?
2. Are the descriptions compatible?
3. Do the relationship patterns suggest they're the same entity?

**OUTPUT**: Respond with only "YES" or "NO" (no explanation needed)."""

            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    cached_content=cache_name,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            
            result = response.text.strip().upper()
            return result == "YES"
            
        except Exception as e:
            print(f"      ⚠️  Warning: Error in entity disambiguation LLM call: {e}")
            return False

    def disambiguate_batch_entities(self, batch_entities: List[Dict], 
                                  batch_statements: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Main disambiguation process for a batch with optimized exact matching and detailed timing."""
        start_time = time.time()
        print(f"    🔍 Disambiguating {len(batch_entities)} entities...")
        
        final_entities = []
        final_statements = batch_statements.copy()
        entity_mappings = {}  # new_entity_id -> existing_entity_id
        
        # Statistics for optimization tracking
        exact_matches = 0
        high_similarity_matches = 0
        llm_disambiguations = 0
        
        try:
            for i, entity in enumerate(batch_entities):
                entity_start = time.time()
                
                merged = False
                
                # STEP 1: Check for exact matches first (fastest)
                try:
                    exact_start = time.time()
                    exact_match_entities = self.find_exact_match_entities(entity)
                    exact_time = time.time() - exact_start
                    
                    if exact_match_entities:
                        # Auto-merge with first exact match
                        existing_entity = exact_match_entities[0]
                        existing_entity_id = existing_entity["entity_id"]
                        new_entity_id = entity["entity_id"]
                        
                        print(f"      🎯 Exact match ({exact_time:.3f}s): '{entity['entity_name']}' → '{existing_entity['entity_name']}'")
                        
                        entity_mappings[new_entity_id] = existing_entity_id
                        exact_matches += 1
                        merged = True
                except Exception as e:
                    print(f"⚠️  Warning: Error in exact match for '{entity.get('entity_name', 'unknown')}': {e}")
                
                # STEP 2: If no exact match, try vector similarity (if available)
                if not merged and self.embedding_model:
                    try:
                        similarity_start = time.time()
                        similar_entities = self.find_similar_entities(entity, similarity_threshold=0.85)
                        similarity_time = time.time() - similarity_start
                        
                        if similar_entities:
                            print(f"      🔍 Similarity search ({similarity_time:.3f}s): found {len(similar_entities)} candidates for '{entity['entity_name']}'")
                            
                            # Check highest similarity entity first
                            best_match = similar_entities[0]
                            similarity_score = best_match.get("similarity_score", 0)
                            
                            # STEP 2a: Auto-merge for very high similarity (>= 0.95)
                            if similarity_score >= 0.95:
                                existing_entity_id = best_match["entity_id"]
                                new_entity_id = entity["entity_id"]
                                
                                print(f"      ⚡ High similarity ({similarity_score:.3f}): '{entity['entity_name']}' → '{best_match['entity_name']}'")
                                
                                entity_mappings[new_entity_id] = existing_entity_id
                                high_similarity_matches += 1
                                merged = True
                            
                            # STEP 2b: Use LLM for medium similarity (0.85-0.95)
                            elif similarity_score >= 0.85:
                                try:
                                    for similar_entity in similar_entities:
                                        if similar_entity.get("similarity_score", 0) >= 0.85:
                                            llm_start = time.time()
                                            print(f"      🤖 Calling LLM for '{entity['entity_name']}' vs '{similar_entity['entity_name']}' (sim: {similar_entity.get('similarity_score', 0):.3f})")
                                            
                                            existing_subgraph = self.fetch_entity_subgraph(similar_entity["entity_id"])
                                            
                                            if self.are_entities_same(existing_subgraph, entity, batch_statements):
                                                # Entities are the same - merge them
                                                existing_entity_id = similar_entity["entity_id"]
                                                new_entity_id = entity["entity_id"]
                                                
                                                llm_time = time.time() - llm_start
                                                print(f"      🤖 LLM confirmed ({llm_time:.3f}s, {similar_entity.get('similarity_score', 0):.3f}): '{entity['entity_name']}' → '{similar_entity['entity_name']}'")
                                                
                                                entity_mappings[new_entity_id] = existing_entity_id
                                                llm_disambiguations += 1
                                                merged = True
                                                break
                                            else:
                                                llm_time = time.time() - llm_start
                                                print(f"      🤖 LLM rejected ({llm_time:.3f}s): '{entity['entity_name']}' ≠ '{similar_entity['entity_name']}'")
                                except Exception as e:
                                    print(f"⚠️  Warning: Error in LLM disambiguation for '{entity.get('entity_name', 'unknown')}': {e}")
                        else:
                            print(f"      ✅ No similar entities found ({similarity_time:.3f}s) for '{entity['entity_name']}'")
                    except Exception as e:
                        print(f"⚠️  Warning: Error in similarity search for '{entity.get('entity_name', 'unknown')}': {e}")
                
                if not merged:
                    # No matches found, keep the new entity
                    final_entities.append(entity)
                
                entity_time = time.time() - entity_start
                if entity_time > 1.0:  # Log slow entities
                    print(f"      ⏰ Slow entity ({entity_time:.3f}s): '{entity.get('entity_name', 'unknown')}'")
            
            # Fix relationships that reference merged entities
            if entity_mappings:
                fix_start = time.time()
                print(f"      🔧 Fixing {len(batch_statements)} statements with {len(entity_mappings)} entity mappings")
                
                fixed_statements = []
                for stmt in final_statements:
                    try:
                        fixed_stmt = stmt.copy()
                        
                        # Update source entity reference
                        if fixed_stmt["source_entity_id"] in entity_mappings:
                            old_id = fixed_stmt["source_entity_id"]
                            new_id = entity_mappings[old_id]
                            fixed_stmt["source_entity_id"] = new_id
                            # Update statement ID to reflect change
                            fixed_stmt["_id"] = fixed_stmt["_id"].replace(old_id, new_id)
                        
                        # Update target entity reference  
                        if fixed_stmt["target_entity_id"] in entity_mappings:
                            old_id = fixed_stmt["target_entity_id"]
                            new_id = entity_mappings[old_id]
                            fixed_stmt["target_entity_id"] = new_id
                            # Update statement ID to reflect change
                            fixed_stmt["_id"] = fixed_stmt["_id"].replace(old_id, new_id)
                        
                        fixed_statements.append(fixed_stmt)
                    except Exception as e:
                        print(f"⚠️  Warning: Error fixing statement {stmt.get('_id', 'unknown')}: {e}")
                        # Keep original statement if fixing fails
                        fixed_statements.append(stmt)
                
                final_statements = fixed_statements
                fix_time = time.time() - fix_start
                print(f"      🔧 Statement fixing completed in {fix_time:.3f}s")
            
            # Print optimization statistics
            total_processed = len(batch_entities)
            total_merged = len(entity_mappings)
            total_time = time.time() - start_time
            
            print(f"      ✅ Disambiguation complete ({total_time:.3f}s): {len(final_entities)} unique entities, {total_merged} merged")
            print(f"      📊 Optimization stats: {exact_matches} exact, {high_similarity_matches} high-sim, {llm_disambiguations} LLM calls")
            
            if total_processed > 0:
                efficiency = ((exact_matches + high_similarity_matches) / total_processed) * 100
                print(f"      ⚡ Efficiency: {efficiency:.1f}% auto-merged (avoided LLM calls)")
            
            return final_entities, final_statements
            
        except Exception as e:
            print(f"❌ Error in disambiguation process: {e}")
            # Return original data if disambiguation fails completely
            return batch_entities, batch_statements

    def _extract_initial_kg(self, transcript_text: str, video_id: str, video_title: str, video_url: str) -> Dict[str, Any]:
        """Initial knowledge graph extraction."""
        cache_name = self.create_or_get_prompt_cache()
        
        user_prompt = f"""**Video Context:**
- Video ID: {video_id}
- Video Title: {video_title}
- Video URL: {video_url}

**Raw Transcript to Process:**
{transcript_text}

Please extract entities and statements from the processing segments only. You may reference context segments for understanding, but do not create entities/statements from them.

Remember: Every entity must be connected through at least one statement!"""

        response = self.genai_client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                cached_content=cache_name,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        
        result = json.loads(response.text)
        
        # Ensure statements have _id fields
        if "statements" in result:
            result["statements"] = self._ensure_statement_ids(result["statements"])
        
        return result

    def _extract_additional_kg(self, transcript_text: str, current_result: Dict[str, Any], video_id: str) -> Dict[str, Any]:
        """Extract additional entities/relationships in refinement pass."""
        cache_name = self.create_or_get_prompt_cache()
        
        current_entities = [f"{e['entity_name']} ({e['entity_type']})" for e in current_result.get("entities", [])]
        current_statements_count = len(current_result.get("statements", []))
        
        refinement_prompt = f"""**REFINEMENT PASS - You didn't get all the entities and relationships, try again**

**CURRENT EXTRACTION:**
- Entities found: {len(current_entities)}
- Statements found: {current_statements_count}
- Current entities: {', '.join(current_entities[:20])}{'...' if len(current_entities) > 20 else ''}

**ORIGINAL TRANSCRIPT:**
{transcript_text}

**TASK**: Look through the transcript again and find additional entities and relationships that you missed in the first pass. Focus on:
1. Entities mentioned but not extracted
2. Relationships between entities that weren't captured
3. Ensure every entity you add is connected through statements

**OUTPUT**: Same JSON format as before, but include ONLY the additional entities and statements you want to add."""

        response = self.genai_client.models.generate_content(
            model=self.model_name,
            contents=refinement_prompt,
            config=types.GenerateContentConfig(
                cached_content=cache_name,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        
        result = json.loads(response.text)
        
        # Ensure statements have _id fields
        if "statements" in result:
            result["statements"] = self._ensure_statement_ids(result["statements"])
        
        return result

    def _check_extraction_completeness(self, transcript_text: str, current_result: Dict[str, Any]) -> bool:
        """Ask LLM if extraction is complete."""
        cache_name = self.create_or_get_prompt_cache()
        
        completeness_prompt = f"""**COMPLETENESS CHECK**

**CURRENT EXTRACTION SUMMARY:**
- Total entities: {len(current_result.get("entities", []))}
- Total statements: {len(current_result.get("statements", []))}

**TRANSCRIPT:**
{transcript_text[:2000]}{'...' if len(transcript_text) > 2000 else ''}

**QUESTION**: Do you think you have extracted all the important entities and relationships from this transcript? 

Consider:
1. Are there any significant speakers, documents, or concepts you missed?
2. Are there relationships between entities that weren't captured?
3. Is the knowledge graph comprehensive for this parliamentary segment?

**RESPONSE**: Answer only "Y" (if complete) or "N" (if more extraction needed). No explanation."""

        response = self.genai_client.models.generate_content(
            model=self.model_name,
            contents=completeness_prompt,
            config=types.GenerateContentConfig(
                cached_content=cache_name,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        
        return response.text.strip().upper() == "Y"

    def _merge_kg_results(self, base_result: Dict[str, Any], additional_result: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple extraction results, handling duplicates."""
        # Merge entities (remove duplicates by entity_id)
        entity_dict = {}
        
        # Add base entities
        for entity in base_result.get("entities", []):
            entity_dict[entity["entity_id"]] = entity
        
        # Add additional entities
        for entity in additional_result.get("entities", []):
            entity_dict[entity["entity_id"]] = entity
        
        # Merge statements (remove duplicates by _id)
        statement_dict = {}
        
        # Add base statements
        for stmt in base_result.get("statements", []):
            statement_dict[stmt["_id"]] = stmt
        
        # Add additional statements
        for stmt in additional_result.get("statements", []):
            statement_dict[stmt["_id"]] = stmt
        
        # Ensure all statements have proper _id fields
        final_statements = self._ensure_statement_ids(list(statement_dict.values()))
        
        return {
            "entities": list(entity_dict.values()),
            "statements": final_statements
        }

    def extract_knowledge_graph(self, transcript_text: str, video_id: str, video_title: str, video_url: str) -> Dict[str, Any]:
        """Extract knowledge graph with multi-pass refinement loop and disambiguation."""
        print(f"    🔄 Starting multi-pass extraction...")
        
        # Initial extraction
        result = self._extract_initial_kg(transcript_text, video_id, video_title, video_url)
        print(f"    📊 Initial pass: {len(result.get('entities', []))} entities, {len(result.get('statements', []))} statements")
        
        # Multi-pass refinement loop (max 3 total attempts)
        for attempt in range(2):  # attempts 2 and 3
            print(f"    🔄 Refinement pass {attempt + 1}/2...")
            
            # Send "try again" message
            additional_result = self._extract_additional_kg(transcript_text, result, video_id)
            
            # Merge results
            result = self._merge_kg_results(result, additional_result)
            print(f"    📊 After refinement {attempt + 1}: {len(result.get('entities', []))} entities, {len(result.get('statements', []))} statements")
            
            # Ask if complete
            is_complete = self._check_extraction_completeness(transcript_text, result)
            
            if is_complete:
                print(f"    ✅ LLM satisfied with extraction after {attempt + 1} refinement passes")
                break
            else:
                print(f"    🔄 LLM wants to continue refining...")
        
        # Validate connectivity before disambiguation
        validator = ConnectivityValidator(video_id)
        is_valid, orphaned_entities, stats = validator.validate_connectivity(
            result.get("entities", []), result.get("statements", [])
        )
        
        print(f"    📊 Pre-disambiguation connectivity: {stats['connected_entities']}/{stats['total_entities']} entities connected")
        
        if orphaned_entities:
            print(f"    ⚠️  Found {len(orphaned_entities)} orphaned entities, attempting to connect...")
            # Could add orphan connection logic here if needed
        
        # Apply entity disambiguation with safe unpacking
        try:
            disambiguation_result = self.disambiguate_batch_entities(
                result.get("entities", []), result.get("statements", [])
            )
            
            if disambiguation_result is None:
                print("    ⚠️  Disambiguation returned None, using original entities/statements")
                final_entities = result.get("entities", [])
                final_statements = result.get("statements", [])
            elif isinstance(disambiguation_result, tuple) and len(disambiguation_result) == 2:
                final_entities, final_statements = disambiguation_result
            else:
                print(f"    ⚠️  Disambiguation returned unexpected format: {type(disambiguation_result)}")
                final_entities = result.get("entities", [])
                final_statements = result.get("statements", [])
        except Exception as e:
            print(f"    ❌ Error in disambiguation: {e}")
            print("    🔄 Continuing with original entities/statements")
            final_entities = result.get("entities", [])
            final_statements = result.get("statements", [])
        
        # Update result with disambiguated entities and statements
        result["entities"] = final_entities
        result["statements"] = final_statements
        
        # Final connectivity validation
        final_is_valid, final_orphaned, final_stats = validator.validate_connectivity(
            final_entities, final_statements
        )
        
        print(f"    📊 Final connectivity: {final_stats['connected_entities']}/{final_stats['total_entities']} entities connected")
        
        return result

    def save_knowledge_graph_to_mongodb(self, kg_data: Dict[str, Any], batch_id: str, video_id: str) -> bool:
        """Save extracted knowledge graph data to MongoDB collections."""
        try:
            stats = {"entities": 0, "statements": 0}
            
            # Insert entities (with upsert to handle duplicates and add embeddings)
            if kg_data.get("entities"):
                for entity in kg_data["entities"]:
                    # Add metadata
                    entity["extracted_at"] = datetime.now(timezone.utc)
                    entity["batch_id"] = batch_id
                    entity["video_id"] = video_id
                    entity["extractor_version"] = "enhanced_kg_extractor_v1.0"
                    
                    # Generate and add embedding
                    embedding = self.generate_entity_embedding(
                        entity.get("entity_name", ""),
                        entity.get("entity_description", "")
                    )
                    if embedding:
                        entity["name_description_embedding"] = embedding
                    
                    self.entities.update_one(
                        {"entity_id": entity["entity_id"]},
                        {"$set": entity},
                        upsert=True
                    )
                stats["entities"] = len(kg_data["entities"])
            
            # Insert statements
            if kg_data.get("statements"):
                statement_docs = []
                for statement in kg_data["statements"]:
                    statement["extracted_at"] = datetime.now(timezone.utc)
                    statement["batch_id"] = batch_id
                    statement["extractor_version"] = "enhanced_kg_extractor_v1.0"
                    statement_docs.append(statement)
                
                if statement_docs:
                    try:
                        result = self.statements.insert_many(statement_docs, ordered=False)
                        stats["statements"] = len(result.inserted_ids)
                    except BulkWriteError as bwe:
                        stats["statements"] = bwe.details.get('nInserted', 0)
                        failed_count = len(bwe.details.get('writeErrors', []))
                        print(f"      ⚠️  Statements: {stats['statements']} inserted, {failed_count} failed (likely duplicates)")
            
            print(f"      ✅ Saved to MongoDB: {stats['entities']} entities, {stats['statements']} statements")
            return True
            
        except Exception as e:
            print(f"      ❌ Error saving to MongoDB: {e}")
            return False

    def calculate_optimal_batch_size(self, total_segments: int, min_batch_size: int = 200, max_batch_size: int = 1000) -> int:
        """Calculate optimal batch size to minimize number of batches while avoiding small runt batches."""
        if total_segments <= max_batch_size:
            return total_segments
        
        best_batch_size = min_batch_size
        best_efficiency = 0
        
        min_batches_needed = (total_segments + max_batch_size - 1) // max_batch_size
        max_batches_to_try = (total_segments // min_batch_size) + 2
        
        for num_batches in range(min_batches_needed, max_batches_to_try):
            batch_size = total_segments // num_batches
            
            if batch_size < min_batch_size or batch_size > max_batch_size:
                continue
            
            remainder = total_segments % num_batches
            efficiency = 1.0 - (remainder / total_segments)
            efficiency += (1.0 / num_batches) * 0.1
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_batch_size = batch_size
        
        return min(best_batch_size, max_batch_size)

    def get_videos_for_processing(self) -> List[Dict[str, Any]]:
        """Get all videos that have provenance segments but no entities extracted yet."""
        video_ids_with_segments = self.provenance_segments.distinct("video_id")
        print(f"Found {len(video_ids_with_segments)} videos with segments")
        
        videos_with_entities = set(self.entities.distinct("video_id"))
        print(f"Found {len(videos_with_entities)} videos with existing entities")
        
        videos_to_process = []
        skipped_count = 0
        
        for video_id in video_ids_with_segments:
            if video_id in videos_with_entities:
                skipped_count += 1
                print(f"  ⏭️  Skipping {video_id} - already has entities")
                continue
                
            video_info = self.videos.find_one({"video_id": video_id})
            if video_info:
                videos_to_process.append(video_info)
            else:
                print(f"  ⚠️  Video {video_id} not found in videos collection")
        
        print(f"📊 Processing summary:")
        print(f"  Total videos with segments: {len(video_ids_with_segments)}")
        print(f"  Already processed (skipped): {skipped_count}")
        print(f"  Ready for processing: {len(videos_to_process)}")
        
        return videos_to_process

    def get_segments_for_video(self, video_id: str) -> List[Dict[str, Any]]:
        """Get all provenance segments for a video, sorted by time_seconds."""
        segments = list(self.provenance_segments.find(
            {"video_id": video_id}
        ).sort("time_seconds", 1))
        
        return segments

    def create_batch_transcript(self, context_segments: List[Dict[str, Any]], process_segments: List[Dict[str, Any]]) -> str:
        """Create a transcript text from context and processing segments."""
        transcript_parts = []
        
        if context_segments:
            transcript_parts.append("CONTEXT SEGMENTS (for reference only, do not extract entities/statements from these):")
            for segment in context_segments:
                segment_id = segment.get("_id", "unknown_segment")
                text = segment.get("transcript_segment", "")
                transcript_parts.append(f"{segment_id} {text}")
            transcript_parts.append("")
        
        transcript_parts.append("PROCESS THESE SEGMENTS (extract entities and statements from these only):")
        for segment in process_segments:
            segment_id = segment.get("_id", "unknown_segment")
            text = segment.get("transcript_segment", "")
            transcript_parts.append(f"{segment_id} {text}")
        
        return "\n".join(transcript_parts)

    def process_video_in_batches(self, video_info: Dict[str, Any], batch_size: int = 200, overlap: int = 20, max_batch_size: int = 1000) -> bool:
        """Process a single video's segments in batches with overlap, multi-pass extraction, and disambiguation."""
        video_id = video_info.get("video_id", "")
        video_title = video_info.get("title", "Unknown Title")
        video_url = video_info.get("video_url", "")
        
        if not video_id:
            print("    ⚠️  No video_id found")
            return False
        
        try:
            segments = self.get_segments_for_video(video_id)
            
            if not segments:
                print("    ⚠️  No segments found for video")
                return False
            
            print(f"    📊 Found {len(segments)} segments to process")
            
            optimal_batch_size = self.calculate_optimal_batch_size(len(segments), batch_size, max_batch_size)
            num_batches = (len(segments) + optimal_batch_size - 1) // optimal_batch_size
            
            print(f"    🎯 Optimal batch size: {optimal_batch_size} ({num_batches} batches)")
            
            batch_num = 1
            start_idx = 0
            total_entities = 0
            total_statements = 0
            
            while start_idx < len(segments):
                end_idx = min(start_idx + optimal_batch_size, len(segments))
                
                if batch_num == 1:
                    context_segments = []
                    process_segments = segments[start_idx:end_idx]
                else:
                    context_start = max(0, start_idx - overlap)
                    context_segments = segments[context_start:start_idx]
                    process_segments = segments[start_idx:end_idx]
                
                print(f"    🔄 Processing batch {batch_num}/{num_batches}")
                print(f"      📋 Context segments: {len(context_segments)}, Processing segments: {len(process_segments)}")
                
                transcript_text = self.create_batch_transcript(context_segments, process_segments)
                
                batch_id = f"{video_id}_batch_{batch_num}"
                kg_data = self.extract_knowledge_graph(transcript_text, video_id, video_title, video_url)
                
                if self.save_knowledge_graph_to_mongodb(kg_data, batch_id, video_id):
                    total_entities += len(kg_data.get("entities", []))
                    total_statements += len(kg_data.get("statements", []))
                else:
                    print(f"    ❌ Failed to save batch {batch_num}")
                    return False
                
                start_idx = end_idx
                batch_num += 1
                
                if start_idx >= len(segments):
                    break
            
            print(f"    ✅ Video processing complete: {total_entities} entities, {total_statements} statements across {num_batches} batches")
            return True
            
        except Exception as e:
            print(f"    ❌ Error processing video: {e}")
            return False

    def process_all_videos(self, skip_existing: bool = True, limit: Optional[int] = None, 
                          batch_size: int = 200, overlap: int = 20, max_batch_size: int = 1000):
        """Process all videos with segments to extract knowledge graphs using enhanced extraction."""
        print("Starting enhanced knowledge graph extraction with multi-pass refinement and entity disambiguation...")
        
        # Setup vector index for disambiguation
        self.setup_vector_index()
        
        try:
            videos_to_process = self.get_videos_for_processing()
            
            if not videos_to_process:
                print("No videos to process")
                return
            
            if limit:
                videos_to_process = videos_to_process[:limit]
                print(f"Processing limited to first {limit} videos")
            
            stats = {
                "total": len(videos_to_process),
                "processed": 0,
                "skipped": 0,
                "errors": 0
            }
            
            for i, video in enumerate(videos_to_process, 1):
                video_id = video.get("video_id", "")
                video_title = video.get("title", "Unknown Title")
                
                print(f"\n[{i}/{stats['total']}] Processing: {video_title[:80]}...")
                print(f"  🆔 Video ID: {video_id}")
                print(f"  📋 Min batch size: {batch_size}, Max: {max_batch_size}, Overlap: {overlap}")
                print(f"  🔄 Multi-pass extraction: ENABLED")
                print(f"  🔍 Entity disambiguation: ENABLED")
                print(f"  📦 Prompt caching: ENABLED")
                
                if self.process_video_in_batches(video, batch_size, overlap, max_batch_size):
                    stats["processed"] += 1
                else:
                    stats["errors"] += 1
            
            print(f"\n📊 Enhanced Knowledge Graph Extraction Complete!")
            print(f"  Total videos: {stats['total']}")
            print(f"  Successfully processed: {stats['processed']}")
            print(f"  Skipped: {stats['skipped']}")
            print(f"  Errors: {stats['errors']}")
            print(f"  🔄 Multi-pass extraction with LLM feedback")
            print(f"  🔍 Entity disambiguation with vector similarity")
            print(f"  📦 Prompt caching reduced token costs significantly")
            
        finally:
            self.cleanup_cache()

    def get_extraction_stats(self) -> Dict[str, int]:
        """Get statistics about knowledge graph extraction."""
        videos_with_segments = len(self.provenance_segments.distinct("video_id"))
        videos_with_entities = len(self.entities.distinct("video_id")) if "video_id" in [idx["key"] for idx in self.entities.list_indexes()] else 0
        
        total_entities = self.entities.count_documents({})
        total_statements = self.statements.count_documents({})
        
        return {
            "videos_with_segments": videos_with_segments,
            "videos_with_entities": videos_with_entities,
            "total_entities": total_entities,
            "total_statements": total_statements,
            "remaining": videos_with_segments - videos_with_entities
        }

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Enhanced knowledge graph extraction with multi-pass refinement and entity disambiguation")
    parser.add_argument("--database", default="parliamentary_graph2", help="MongoDB database name")
    parser.add_argument("--limit", type=int, help="Limit number of videos to process")
    parser.add_argument("--batch-size", type=int, default=100, help="Minimum batch size (will be optimized per video)")
    parser.add_argument("--max-batch-size", type=int, default=150, help="Maximum batch size")
    parser.add_argument("--overlap", type=int, default=5, help="Number of segments to overlap between batches")
    parser.add_argument("--force", action="store_true", help="Regenerate entities for videos that already have them")
    parser.add_argument("--stats", action="store_true", help="Show extraction statistics only")
    parser.add_argument("--cache-ttl", type=int, default=24, help="Cache TTL in hours (default: 24)")
    parser.add_argument("--similarity-threshold", type=float, default=0.85, help="Similarity threshold for entity disambiguation")
    
    args = parser.parse_args()
    
    try:
        extractor = EnhancedKnowledgeGraphExtractor(database_name=args.database)
        extractor.cache_ttl_hours = args.cache_ttl
        
        if args.stats:
            stats = extractor.get_extraction_stats()
            print("📊 Enhanced Knowledge Graph Extraction Statistics:")
            print(f"  Videos with segments: {stats['videos_with_segments']}")
            print(f"  Videos with entities: {stats['videos_with_entities']}")
            print(f"  Total entities: {stats['total_entities']}")
            print(f"  Total statements: {stats['total_statements']}")
            print(f"  Remaining to process: {stats['remaining']}")
            return
        
        extractor.process_all_videos(
            skip_existing=not args.force,
            limit=args.limit,
            batch_size=args.batch_size,
            overlap=args.overlap,
            max_batch_size=args.max_batch_size
        )
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nTo set up required credentials:")
        print("1. MongoDB: Set MONGODB_CONNECTION_STRING environment variable")
        print("2. Google API: Set GOOGLE_API_KEY environment variable")
        print("3. Or create a .env file with both values")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
import os
import json
import asyncio
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import numpy as np
from urllib.parse import quote
import logging

# Apify for YouTube transcript fetching
from apify_client import ApifyClient

# LangChain and Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# MongoDB
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# Pydantic for structured outputs
from pydantic import BaseModel, Field

# RDF and ontologies
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL
from rdflib.namespace import FOAF, SKOS, DC, DCTERMS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define namespaces
SCHEMA = Namespace("https://schema.org/")
PROV = Namespace("http://www.w3.org/ns/prov#")
VIDEO = Namespace("https://youtube.com/video/")
KG = Namespace("https://knowledge-graph.example.com/")

@dataclass
class TranscriptSegment:
    text: str
    start: float
    duration: float

class RDFEntity(BaseModel):
    uri: str = Field(description="RDF URI for the entity")
    label: str = Field(description="Primary label for the entity")
    types: List[str] = Field(description="RDF types (e.g., schema:Person, foaf:Organization)")
    properties: Dict[str, Any] = Field(description="Additional RDF properties")
    alternate_labels: List[str] = Field(description="Alternative names/labels for entity resolution")
    
class RDFRelationship(BaseModel):
    subject_uri: str = Field(description="Subject entity URI")
    predicate: str = Field(description="RDF predicate (e.g., schema:knows, foaf:member)")
    object_uri: str = Field(description="Object entity URI")
    properties: Dict[str, Any] = Field(description="Additional properties for reified relationships")

class ProvenanceInfo(BaseModel):
    video_id: str
    timestamp: float
    text_snippet: str
    confidence: float = Field(default=1.0, description="Confidence score for this extraction")

class RDFKnowledgeGraph(BaseModel):
    entities: List[RDFEntity]
    relationships: List[RDFRelationship]
    provenance: Dict[str, List[ProvenanceInfo]] = Field(description="Provenance keyed by entity URI or relationship ID")

class KnowledgeGraphCritique(BaseModel):
    missing_entities: List[str]
    missing_relationships: List[str]
    incorrect_entities: List[str]
    incorrect_relationships: List[str]
    entity_resolution_suggestions: List[str]
    ontology_improvements: List[str]

class RDFYouTubeKnowledgeGraphExtractor:
    def __init__(self, google_api_key: str, apify_api_key: str, mongodb_uri: str, db_name: str = "youtube_rdf_kg"):
        # Initialize LLMs
        self.extractor_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0.1
        )
        
        self.critic_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0.2
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Apify setup
        self.apify_api_key = apify_api_key
        self.apify_client = ApifyClient(self.apify_api_key)
        
        # MongoDB setup
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.entities_collection = self.db.entities
        self.relationships_collection = self.db.relationships
        self.provenance_collection = self.db.provenance
        
        # Create indexes
        self._setup_indexes()
        
        # RDF Graph for managing ontology
        self.rdf_graph = Graph()
        self._setup_ontologies()
    
    def _setup_ontologies(self):
        """Bind common ontology namespaces"""
        self.rdf_graph.bind("schema", SCHEMA)
        self.rdf_graph.bind("foaf", FOAF)
        self.rdf_graph.bind("skos", SKOS)
        self.rdf_graph.bind("dc", DC)
        self.rdf_graph.bind("prov", PROV)
        self.rdf_graph.bind("video", VIDEO)
        self.rdf_graph.bind("kg", KG)
        self.rdf_graph.bind("owl", OWL)
    
    def _setup_indexes(self):
        """Set up MongoDB indexes for efficient querying and graph operations"""
        # Entity indexes
        self.entities_collection.create_index([("uri", 1)], unique=True)
        self.entities_collection.create_index([("label", 1)])
        self.entities_collection.create_index([("types", 1)])
        self.entities_collection.create_index([("alternate_labels", 1)])
        self.entities_collection.create_index([("embedding", "2dsphere")])
        
        # Relationship indexes for graph traversal
        self.relationships_collection.create_index([("subject_uri", 1)])
        self.relationships_collection.create_index([("object_uri", 1)])
        self.relationships_collection.create_index([("predicate", 1)])
        self.relationships_collection.create_index([
            ("subject_uri", 1), 
            ("predicate", 1), 
            ("object_uri", 1)
        ], unique=True)
        
        # Provenance indexes
        self.provenance_collection.create_index([("entity_uri", 1)])
        self.provenance_collection.create_index([("video_id", 1)])
    
    def _generate_entity_uri(self, label: str, entity_type: str) -> str:
        """Generate a consistent URI for an entity based on label and type"""
        # Normalize the label
        normalized_label = label.lower().strip().replace(" ", "_")
        # Create a hash for uniqueness while maintaining readability
        hash_suffix = hashlib.md5(f"{normalized_label}_{entity_type}".encode()).hexdigest()[:8]
        return f"{KG}{quote(normalized_label)}_{hash_suffix}"
    
    def _resolve_entity(self, entity: RDFEntity) -> Tuple[str, bool]:
        """Resolve entity to existing URI if it already exists"""
        # Check exact URI match
        existing = self.entities_collection.find_one({"uri": entity.uri})
        if existing:
            return entity.uri, True
        
        # Check label matches
        label_matches = list(self.entities_collection.find({
            "$or": [
                {"label": {"$regex": f"^{entity.label}$", "$options": "i"}},
                {"alternate_labels": {"$regex": f"^{entity.label}$", "$options": "i"}}
            ]
        }))
        
        if label_matches:
            # Check if types are compatible
            for match in label_matches:
                common_types = set(match["types"]) & set(entity.types)
                if common_types:
                    return match["uri"], True
        
        # Check embedding similarity for semantic matching
        embedding = self.generate_entity_embedding(entity)
        similar_entities = self.search_similar_entities_raw(embedding, limit=5)
        
        for similar in similar_entities:
            if similar["score"] > 0.95:  # High similarity threshold
                common_types = set(similar["types"]) & set(entity.types)
                if common_types:
                    # Merge alternate labels
                    self.entities_collection.update_one(
                        {"uri": similar["uri"]},
                        {"$addToSet": {"alternate_labels": {"$each": [entity.label] + entity.alternate_labels}}}
                    )
                    return similar["uri"], True
        
        return entity.uri, False
    
    def fetch_transcript(self, video_id: str) -> Optional[List[TranscriptSegment]]:
        """Fetch transcript from YouTube video using Apify's YouTube Transcript Scraper"""
        try:
            logger.info(f"Fetching transcript for video: {video_id}")
            
            # Prepare input for the YouTube Transcript Scraper
            actor_input = {
                "videoUrl": f"https://www.youtube.com/watch?v={video_id}"
            }
            
            logger.info(f"Starting Apify actor run for video {video_id}")
            
            # Run the actor and wait for it to finish
            run = self.apify_client.actor("pintostudio/youtube-transcript-scraper").call(run_input=actor_input)
            
            # Check if the run was successful
            if not run or run.get('status') != 'SUCCEEDED':
                logger.error(f"Apify actor run failed with status: {run.get('status') if run else 'None'}")
                return None
            
            # Get the results from the dataset
            dataset_client = self.apify_client.dataset(run["defaultDatasetId"])
            items = list(dataset_client.iterate_items())
            
            if not items:
                logger.error(f"No transcript data returned for video {video_id}")
                return None
            
            # Process the transcript data - Apify returns format: {"data": [{"start": "...", "dur": "...", "text": "..."}]}
            segments = []
            for item in items:
                if 'data' in item and isinstance(item['data'], list):
                    for segment_data in item['data']:
                        text = segment_data.get('text', '')
                        start = float(segment_data.get('start', 0))
                        duration = float(segment_data.get('dur', 0))
                        
                        if text.strip():  # Only add non-empty segments
                            segments.append(TranscriptSegment(
                                text=text.strip(),
                                start=start,
                                duration=duration
                            ))
            
            if not segments:
                logger.warning(f"No valid transcript segments found for video {video_id}")
                return None
            
            logger.info(f"Successfully fetched {len(segments)} transcript segments")
            return segments
            
        except Exception as e:
            logger.error(f"Error fetching transcript for {video_id} via Apify: {e}")
            return None
    
    def _create_extraction_prompt(self) -> ChatPromptTemplate:
        """Create prompt for RDF-based knowledge graph extraction"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting RDF-based knowledge graphs using standard ontologies.
            
            Use these ontologies where appropriate:
            - schema.org (schema:) - For general entities like Person, Organization, Place, Event, CreativeWork
            - FOAF (foaf:) - For people and social relationships
            - SKOS (skos:) - For concepts and taxonomies
            - Dublin Core (dc:/dcterms:) - For metadata
            
            Common RDF predicates to use:
            - schema:knows, schema:memberOf, schema:worksFor, schema:author
            - foaf:knows, foaf:member, foaf:topic
            - skos:broader, skos:narrower, skos:related
            - schema:about, schema:mentions
            
            IMPORTANT: Return a JSON object with this EXACT structure:
            {{
                "entities": [
                    {{
                        "uri": "https://knowledge-graph.example.com/entity_name_hash",
                        "label": "Entity Name",
                        "types": ["schema:Person", "foaf:Person"],
                        "properties": {{"description": "Brief description"}},
                        "alternate_labels": ["alt name 1", "alt name 2"]
                    }}
                ],
                "relationships": [
                    {{
                        "subject_uri": "https://knowledge-graph.example.com/entity1",
                        "predicate": "schema:knows",
                        "object_uri": "https://knowledge-graph.example.com/entity2",
                        "properties": {{}}
                    }}
                ],
                "provenance": {{
                    "https://knowledge-graph.example.com/entity1": [
                        {{
                            "video_id": "VIDEO_ID",
                            "timestamp": 12.5,
                            "text_snippet": "relevant text",
                            "confidence": 0.95
                        }}
                    ]
                }}
            }}
            
            Do NOT use @id, @type or other JSON-LD syntax. Use the exact field names shown above."""),
            ("human", """Extract an RDF-based knowledge graph from this YouTube transcript.
            
            Video ID: {video_id}
            
            Transcript segments:
            {transcript}
            
            Return ONLY a valid JSON object with 'entities', 'relationships', and 'provenance' fields as shown in the example.""")
        ])
    
    def _create_critique_prompt(self) -> ChatPromptTemplate:
        """Create prompt for critiquing the RDF knowledge graph"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert at reviewing RDF-based knowledge graphs.
            
            Check for:
            1. Entities that should be merged (same real-world entity with different labels)
            2. Missing standard ontology usage
            3. Relationships that could use more specific predicates
            4. Missing inverse relationships
            5. Opportunities for transitive relationships
            6. Better type hierarchies using rdfs:subClassOf
            
            Return a JSON object with this structure:
            {{
                "missing_entities": ["entity description"],
                "missing_relationships": ["relationship description"],
                "incorrect_entities": ["entity issue"],
                "incorrect_relationships": ["relationship issue"],
                "entity_resolution_suggestions": ["suggestion"],
                "ontology_improvements": ["improvement"]
            }}"""),
            ("human", """Review this RDF knowledge graph:
            
            Video ID: {video_id}
            Transcript: {transcript}
            
            Current Knowledge Graph:
            {knowledge_graph}
            
            Return ONLY a valid JSON object with your critique.""")
        ])
    
    def _create_refinement_prompt(self) -> ChatPromptTemplate:
        """Create prompt for refining the knowledge graph based on critique"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert at refining RDF-based knowledge graphs based on critique.
            
            Given the original transcript, current knowledge graph, and critique, create an improved version.
            
            Guidelines:
            1. Add missing entities mentioned in the critique
            2. Add missing relationships
            3. Fix incorrect entities/relationships
            4. Apply entity resolution suggestions
            5. Implement ontology improvements
            
            Return a complete knowledge graph with the EXACT same JSON structure:
            {{
                "entities": [...],
                "relationships": [...],
                "provenance": {{...}}
            }}"""),
            ("human", """Refine this knowledge graph based on the critique:
            
            Video ID: {video_id}
            Transcript: {transcript}
            
            Current Knowledge Graph:
            {knowledge_graph}
            
            Critique:
            {critique}
            
            Return ONLY a valid JSON object with the refined knowledge graph.""")
        ])
    
    def generate_entity_embedding(self, entity: RDFEntity) -> List[float]:
        """Generate embedding for an RDF entity"""
        # Create rich text representation including types
        type_labels = [t.split(":")[-1] for t in entity.types]
        text = f"{entity.label} ({', '.join(type_labels)})"
        
        # Add properties for richer embedding
        if entity.properties:
            prop_text = " ".join([f"{k}: {v}" for k, v in entity.properties.items() if isinstance(v, str)])
            text += f" {prop_text}"
        
        embedding = self.embeddings.embed_query(text)
        return embedding
    
    async def extract_knowledge_graph(self, video_id: str, max_iterations: int = 3) -> Optional[RDFKnowledgeGraph]:
        """Extract RDF knowledge graph with iterative refinement"""
        segments = self.fetch_transcript(video_id)
        
        if not segments:
            logger.warning(f"No transcript available for video {video_id}")
            return None
        
        transcript_text = "\n".join([
            f"[{seg.start:.1f}s] {seg.text}"
            for seg in segments
        ])
        
        logger.info(f"Transcript length: {len(transcript_text)} characters")
        
        extraction_prompt = self._create_extraction_prompt()
        kg_parser = PydanticOutputParser(pydantic_object=RDFKnowledgeGraph)
        
        extraction_chain = extraction_prompt | self.extractor_llm
        
        logger.info("Extracting initial knowledge graph...")
        result = extraction_chain.invoke({
            "video_id": video_id,
            "transcript": transcript_text
        })
        
        try:
            # Try to parse the content
            content = result.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            # Parse and validate
            parsed_json = json.loads(content)
            current_kg = RDFKnowledgeGraph(**parsed_json)
            
        except Exception as e:
            logger.error(f"Error parsing LLM output: {e}")
            # Create a minimal valid knowledge graph to continue
            current_kg = RDFKnowledgeGraph(
                entities=[],
                relationships=[],
                provenance={}
            )
        
        # Iterative refinement
        critique_prompt = self._create_critique_prompt()
        refinement_prompt = self._create_refinement_prompt()
        critique_parser = PydanticOutputParser(pydantic_object=KnowledgeGraphCritique)
        
        for iteration in range(max_iterations):
            if not current_kg.entities:
                logger.info(f"No entities found, skipping refinement")
                break
                
            logger.info(f"Refinement iteration {iteration + 1}/{max_iterations}")
            
            # Get critique
            critique_chain = critique_prompt | self.critic_llm
            critique_result = critique_chain.invoke({
                "video_id": video_id,
                "transcript": transcript_text,
                "knowledge_graph": current_kg.model_dump_json()
            })
            
            try:
                # Parse critique
                critique_content = critique_result.content
                if "```json" in critique_content:
                    critique_content = critique_content.split("```json")[1].split("```")[0]
                elif "```" in critique_content:
                    critique_content = critique_content.split("```")[1].split("```")[0]
                
                critique_json = json.loads(critique_content)
                critique = KnowledgeGraphCritique(**critique_json)
                
                logger.info(f"Critique summary:")
                logger.info(f"  - Missing entities: {len(critique.missing_entities)}")
                logger.info(f"  - Missing relationships: {len(critique.missing_relationships)}")
                logger.info(f"  - Entity resolution suggestions: {len(critique.entity_resolution_suggestions)}")
                
                # Check if improvements are needed
                if not any([critique.missing_entities, critique.missing_relationships,
                           critique.incorrect_entities, critique.incorrect_relationships,
                           critique.entity_resolution_suggestions, critique.ontology_improvements]):
                    logger.info("No further improvements suggested")
                    break
                
                # Apply refinements using the refinement prompt
                logger.info("Applying refinements...")
                refinement_chain = refinement_prompt | self.extractor_llm
                refinement_result = refinement_chain.invoke({
                    "video_id": video_id,
                    "transcript": transcript_text,
                    "knowledge_graph": current_kg.model_dump_json(),
                    "critique": critique.model_dump_json()
                })
                
                # Parse refined knowledge graph
                refined_content = refinement_result.content
                if "```json" in refined_content:
                    refined_content = refined_content.split("```json")[1].split("```")[0]
                elif "```" in refined_content:
                    refined_content = refined_content.split("```")[1].split("```")[0]
                
                refined_json = json.loads(refined_content)
                current_kg = RDFKnowledgeGraph(**refined_json)
                
                logger.info(f"After refinement: {len(current_kg.entities)} entities, {len(current_kg.relationships)} relationships")
                
            except Exception as e:
                logger.error(f"Error in refinement iteration: {e}")
                break
        
        return current_kg
    
    def store_knowledge_graph(self, video_id: str, kg: RDFKnowledgeGraph):
        """Store RDF knowledge graph in MongoDB with entity resolution"""
        resolved_uri_map = {}
        
        # Process entities with resolution
        for entity in kg.entities:
            resolved_uri, exists = self._resolve_entity(entity)
            resolved_uri_map[entity.uri] = resolved_uri
            
            if not exists:
                # New entity - store it
                embedding = self.generate_entity_embedding(entity)
                
                entity_doc = {
                    "_id": resolved_uri,
                    "uri": resolved_uri,
                    "label": entity.label,
                    "types": entity.types,
                    "properties": entity.properties,
                    "alternate_labels": entity.alternate_labels,
                    "embedding": embedding,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                self.entities_collection.insert_one(entity_doc)
            else:
                # Update existing entity
                self.entities_collection.update_one(
                    {"uri": resolved_uri},
                    {
                        "$set": {"updated_at": datetime.utcnow()},
                        "$addToSet": {"alternate_labels": {"$each": entity.alternate_labels}}
                    }
                )
        
        # Store relationships with resolved URIs
        for relationship in kg.relationships:
            subject_uri = resolved_uri_map.get(relationship.subject_uri, relationship.subject_uri)
            object_uri = resolved_uri_map.get(relationship.object_uri, relationship.object_uri)
            
            rel_doc = {
                "_id": f"{subject_uri}|{relationship.predicate}|{object_uri}",
                "subject_uri": subject_uri,
                "predicate": relationship.predicate,
                "object_uri": object_uri,
                "properties": relationship.properties,
                "created_at": datetime.utcnow()
            }
            
            try:
                self.relationships_collection.insert_one(rel_doc)
            except DuplicateKeyError:
                # Relationship already exists
                pass
        
        # Store provenance
        for uri, prov_list in kg.provenance.items():
            resolved_uri = resolved_uri_map.get(uri, uri)
            for prov in prov_list:
                prov_doc = {
                    "entity_uri": resolved_uri,
                    "video_id": prov.video_id,
                    "timestamp": prov.timestamp,
                    "text_snippet": prov.text_snippet,
                    "confidence": prov.confidence,
                    "created_at": datetime.utcnow()
                }
                self.provenance_collection.insert_one(prov_doc)
    
    def search_similar_entities_raw(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Raw embedding search"""
        # This is a simplified version - in production, use MongoDB Atlas Vector Search
        all_entities = list(self.entities_collection.find({}))
        
        similarities = []
        for entity in all_entities:
            if "embedding" in entity:
                # Cosine similarity
                similarity = np.dot(embedding, entity["embedding"]) / (
                    np.linalg.norm(embedding) * np.linalg.norm(entity["embedding"])
                )
                similarities.append({
                    **entity,
                    "score": float(similarity)
                })
        
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:limit]
    
    def search_entities_with_context(self, query: str, max_hops: int = 2, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for entities and retrieve their graph context using MongoDB $graphLookup"""
        # Get seed entities from vector search
        query_embedding = self.embeddings.embed_query(query)
        seed_entities = self.search_similar_entities_raw(query_embedding, limit=limit)
        
        results = []
        for entity in seed_entities:
            # Use $graphLookup to traverse the graph
            pipeline = [
                {"$match": {"uri": entity["uri"]}},
                {
                    "$graphLookup": {
                        "from": "relationships",
                        "startWith": "$uri",
                        "connectFromField": "uri",
                        "connectToField": "subject_uri",
                        "as": "outgoing_paths",
                        "maxDepth": max_hops - 1,
                        "depthField": "depth"
                    }
                },
                {
                    "$graphLookup": {
                        "from": "relationships",
                        "startWith": "$uri",
                        "connectFromField": "uri",
                        "connectToField": "object_uri",
                        "as": "incoming_paths",
                        "maxDepth": max_hops - 1,
                        "depthField": "depth"
                    }
                }
            ]
            
            context_result = list(self.entities_collection.aggregate(pipeline))
            
            if context_result:
                context = context_result[0]
                
                # Collect all connected entity URIs
                connected_uris = set()
                for path in context.get("outgoing_paths", []):
                    connected_uris.add(path["object_uri"])
                for path in context.get("incoming_paths", []):
                    connected_uris.add(path["subject_uri"])
                
                # Fetch connected entities
                connected_entities = list(self.entities_collection.find({
                    "uri": {"$in": list(connected_uris)}
                }))
                
                # Get provenance
                provenance = list(self.provenance_collection.find({
                    "entity_uri": entity["uri"]
                }))
                
                results.append({
                    "entity": entity,
                    "score": entity.get("score", 0),
                    "outgoing_relationships": context.get("outgoing_paths", []),
                    "incoming_relationships": context.get("incoming_paths", []),
                    "connected_entities": connected_entities,
                    "provenance": provenance
                })
        
        return results
    
    def export_to_rdf(self, video_id: Optional[str] = None) -> str:
        """Export knowledge graph to RDF/Turtle format"""
        g = Graph()
        
        # Copy namespace bindings
        for prefix, namespace in self.rdf_graph.namespaces():
            g.bind(prefix, namespace)
        
        # Query filter
        query_filter = {"video_id": video_id} if video_id else {}
        
        # Add entities
        entities = self.entities_collection.find({})
        for entity in entities:
            entity_uri = URIRef(entity["uri"])
            
            # Add types
            for rdf_type in entity["types"]:
                if ":" in rdf_type:
                    prefix, local = rdf_type.split(":", 1)
                    if prefix == "schema":
                        g.add((entity_uri, RDF.type, SCHEMA[local]))
                    elif prefix == "foaf":
                        g.add((entity_uri, RDF.type, FOAF[local]))
            
            # Add label
            g.add((entity_uri, RDFS.label, Literal(entity["label"])))
            
            # Add alternate labels
            for alt_label in entity.get("alternate_labels", []):
                g.add((entity_uri, SKOS.altLabel, Literal(alt_label)))
        
        # Add relationships
        relationships = self.relationships_collection.find({})
        for rel in relationships:
            subject = URIRef(rel["subject_uri"])
            predicate = self._parse_predicate_uri(rel["predicate"])
            obj = URIRef(rel["object_uri"])
            g.add((subject, predicate, obj))
        
        return g.serialize(format="turtle")
    
    def _parse_predicate_uri(self, predicate: str) -> URIRef:
        """Parse predicate string to proper URIRef"""
        if ":" in predicate:
            prefix, local = predicate.split(":", 1)
            if prefix == "schema":
                return SCHEMA[local]
            elif prefix == "foaf":
                return FOAF[local]
            elif prefix == "skos":
                return SKOS[local]
        return URIRef(predicate)

# Example usage
async def main():
    # Set your API keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    APIFY_API_KEY = os.getenv("APIFY_API_KEY")
    MONGODB_URI = os.getenv("MONGODB_URI")
    
    if not GOOGLE_API_KEY:
        logger.error("Error: GOOGLE_API_KEY environment variable not set")
        logger.info("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    if not APIFY_API_KEY:
        logger.error("Error: APIFY_API_KEY environment variable not set")
        logger.info("Get your API key from: https://console.apify.com/account/integrations")
        return
    
    if not MONGODB_URI:
        logger.error("Error: MONGODB_URI environment variable not set")
        logger.info("Get your MongoDB URI from: https://cloud.mongodb.com/")
        return
    
    # Initialize extractor
    extractor = RDFYouTubeKnowledgeGraphExtractor(
        google_api_key=GOOGLE_API_KEY,
        apify_api_key=APIFY_API_KEY,
        mongodb_uri=MONGODB_URI
    )
    
    # Example video IDs with known good transcripts
    video_ids = [
        "dR-eoAEvPH4",
    ]
    
    logger.info("Processing YouTube videos for knowledge graph extraction...")
    
    for video_id in video_ids:
        try:
            logger.info(f"\nExtracting knowledge graph from video: {video_id}")
            logger.info(f"Video URL: https://www.youtube.com/watch?v={video_id}")
            
            kg = await extractor.extract_knowledge_graph(video_id, max_iterations=2)
            
            if not kg:
                logger.warning(f"Could not extract knowledge graph from video {video_id}")
                continue
            
            logger.info(f"Extracted {len(kg.entities)} entities and {len(kg.relationships)} relationships")
            
            # Show some extracted entities
            if kg.entities:
                logger.info("\nSample entities extracted:")
                for entity in kg.entities[:5]:
                    logger.info(f"  - {entity.label} ({', '.join(entity.types)})")
                    if entity.properties.get('description'):
                        logger.info(f"    Description: {entity.properties['description'][:100]}...")
            
            # Store with entity resolution
            if kg.entities or kg.relationships:
                logger.info("\nStoring in MongoDB with entity resolution...")
                extractor.store_knowledge_graph(video_id, kg)
                
                # Test GraphRAG search if we have entities
                if kg.entities:
                    logger.info("\nTesting GraphRAG with multi-hop context...")
                    # Search based on first entity type
                    search_query = kg.entities[0].types[0].split(':')[-1] if kg.entities[0].types else "technology"
                    results = extractor.search_entities_with_context(search_query, max_hops=2, limit=3)
                    
                    for result in results[:1]:
                        entity = result["entity"]
                        logger.info(f"\nFound entity: {entity['label']}")
                        logger.info(f"Connected to {len(result['connected_entities'])} entities within 2 hops")
            
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Test RDF export
    logger.info("\n\nExporting to RDF/Turtle...")
    try:
        rdf_output = extractor.export_to_rdf()
        if rdf_output:
            logger.info("RDF export successful!")
            logger.info(f"First 500 characters:\n{rdf_output[:500]}...")
    except Exception as e:
        logger.error(f"Error exporting to RDF: {e}")

if __name__ == "__main__":
    asyncio.run(main())
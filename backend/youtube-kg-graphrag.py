import os
import json
import asyncio
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
from urllib.parse import quote

# YouTube transcript fetching
from youtube_transcript_api import YouTubeTranscriptApi

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
    def __init__(self, google_api_key: str, mongodb_uri: str, db_name: str = "youtube_rdf_kg"):
        # Initialize LLMs
        self.extractor_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0.1
        )
        
        self.critic_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0.2
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
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
    
    def fetch_transcript(self, video_id: str) -> List[TranscriptSegment]:
        """Fetch transcript from YouTube video"""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            segments = [
                TranscriptSegment(
                    text=item['text'],
                    start=item['start'],
                    duration=item['duration']
                )
                for item in transcript_list
            ]
            return segments
        except Exception as e:
            print(f"Error fetching transcript: {e}")
            raise
    
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
            
            For each entity:
            1. Generate a unique URI
            2. Assign appropriate RDF types
            3. Include multiple labels if the entity is referred to differently
            4. Add relevant properties
            
            For relationships:
            1. Use standard predicates when possible
            2. Include properties for qualified relationships
            
            Track provenance for everything with video_id, timestamp, and text_snippet."""),
            ("human", """Extract an RDF-based knowledge graph from this YouTube transcript.
            
            Video ID: {video_id}
            
            Transcript segments:
            {transcript}
            
            Return a JSON with 'entities', 'relationships', and 'provenance' following the RDF model.""")
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
            6. Better type hierarchies using rdfs:subClassOf"""),
            ("human", """Review this RDF knowledge graph:
            
            Video ID: {video_id}
            Transcript: {transcript}
            
            Current Knowledge Graph:
            {knowledge_graph}
            
            Suggest improvements for entity resolution and ontology usage.""")
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
    
    async def extract_knowledge_graph(self, video_id: str, max_iterations: int = 3) -> RDFKnowledgeGraph:
        """Extract RDF knowledge graph with iterative refinement"""
        segments = self.fetch_transcript(video_id)
        
        transcript_text = "\n".join([
            f"[{seg.start:.1f}s] {seg.text}"
            for seg in segments
        ])
        
        extraction_prompt = self._create_extraction_prompt()
        kg_parser = PydanticOutputParser(pydantic_object=RDFKnowledgeGraph)
        
        extraction_chain = extraction_prompt | self.extractor_llm
        
        result = extraction_chain.invoke({
            "video_id": video_id,
            "transcript": transcript_text
        })
        
        current_kg = kg_parser.parse(result.content)
        
        # Iterative refinement
        critique_prompt = self._create_critique_prompt()
        critique_parser = PydanticOutputParser(pydantic_object=KnowledgeGraphCritique)
        
        for iteration in range(max_iterations):
            print(f"Refinement iteration {iteration + 1}/{max_iterations}")
            
            critique_chain = critique_prompt | self.critic_llm
            critique_result = critique_chain.invoke({
                "video_id": video_id,
                "transcript": transcript_text,
                "knowledge_graph": current_kg.json()
            })
            
            critique = critique_parser.parse(critique_result.content)
            
            if not any([critique.missing_entities, critique.missing_relationships,
                       critique.incorrect_entities, critique.incorrect_relationships,
                       critique.entity_resolution_suggestions]):
                print("No further improvements suggested")
                break
            
            # Apply refinements (similar to before but with RDF focus)
            # ... refinement logic ...
        
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
        # For now, we'll do a basic implementation
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
    MONGODB_URI = os.getenv("MONGODB_URI")
    
    # Initialize extractor
    extractor = RDFYouTubeKnowledgeGraphExtractor(
        google_api_key=GOOGLE_API_KEY,
        mongodb_uri=MONGODB_URI
    )
    
    # Process multiple videos
    video_ids = ["video_id_1", "video_id_2"]  # Replace with actual video IDs
    
    for video_id in video_ids:
        try:
            print(f"\nExtracting knowledge graph from video: {video_id}")
            kg = await extractor.extract_knowledge_graph(video_id, max_iterations=3)
            
            print(f"Extracted {len(kg.entities)} entities and {len(kg.relationships)} relationships")
            
            # Store with entity resolution
            print("Storing in MongoDB with entity resolution...")
            extractor.store_knowledge_graph(video_id, kg)
            
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
    
    # GraphRAG search with multi-hop context
    print("\n\nTesting GraphRAG with multi-hop context...")
    results = extractor.search_entities_with_context("ocean ecosystems", max_hops=2, limit=3)
    
    for result in results:
        entity = result["entity"]
        print(f"\n{'='*60}")
        print(f"Entity: {entity['label']} ({', '.join(entity['types'])})")
        print(f"Score: {result['score']:.3f}")
        print(f"Connected to {len(result['connected_entities'])} entities within 2 hops")
        
        # Show some relationships
        print("\nSample relationships:")
        for rel in result["outgoing_relationships"][:3]:
            print(f"  -> {rel['predicate']} -> {rel['object_uri']}")
        
        # Show provenance
        print("\nProvenance:")
        for prov in result["provenance"][:2]:
            print(f"  Video: {prov['video_id']} at {prov['timestamp']:.1f}s")
    
    # Export to RDF
    print("\n\nExporting to RDF/Turtle...")
    rdf_output = extractor.export_to_rdf()
    print(rdf_output[:500] + "...")  # Show first 500 chars

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
RDF to MongoDB Graph Loader with Vector Search Support

This script loads RDF Turtle files and saves them as graph structures in MongoDB Atlas
for GraphRAG-style querying and analysis, with support for vector embeddings.

Requirements:
- pymongo
- rdflib
- python-dotenv (optional, for environment variables)
- sentence-transformers (for generating embeddings)

Usage:
    python graph_loader.py <rdf_file.ttl> [--skip-embeddings]
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import hashlib
from datetime import datetime, timezone
import re

try:
    from pymongo import MongoClient, ASCENDING, TEXT
    from pymongo.errors import ConnectionFailure, DuplicateKeyError
    from rdflib import Graph, URIRef, Literal, BNode
    from rdflib.namespace import RDF, RDFS, OWL, FOAF, XSD
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install pymongo rdflib python-dotenv")
    sys.exit(1)

# Optional: Import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  sentence-transformers not available. Install with: pip install sentence-transformers")

# Load environment variables
load_dotenv()

class RDFToMongoLoader:
    def __init__(self, connection_string: str = None, database_name: str = "parliamentary_graph", 
                 use_embeddings: bool = True):
        """
        Initialize the RDF to MongoDB loader.
        
        Args:
            connection_string: MongoDB Atlas connection string. If None, will try to get from environment.
            database_name: Name of the MongoDB database to use
            use_embeddings: Whether to generate vector embeddings for text content
        """
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
            print("✅ Successfully connected to MongoDB Atlas")
        except ConnectionFailure as e:
            raise ConnectionFailure(f"Failed to connect to MongoDB: {e}")
        
        self.db = self.client[database_name]
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE
        
        # Initialize embedding model if available
        if self.use_embeddings:
            try:
                print("🔄 Loading embedding model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Embedding model loaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to load embedding model: {e}")
                self.use_embeddings = False
        
        self.setup_collections()
    
    def setup_collections(self):
        """Set up MongoDB collections with appropriate indexes."""
        
        # Nodes collection for entities
        self.nodes = self.db.nodes
        try:
            self.nodes.create_index([("uri", ASCENDING)], unique=True)
        except:
            pass  # Index already exists
        try:
            self.nodes.create_index([("type", ASCENDING)])
        except:
            pass
        try:
            self.nodes.create_index([("label", TEXT), ("searchable_text", TEXT)])
        except:
            pass
        try:
            self.nodes.create_index([("source_video", ASCENDING)])
        except:
            pass
        
        # Edges collection for relationships
        self.edges = self.db.edges
        try:
            self.edges.create_index([("subject", ASCENDING)])
        except:
            pass
        try:
            self.edges.create_index([("predicate", ASCENDING)])
        except:
            pass
        try:
            self.edges.create_index([("object", ASCENDING)])
        except:
            pass
        try:
            self.edges.create_index([("source_video", ASCENDING)])
        except:
            pass
        try:
            self.edges.create_index([("subject", ASCENDING), ("predicate", ASCENDING), ("object", ASCENDING)], unique=True)
        except:
            pass
        
        # Statements collection for reified statements (provenance)
        self.statements = self.db.statements
        
        # Drop and recreate the problematic index
        try:
            self.statements.drop_index("statement_id_1")
        except:
            pass  # Index doesn't exist
        
        try:
            self.statements.create_index([("global_statement_id", ASCENDING)], unique=True)
        except:
            pass
        try:
            self.statements.create_index([("statement_id", ASCENDING)])
        except:
            pass
        try:
            self.statements.create_index([("subject", ASCENDING)])
        except:
            pass
        try:
            self.statements.create_index([("predicate", ASCENDING)])
        except:
            pass
        try:
            self.statements.create_index([("object", ASCENDING)])
        except:
            pass
        try:
            self.statements.create_index([("source_video", ASCENDING)])
        except:
            pass
        try:
            self.statements.create_index([("start_offset", ASCENDING)])
        except:
            pass
        try:
            self.statements.create_index([("end_offset", ASCENDING)])
        except:
            pass
        
        # Videos collection for source metadata
        self.videos = self.db.videos
        try:
            self.videos.create_index([("video_id", ASCENDING)], unique=True)
        except:
            pass
        
        print("✅ Collections and indexes set up successfully")
        
        if self.use_embeddings:
            print("ℹ️  Vector search indexes need to be created manually in Atlas:")
            print("   1. Go to your Atlas cluster")
            print("   2. Navigate to Search > Create Search Index")
            print("   3. Choose 'Vector Search' and use the nodes collection")
            print("   4. Use this configuration:")
            print("""   {
     "fields": [
       {
         "type": "vector",
         "path": "embedding",
         "numDimensions": 384,
         "similarity": "cosine"
       }
     ]
   }""")
    
    def extract_video_id_from_ttl(self, ttl_file: str) -> str:
        """Extract video ID from TTL filename."""
        return Path(ttl_file).stem
    
    def uri_to_string(self, uri) -> str:
        """Convert RDFLib URI/Literal to string."""
        if isinstance(uri, URIRef):
            return str(uri)
        elif isinstance(uri, Literal):
            return str(uri)
        elif isinstance(uri, BNode):
            return f"_:{uri}"
        else:
            return str(uri)
    
    def extract_local_name(self, uri_str: str) -> str:
        """Extract local name from URI for labeling."""
        if '#' in uri_str:
            return uri_str.split('#')[-1]
        elif '/' in uri_str:
            return uri_str.split('/')[-1]
        else:
            return uri_str
    
    def extract_label_from_properties(self, properties: Dict[str, Any]) -> Optional[str]:
        """
        Extract the best label from properties, checking multiple possible fields.
        Priority order: schema:name, rdfs:label, foaf:name, dcterms:title, schema:title
        """
        # Define common label/name/title properties in priority order
        label_properties = [
            "http://schema.org/name",           # schema:name
            str(RDFS.label),                    # rdfs:label  
            str(FOAF.name),                     # foaf:name
            "http://purl.org/dc/terms/title",   # dcterms:title
            "http://schema.org/title",          # schema:title
            "http://www.w3.org/2004/02/skos/core#prefLabel",  # skos:prefLabel
        ]
        
        for prop_uri in label_properties:
            if prop_uri in properties:
                value = properties[prop_uri]
                # Handle both single values and lists
                if isinstance(value, list):
                    return str(value[0]) if value else None
                else:
                    return str(value)
        
        return None
    
    def create_searchable_text(self, uri_str: str, label: str, properties: Dict[str, Any], node_types: List[str]) -> str:
        """
        Create a comprehensive searchable text field from all node information.
        """
        text_parts = []
        
        # Add label/name (highest priority)
        if label:
            text_parts.append(label)
        
        # Add local name from URI (if different from label)
        local_name = self.extract_local_name(uri_str)
        if local_name and local_name != label:
            # Convert camelCase to readable text
            readable_local = re.sub(r'([a-z])([A-Z])', r'\1 \2', local_name)
            text_parts.append(readable_local)
        
        # Add all name/title/label properties from the RDF
        name_properties = [
            "http://schema.org/name",
            "http://schema.org/title", 
            "http://www.w3.org/2000/01/rdf-schema#label",
            "http://xmlns.com/foaf/0.1/name",
            "http://purl.org/dc/terms/title",
            "http://www.w3.org/2004/02/skos/core#prefLabel"
        ]
        
        for prop_uri in name_properties:
            if prop_uri in properties:
                value = properties[prop_uri]
                if isinstance(value, list):
                    for v in value:
                        if v and str(v).strip() and str(v) not in text_parts:
                            text_parts.append(str(v).strip())
                else:
                    if value and str(value).strip() and str(value) not in text_parts:
                        text_parts.append(str(value).strip())
        
        # Add descriptive properties
        descriptive_properties = [
            "http://schema.org/description",
            "http://purl.org/dc/terms/description", 
            "http://www.w3.org/2000/01/rdf-schema#comment",
            "http://example.com/barbados-parliament-ontology#hasRole",
        ]
        
        for prop_uri in descriptive_properties:
            if prop_uri in properties:
                value = properties[prop_uri]
                if isinstance(value, list):
                    for v in value:
                        if v and str(v).strip():
                            text_parts.append(str(v).strip())
                else:
                    if value and str(value).strip():
                        text_parts.append(str(value).strip())
        
        # Add readable type names (but not generic ones)
        for node_type in node_types:
            type_name = self.extract_local_name(node_type)
            if type_name and type_name not in ['Entity', 'Thing']:
                # Convert camelCase to readable text
                readable_type = re.sub(r'([a-z])([A-Z])', r'\1 \2', type_name)
                text_parts.append(readable_type)
        
        # Clean up and deduplicate
        clean_parts = []
        seen = set()
        
        for part in text_parts:
            if part and isinstance(part, str):
                # Basic cleaning
                clean_part = part.strip()
                
                # Skip very short or generic parts
                if len(clean_part) < 2 or clean_part.lower() in ['entity', 'thing', 'object']:
                    continue
                
                # Convert to lowercase for deduplication check
                clean_lower = clean_part.lower()
                if clean_lower not in seen:
                    seen.add(clean_lower)
                    clean_parts.append(clean_part)
        
        return ' '.join(clean_parts)
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate vector embedding for text content."""
        if not self.use_embeddings or not text:
            return None
        
        try:
            # Clean text for embedding
            clean_text = re.sub(r'\s+', ' ', text).strip()
            if len(clean_text) < 3:  # Skip very short text
                return None
            
            embedding = self.embedding_model.encode(clean_text)
            return embedding.tolist()
        except Exception as e:
            print(f"⚠️  Failed to generate embedding: {e}")
            return None
    
    def determine_node_type(self, uri: URIRef, graph: Graph) -> List[str]:
        """Determine the type(s) of a node from the RDF graph."""
        types = []
        for triple in graph.triples((uri, RDF.type, None)):
            types.append(str(triple[2]))
        
        # If no explicit type, infer from URI patterns
        if not types:
            uri_str = str(uri)
            if 'Person' in uri_str or 'foaf' in uri_str:
                types.append(str(FOAF.Person))
            elif 'Concept' in uri_str:
                types.append("http://example.com/barbados-parliament-ontology#Concept")
            elif 'Statement' in uri_str:
                types.append(str(RDF.Statement))
            else:
                types.append("http://example.org/Entity")
        
        return types
    
    def extract_properties(self, uri: URIRef, graph: Graph) -> Dict[str, Any]:
        """Extract all properties for a given URI."""
        properties = {}
        
        # Get all triples for this URI
        all_triples = list(graph.triples((uri, None, None)))
        
        for triple in all_triples:
            pred_str = str(triple[1])
            obj_str = self.uri_to_string(triple[2])
            
            # Skip rdf:type as it's handled separately
            if pred_str == str(RDF.type):
                continue
            
            # Group multiple values for the same property
            if pred_str not in properties:
                properties[pred_str] = []
            properties[pred_str].append(obj_str)
        
        # Convert single-item lists to single values
        for key, value in properties.items():
            if len(value) == 1:
                properties[key] = value[0]
        
        return properties
    
    def create_statement_id(self, subject: str, predicate: str, object_val: str) -> str:
        """Create a unique ID for a statement."""
        content = f"{subject}|{predicate}|{object_val}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def load_rdf_file(self, ttl_file: str) -> Graph:
        """Load and parse RDF Turtle file."""
        print(f"Loading RDF file: {ttl_file}")
        
        try:
            graph = Graph()
            # Suppress conversion warnings for malformed literals more comprehensively
            import warnings
            import logging
            
            # Suppress both warnings and logging messages
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Also suppress rdflib logger
                rdflib_logger = logging.getLogger('rdflib')
                original_level = rdflib_logger.level
                rdflib_logger.setLevel(logging.ERROR)
                
                try:
                    graph.parse(ttl_file, format='turtle')
                finally:
                    # Restore original logging level
                    rdflib_logger.setLevel(original_level)
                    
            print(f"✅ Loaded {len(graph)} triples from RDF file")
            return graph
        except Exception as e:
            raise Exception(f"Error loading RDF file: {e}")
    
    def process_nodes(self, graph: Graph, video_id: str):
        """Extract and save all nodes from the RDF graph."""
        print("Processing nodes...")
        
        # Get all unique subjects and objects
        entities = set()
        for triple in graph:
            if isinstance(triple[0], (URIRef, BNode)):
                entities.add(triple[0])
            if isinstance(triple[2], URIRef):
                entities.add(triple[2])
        
        nodes_added = 0
        nodes_updated = 0
        embeddings_generated = 0
        
        for entity in entities:
            if isinstance(entity, BNode):
                continue  # Skip blank nodes for now
            
            uri_str = self.uri_to_string(entity)
            node_types = self.determine_node_type(entity, graph)
            properties = self.extract_properties(entity, graph)
            
            # Extract label using improved method
            label = self.extract_label_from_properties(properties)
            if not label:
                label = self.extract_local_name(uri_str)
            
            # Create searchable text
            searchable_text = self.create_searchable_text(uri_str, label, properties, node_types)
            
            # Generate embedding
            embedding = None
            if self.use_embeddings and searchable_text:
                embedding = self.generate_embedding(searchable_text)
                if embedding:
                    embeddings_generated += 1
            
            node_doc = {
                "uri": uri_str,
                "local_name": self.extract_local_name(uri_str),
                "label": label,
                "searchable_text": searchable_text,
                "type": node_types,
                "properties": properties,
                "source_video": [video_id],  # Always store as array for consistency
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            # Add embedding if generated
            if embedding:
                node_doc["embedding"] = embedding
            
            try:
                self.nodes.insert_one(node_doc)
                nodes_added += 1
            except DuplicateKeyError:
                # Check if we need to migrate source_video from string to array
                existing_node = self.nodes.find_one({"uri": uri_str})
                if existing_node and isinstance(existing_node.get("source_video"), str):
                    # Migrate: convert string to array and add new video_id
                    existing_videos = [existing_node["source_video"]]
                    if video_id not in existing_videos:
                        existing_videos.append(video_id)
                    
                    update_doc = {
                        "properties": properties,
                        "label": label,
                        "searchable_text": searchable_text,
                        "updated_at": datetime.now(timezone.utc),
                        "source_video": existing_videos
                    }
                    
                    if embedding:
                        update_doc["embedding"] = embedding
                    
                    self.nodes.update_one(
                        {"uri": uri_str},
                        {"$set": update_doc}
                    )
                else:
                    # Normal update with array
                    update_doc = {
                        "properties": properties,
                        "label": label,
                        "searchable_text": searchable_text,
                        "updated_at": datetime.now(timezone.utc)
                    }
                    
                    if embedding:
                        update_doc["embedding"] = embedding
                    
                    self.nodes.update_one(
                        {"uri": uri_str},
                        {
                            "$set": update_doc,
                            "$addToSet": {"source_video": video_id}
                        }
                    )
                nodes_updated += 1
        
        print(f"✅ Processed nodes: {nodes_added} added, {nodes_updated} updated")
        if self.use_embeddings:
            print(f"   Vector embeddings: {embeddings_generated} generated")
    
    def process_edges(self, graph: Graph, video_id: str):
        """Extract and save all edges from the RDF graph."""
        print("Processing edges...")
        
        edges_added = 0
        edges_skipped = 0
        
        for triple in graph:
            if isinstance(triple[0], BNode) or isinstance(triple[2], BNode):
                continue  # Skip blank nodes
            
            subject_str = self.uri_to_string(triple[0])
            predicate_str = self.uri_to_string(triple[1])
            object_str = self.uri_to_string(triple[2])
            
            edge_doc = {
                "subject": subject_str,
                "predicate": predicate_str,
                "object": object_str,
                "source_video": [video_id],  # Always store as array for consistency
                "created_at": datetime.now(timezone.utc)
            }
            
            try:
                self.edges.insert_one(edge_doc)
                edges_added += 1
            except DuplicateKeyError:
                edges_skipped += 1
        
        print(f"✅ Processed edges: {edges_added} added, {edges_skipped} duplicates skipped")
    
    def create_global_blank_node_id(self, blank_node: BNode, video_id: str) -> str:
        """Create a globally unique ID for a blank node using video_id."""
        # Use video_id + blank node ID to ensure global uniqueness
        return f"{video_id}_{str(blank_node)}"
    
    def extract_provenance_from_blank_node(self, blank_node: BNode, graph: Graph) -> Dict[str, Any]:
        """Extract provenance information from a prov:wasDerivedFrom blank node."""
        provenance = {}
        
        # Get all properties of the blank node
        for pred, obj in graph.predicate_objects(blank_node):
            pred_str = str(pred)
            obj_str = str(obj)
            
            # Look for transcript segment properties
            if pred_str == "http://example.com/barbados-parliament-ontology#fromVideo":
                provenance["from_video"] = obj_str
            elif pred_str == "http://example.com/barbados-parliament-ontology#startTimeOffset":
                try:
                    provenance["start_offset"] = float(obj_str)
                except (ValueError, TypeError):
                    provenance["start_offset"] = obj_str
            elif pred_str == "http://example.com/barbados-parliament-ontology#endTimeOffset":
                try:
                    provenance["end_offset"] = float(obj_str)
                except (ValueError, TypeError):
                    provenance["end_offset"] = obj_str
            elif pred_str == str(RDF.type):
                provenance["segment_type"] = obj_str
        
        return provenance
    
    def process_reified_statements(self, graph: Graph, video_id: str):
        """Extract and save reified statements (provenance information)."""
        print("Processing reified statements...")
        
        statements_added = 0
        
        # Find all statements that use prov:wasDerivedFrom
        prov_derived_from_uri = URIRef("http://www.w3.org/ns/prov#wasDerivedFrom")
        
        # Find all statement nodes (these are blank nodes with rdf:type rdf:Statement)
        statement_nodes = list(graph.subjects(RDF.type, RDF.Statement))
        print(f"   Found {len(statement_nodes)} rdf:Statement nodes")
        
        for stmt_node in statement_nodes:
            # Extract the reified statement components
            subject = None
            predicate = None
            object_val = None
            provenance = {}
            
            for prop, value in graph.predicate_objects(stmt_node):
                prop_str = str(prop)
                if prop_str == str(RDF.subject):
                    subject = self.uri_to_string(value)
                elif prop_str == str(RDF.predicate):
                    predicate = self.uri_to_string(value)
                elif prop_str == str(RDF.object):
                    object_val = self.uri_to_string(value)
                elif prop_str == str(prov_derived_from_uri):
                    # Extract provenance from the referenced blank node
                    if isinstance(value, BNode):
                        provenance = self.extract_provenance_from_blank_node(value, graph)
            
            if subject and predicate and object_val:
                statement_id = self.create_statement_id(subject, predicate, object_val)
                
                # Create globally unique statement URI
                global_stmt_uri = f"_:{self.create_global_blank_node_id(stmt_node, video_id)}"
                
                statement_doc = {
                    "statement_id": statement_id,
                    "global_statement_id": f"{video_id}_{statement_id}",  # Globally unique
                    "statement_uri": global_stmt_uri,
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_val,
                    "source_video": video_id,
                    "created_at": datetime.now(timezone.utc)
                }
                
                # Add provenance information
                if provenance:
                    statement_doc.update({
                        "from_video": provenance.get("from_video"),
                        "start_offset": provenance.get("start_offset"),
                        "end_offset": provenance.get("end_offset"),
                        "segment_type": provenance.get("segment_type")
                    })
                
                try:
                    self.statements.insert_one(statement_doc)
                    statements_added += 1
                except DuplicateKeyError:
                    pass  # Skip duplicates
        
        print(f"✅ Processed reified statements: {statements_added} added")
    
    def save_video_metadata(self, video_id: str, ttl_file: str, graph: Graph):
        """Save metadata about the video source."""
        video_doc = {
            "video_id": video_id,
            "source_file": str(Path(ttl_file).name),
            "triple_count": len(graph),
            "processed_at": datetime.now(timezone.utc)
        }
        
        try:
            self.videos.insert_one(video_doc)
            print(f"✅ Video metadata saved for: {video_id}")
        except DuplicateKeyError:
            self.videos.update_one(
                {"video_id": video_id},
                {
                    "$set": {
                        "triple_count": len(graph),
                        "processed_at": datetime.now(timezone.utc)
                    }
                }
            )
            print(f"✅ Video metadata updated for: {video_id}")
    
    def process_ttl_file(self, ttl_file: str):
        """Process a complete TTL file and load it into MongoDB."""
        video_id = self.extract_video_id_from_ttl(ttl_file)
        
        print(f"Processing TTL file: {ttl_file}")
        print(f"Video ID: {video_id}")
        
        # Load RDF graph
        graph = self.load_rdf_file(ttl_file)
        
        # Process all components
        self.process_nodes(graph, video_id)
        self.process_edges(graph, video_id)
        self.process_reified_statements(graph, video_id)
        self.save_video_metadata(video_id, ttl_file, graph)
        
        print(f"✅ Successfully processed {ttl_file} into MongoDB")
        print(f"Total collections: nodes, edges, statements, videos")
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the loaded graph."""
        stats = {
            "nodes": self.nodes.count_documents({}),
            "edges": self.edges.count_documents({}),
            "statements": self.statements.count_documents({}),
            "videos": self.videos.count_documents({})
        }
        
        if self.use_embeddings:
            stats["nodes_with_embeddings"] = self.nodes.count_documents({"embedding": {"$exists": True}})
        
        return stats

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Load RDF Turtle files into MongoDB with vector search support")
    parser.add_argument("ttl_file", help="Path to the RDF Turtle file")
    parser.add_argument("--skip-embeddings", action="store_true", 
                        help="Skip generating vector embeddings (faster processing)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.ttl_file).exists():
        print(f"Error: Input file '{args.ttl_file}' does not exist.")
        sys.exit(1)
    
    try:
        # Initialize loader
        loader = RDFToMongoLoader(use_embeddings=not args.skip_embeddings)
        
        # Process the TTL file
        loader.process_ttl_file(args.ttl_file)
        
        # Show final statistics
        stats = loader.get_stats()
        print(f"\n📊 Final Statistics:")
        for collection, count in stats.items():
            print(f"  {collection}: {count:,} documents")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nTo set up MongoDB connection:")
        print("1. Create a MongoDB Atlas cluster: https://cloud.mongodb.com/")
        print("2. Get your connection string from the Atlas dashboard")
        print("3. Set environment variable: export MONGODB_CONNECTION_STRING='your-connection-string'")
        print("4. Or create a .env file with: MONGODB_CONNECTION_STRING=your-connection-string")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
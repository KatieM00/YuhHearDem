#!/usr/bin/env python3
"""
RDF to MongoDB Graph Loader

This script loads RDF Turtle files and saves them as graph structures in MongoDB Atlas
for GraphRAG-style querying and analysis.

Requirements:
- pymongo
- rdflib
- python-dotenv (optional, for environment variables)

Usage:
    python rdf_to_mongodb.py <rdf_file.ttl>
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import hashlib
from datetime import datetime, timezone

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

# Load environment variables
load_dotenv()

class RDFToMongoLoader:
    def __init__(self, connection_string: str = None, database_name: str = "parliamentary_graph"):
        """
        Initialize the RDF to MongoDB loader.
        
        Args:
            connection_string: MongoDB Atlas connection string. If None, will try to get from environment.
            database_name: Name of the MongoDB database to use
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
            self.nodes.create_index([("label", TEXT)])
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
            self.statements.create_index([("offset", ASCENDING)])
        except:
            pass
        
        # Videos collection for source metadata
        self.videos = self.db.videos
        try:
            self.videos.create_index([("video_id", ASCENDING)], unique=True)
        except:
            pass
        
        print("✅ Collections and indexes set up successfully")
    
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
                types.append("http://example.org/politics#Concept")
            elif 'Statement' in uri_str:
                types.append(str(RDF.Statement))
            else:
                types.append("http://example.org/Entity")
        
        return types
    
    def extract_properties(self, uri: URIRef, graph: Graph) -> Dict[str, Any]:
        """Extract all properties for a given URI."""
        properties = {}
        
        # Debug: show all triples for this URI
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
        
        for entity in entities:
            if isinstance(entity, BNode):
                continue  # Skip blank nodes for now
            
            uri_str = self.uri_to_string(entity)
            node_types = self.determine_node_type(entity, graph)
            properties = self.extract_properties(entity, graph)
            
            # Debug: Show properties for first few nodes
            if nodes_added + nodes_updated < 5:
                print(f"   Debug node {uri_str}:")
                print(f"     All triples for this entity:")
                for triple in graph.triples((entity, None, None)):
                    print(f"       {triple[1]} -> {triple[2]}")
                print(f"     Extracted properties: {properties}")
                print(f"     rdfs:label key: '{str(RDFS.label)}'")
                print(f"     foaf:name key: '{str(FOAF.name)}'")
                print(f"     Has rdfs:label: {str(RDFS.label) in properties}")
                print(f"     Has foaf:name: {str(FOAF.name) in properties}")
                if str(RDFS.label) in properties:
                    print(f"     rdfs:label value: '{properties[str(RDFS.label)]}'")
                if str(FOAF.name) in properties:
                    print(f"     foaf:name value: '{properties[str(FOAF.name)]}'")
                print()
            
            node_doc = {
                "uri": uri_str,
                "local_name": self.extract_local_name(uri_str),
                "type": node_types,
                "properties": properties,
                "source_video": [video_id],  # Always store as array for consistency
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            # Extract label for better searchability
            label = None
            if str(RDFS.label) in properties:
                label = properties[str(RDFS.label)]
                if nodes_added + nodes_updated < 5:
                    print(f"     Using rdfs:label: '{label}'")
            elif str(FOAF.name) in properties:
                label = properties[str(FOAF.name)]
                if nodes_added + nodes_updated < 5:
                    print(f"     Using foaf:name: '{label}'")
            else:
                label = self.extract_local_name(uri_str)
                if nodes_added + nodes_updated < 5:
                    print(f"     Using local name: '{label}' (fallback)")
            
            node_doc["label"] = label
            
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
                    
                    self.nodes.update_one(
                        {"uri": uri_str},
                        {
                            "$set": {
                                "properties": properties,
                                "label": label,  # Update label too
                                "updated_at": datetime.now(timezone.utc),
                                "source_video": existing_videos
                            }
                        }
                    )
                else:
                    # Normal update with array
                    self.nodes.update_one(
                        {"uri": uri_str},
                        {
                            "$set": {
                                "properties": properties,
                                "label": label,  # Update label too
                                "updated_at": datetime.now(timezone.utc)
                            },
                            "$addToSet": {"source_video": video_id}
                        }
                    )
                nodes_updated += 1
        
        print(f"✅ Processed nodes: {nodes_added} added, {nodes_updated} updated")
    
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
    
    def process_reified_statements(self, graph: Graph, video_id: str):
        """Extract and save reified statements (provenance information)."""
        print("Processing reified statements...")
        
        statements_added = 0
        
        # Debug: Show what statement-related triples exist
        print("🔍 Debugging reified statements...")
        
        # Look for any rdf:Statement types (including blank nodes)
        statement_nodes = list(graph.subjects(RDF.type, RDF.Statement))
        print(f"   Found {len(statement_nodes)} rdf:Statement nodes")
        
        # Look for any reification predicates
        reif_subjects = list(graph.subjects(RDF.subject, None))
        reif_predicates = list(graph.subjects(RDF.predicate, None))
        reif_objects = list(graph.subjects(RDF.object, None))
        
        print(f"   Found {len(reif_subjects)} rdf:subject triples")
        print(f"   Found {len(reif_predicates)} rdf:predicate triples")
        print(f"   Found {len(reif_objects)} rdf:object triples")
        
        # Look for ex:source and ex:offset patterns
        source_triples = list(graph.subjects(URIRef("http://example.org/source"), None))
        offset_triples = list(graph.subjects(URIRef("http://example.org/offset"), None))
        
        print(f"   Found {len(source_triples)} ex:source triples")
        print(f"   Found {len(offset_triples)} ex:offset triples")
        
        # Show a few examples of what we found
        if statement_nodes:
            print(f"   Example statement node: {statement_nodes[0]}")
            print(f"   Node type: {type(statement_nodes[0])}")
            print(f"   Is blank node: {isinstance(statement_nodes[0], BNode)}")
            
            # Show properties of first statement node
            example_node = statement_nodes[0]
            print(f"   Properties of example node:")
            for pred, obj in graph.predicate_objects(example_node):
                print(f"     {pred} -> {obj}")
        
        # Find all reified statements (including blank nodes)
        for stmt_node in statement_nodes:
            # Extract the reified statement components
            subject = None
            predicate = None
            object_val = None
            source = None
            offset = None
            
            for prop, value in graph.predicate_objects(stmt_node):
                prop_str = str(prop)
                if prop_str == str(RDF.subject):
                    subject = self.uri_to_string(value)
                elif prop_str == str(RDF.predicate):
                    predicate = self.uri_to_string(value)
                elif prop_str == str(RDF.object):
                    object_val = self.uri_to_string(value)
                elif prop_str == "http://example.org/source":
                    source = self.uri_to_string(value)
                elif prop_str == "http://example.org/offset":
                    try:
                        offset = float(str(value)) if value else None
                    except (ValueError, TypeError):
                        offset = None
            
            print(f"   Processing statement: s={subject}, p={predicate}, o={object_val}")
            
            if subject and predicate and object_val:
                statement_id = self.create_statement_id(subject, predicate, object_val)
                
                # Create globally unique statement URI
                if isinstance(stmt_node, BNode):
                    global_stmt_uri = f"_:{self.create_global_blank_node_id(stmt_node, video_id)}"
                else:
                    global_stmt_uri = self.uri_to_string(stmt_node)
                
                statement_doc = {
                    "statement_id": statement_id,
                    "global_statement_id": f"{video_id}_{statement_id}",  # Globally unique
                    "statement_uri": global_stmt_uri,
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_val,
                    "source": source,
                    "source_video": video_id,
                    "offset": offset,
                    "created_at": datetime.now(timezone.utc)
                }
                
                try:
                    self.statements.insert_one(statement_doc)
                    statements_added += 1
                    print(f"   ✅ Added statement: {statement_doc['global_statement_id']}")
                except DuplicateKeyError:
                    print(f"   ⚠️ Duplicate statement: {statement_doc['global_statement_id']}")
            else:
                print(f"   ❌ Incomplete statement: missing s/p/o")
        
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
        return {
            "nodes": self.nodes.count_documents({}),
            "edges": self.edges.count_documents({}),
            "statements": self.statements.count_documents({}),
            "videos": self.videos.count_documents({})
        }

def main():
    """Main function to run the script."""
    if len(sys.argv) != 2:
        print("Usage: python rdf_to_mongodb.py <rdf_file.ttl>")
        print("\nExample:")
        print("python rdf_to_mongodb.py transcript-dR-eoAEvPH4.ttl")
        sys.exit(1)
    
    ttl_file = sys.argv[1]
    
    # Check if input file exists
    if not Path(ttl_file).exists():
        print(f"Error: Input file '{ttl_file}' does not exist.")
        sys.exit(1)
    
    try:
        # Initialize loader
        loader = RDFToMongoLoader()
        
        # Process the TTL file
        loader.process_ttl_file(ttl_file)
        
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
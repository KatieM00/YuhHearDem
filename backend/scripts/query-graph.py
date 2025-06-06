#!/usr/bin/env python3
"""
MongoDB Graph Query Tool

This script queries the MongoDB graph database and returns relevant nodes
with configurable traversal depth, outputting results as RDF Turtle.

Requirements:
- pymongo
- python-dotenv (optional, for environment variables)

Usage:
    python mongodb_graph_query.py "query string" [--hops N] [--output file.ttl]
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from datetime import datetime, timezone
import re

try:
    from pymongo import MongoClient, TEXT
    from pymongo.errors import ConnectionFailure
    from dotenv import load_dotenv
    from rdflib import Graph, URIRef, Literal, BNode, Namespace
    from rdflib.namespace import RDF, RDFS, OWL, FOAF, XSD
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install pymongo python-dotenv rdflib")
    sys.exit(1)

# Load environment variables
load_dotenv()

class GraphQuerier:
    def __init__(self, connection_string: str = None, database_name: str = "parliamentary_graph"):
        """
        Initialize the graph querier.
        
        Args:
            connection_string: MongoDB Atlas connection string
            database_name: Name of the MongoDB database
        """
        if connection_string is None:
            connection_string = os.getenv('MONGODB_CONNECTION_STRING')
            
        if not connection_string:
            raise ValueError(
                "MongoDB connection string is required. Set MONGODB_CONNECTION_STRING environment variable."
            )
        
        try:
            self.client = MongoClient(connection_string)
            self.client.admin.command('ping')
            print("✅ Connected to MongoDB Atlas")
        except ConnectionFailure as e:
            raise ConnectionFailure(f"Failed to connect to MongoDB: {e}")
        
        self.db = self.client[database_name]
        self.nodes = self.db.nodes
        self.edges = self.db.edges
        self.statements = self.db.statements
        self.videos = self.db.videos
    
    def text_search_nodes(self, query: str, limit: int = 50) -> List[Dict]:
        """
        Perform text search on node labels and properties.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching node documents
        """
        print(f"🔍 Searching for: '{query}'")
        
        # Text search on labels
        text_results = list(self.nodes.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit))
        
        # Also search in property values (case-insensitive regex)
        regex_pattern = re.compile(re.escape(query), re.IGNORECASE)
        property_results = list(self.nodes.find({
            "$or": [
                {"properties": {"$regex": regex_pattern}},
                {"local_name": {"$regex": regex_pattern}},
                {"label": {"$regex": regex_pattern}}
            ]
        }).limit(limit))
        
        # Combine and deduplicate results
        seen_uris = set()
        combined_results = []
        
        for result in text_results + property_results:
            if result["uri"] not in seen_uris:
                seen_uris.add(result["uri"])
                combined_results.append(result)
        
        print(f"📍 Found {len(combined_results)} initial matches")
        return combined_results
    
    def get_connected_nodes(self, node_uris: Set[str], hops: int = 1) -> Set[str]:
        """
        Get all nodes connected to the given nodes within specified hops.
        
        Args:
            node_uris: Set of starting node URIs
            hops: Number of hops to traverse
            
        Returns:
            Set of all connected node URIs
        """
        if hops <= 0:
            return node_uris
        
        current_nodes = set(node_uris)
        all_nodes = set(node_uris)
        
        for hop in range(hops):
            print(f"🔗 Traversing hop {hop + 1}/{hops} (current: {len(current_nodes)} nodes)")
            
            if not current_nodes:
                break
            
            # Find all edges connected to current nodes
            connected_edges = self.edges.find({
                "$or": [
                    {"subject": {"$in": list(current_nodes)}},
                    {"object": {"$in": list(current_nodes)}}
                ]
            })
            
            next_nodes = set()
            for edge in connected_edges:
                next_nodes.add(edge["subject"])
                next_nodes.add(edge["object"])
            
            # Only add truly new nodes for next iteration
            new_nodes = next_nodes - all_nodes
            all_nodes.update(next_nodes)
            current_nodes = new_nodes
            
            print(f"   Added {len(new_nodes)} new nodes (total: {len(all_nodes)})")
        
        return all_nodes
    
    def get_subgraph(self, node_uris: Set[str]) -> Dict[str, Any]:
        """
        Extract subgraph containing specified nodes and their connections.
        
        Args:
            node_uris: Set of node URIs to include
            
        Returns:
            Dictionary containing nodes, edges, and statements
        """
        print(f"📊 Extracting subgraph for {len(node_uris)} nodes")
        
        # Get node details
        nodes = list(self.nodes.find({"uri": {"$in": list(node_uris)}}))
        
        # Get edges between these nodes
        edges = list(self.edges.find({
            "subject": {"$in": list(node_uris)},
            "object": {"$in": list(node_uris)}
        }))
        
        # Get ALL reified statements that involve any of these nodes
        # (not just edges between them)
        statements = list(self.statements.find({
            "$or": [
                {"subject": {"$in": list(node_uris)}},
                {"object": {"$in": list(node_uris)}}
            ]
        }))
        
        print(f"   Nodes: {len(nodes)}, Edges: {len(edges)}, Statements: {len(statements)}")
        
        return {
            "nodes": nodes,
            "edges": edges,
            "statements": statements
        }
    
    def query_graph(self, query: str, hops: int = 2) -> Dict[str, Any]:
        """
        Perform complete graph query with traversal.
        
        Args:
            query: Search query string
            hops: Number of hops to traverse
            
        Returns:
            Subgraph data
        """
        # Find initial matching nodes
        initial_nodes = self.text_search_nodes(query)
        
        if not initial_nodes:
            print("❌ No matching nodes found")
            return {"nodes": [], "edges": [], "statements": []}
        
        # Extract URIs from initial results
        initial_uris = {node["uri"] for node in initial_nodes}
        
        # Perform graph traversal
        all_node_uris = self.get_connected_nodes(initial_uris, hops)
        
        # Extract subgraph
        subgraph = self.get_subgraph(all_node_uris)
        
        return subgraph
    
    def subgraph_to_rdf_graph(self, subgraph: Dict[str, Any]) -> Graph:
        """
        Convert subgraph data to RDFLib Graph for proper deduplication and formatting.
        
        Args:
            subgraph: Subgraph data from query
            
        Returns:
            RDFLib Graph object
        """
        # Create new RDF graph
        graph = Graph()
        
        # Bind prefixes for the Barbados Parliament ontology
        BBP = Namespace("http://example.com/barbados-parliament-ontology#")
        SESS = Namespace("http://example.com/barbados-parliament-session/")
        SCHEMA = Namespace("http://schema.org/")
        ORG = Namespace("http://www.w3.org/ns/org#")
        PROV = Namespace("http://www.w3.org/ns/prov#")
        
        graph.bind("bbp", BBP)
        graph.bind("sess", SESS)
        graph.bind("schema", SCHEMA)
        graph.bind("org", ORG)
        graph.bind("prov", PROV)
        graph.bind("foaf", FOAF)
        graph.bind("owl", OWL)
        graph.bind("rdf", RDF)
        graph.bind("rdfs", RDFS)
        graph.bind("xsd", XSD)
        
        # Helper function to convert string to RDF term
        def string_to_rdf_term(value_str: str):
            if value_str.startswith('http://') or value_str.startswith('https://'):
                return URIRef(value_str)
            elif value_str.startswith('_:'):
                # Blank node
                return BNode(value_str[2:])  # Remove '_:' prefix
            else:
                # Try to detect if it's a number
                try:
                    if '.' in value_str:
                        float(value_str)
                        return Literal(value_str, datatype=XSD.decimal)
                    else:
                        int(value_str)
                        return Literal(value_str, datatype=XSD.integer)
                except ValueError:
                    # Check if it's a year (4 digits)
                    if re.match(r'^\d{4}$', value_str):
                        return Literal(value_str, datatype=XSD.gYear)
                    # Check if it's a date
                    elif re.match(r'^\d{4}-\d{2}-\d{2}$', value_str):
                        return Literal(value_str, datatype=XSD.date)
                    else:
                        # It's a string literal
                        return Literal(value_str)
        
        # Add node type declarations and properties
        for node in subgraph["nodes"]:
            node_uri = URIRef(node["uri"])
            
            # Add type declarations
            for node_type in node.get("type", []):
                type_uri = URIRef(node_type)
                graph.add((node_uri, RDF.type, type_uri))
            
            # Add label if available
            if "label" in node:
                graph.add((node_uri, RDFS.label, Literal(str(node["label"]))))
            
            # Add other properties
            for prop_uri, prop_values in node.get("properties", {}).items():
                if prop_uri == str(RDF.type):
                    continue  # Already handled above
                
                prop_ref = URIRef(prop_uri)
                
                # Handle both single values and lists
                values = prop_values if isinstance(prop_values, list) else [prop_values]
                
                for value in values:
                    value_term = string_to_rdf_term(str(value))
                    graph.add((node_uri, prop_ref, value_term))
        
        # Add edges (triples)
        for edge in subgraph["edges"]:
            subject = URIRef(edge["subject"])
            predicate = URIRef(edge["predicate"])
            obj = string_to_rdf_term(edge["object"])
            graph.add((subject, predicate, obj))
        
        # Add reified statements with enhanced provenance
        for stmt in subgraph["statements"]:
            if "statement_uri" in stmt and stmt["statement_uri"].startswith("_:"):
                # Use the original blank node ID
                stmt_node = BNode(stmt["statement_uri"][2:])  # Remove '_:' prefix
            else:
                # Generate a new blank node
                stmt_node = BNode(f"stmt_{stmt['statement_id'][:8]}")
            
            # Add reification triples
            graph.add((stmt_node, RDF.type, RDF.Statement))
            graph.add((stmt_node, RDF.subject, URIRef(stmt["subject"])))
            graph.add((stmt_node, RDF.predicate, URIRef(stmt["predicate"])))
            graph.add((stmt_node, RDF.object, string_to_rdf_term(stmt["object"])))
            
            # Add enhanced provenance using prov:wasDerivedFrom structure
            if any(key in stmt for key in ["from_video", "start_offset", "end_offset"]):
                # Create a blank node for the transcript segment
                segment_node = BNode(f"segment_{stmt['statement_id'][:8]}")
                
                # Link statement to segment
                graph.add((stmt_node, PROV.wasDerivedFrom, segment_node))
                
                # Add segment type
                graph.add((segment_node, RDF.type, BBP.TranscriptSegment))
                
                # Add segment properties
                if stmt.get("from_video"):
                    graph.add((segment_node, BBP.fromVideo, URIRef(stmt["from_video"])))
                
                if stmt.get("start_offset") is not None:
                    graph.add((segment_node, BBP.startTimeOffset, Literal(stmt["start_offset"], datatype=XSD.decimal)))
                
                if stmt.get("end_offset") is not None:
                    graph.add((segment_node, BBP.endTimeOffset, Literal(stmt["end_offset"], datatype=XSD.decimal)))
        
        return graph

    def subgraph_to_turtle(self, subgraph: Dict[str, Any]) -> str:
        """
        Convert subgraph data to Turtle format using RDFLib for proper formatting.
        
        Args:
            subgraph: Subgraph data from query
            
        Returns:
            Turtle format string
        """
        # Convert to RDF graph first
        rdf_graph = self.subgraph_to_rdf_graph(subgraph)
        
        # Count statements with provenance
        statements_with_provenance = sum(1 for stmt in subgraph['statements'] 
                                       if any(key in stmt for key in ["start_offset", "end_offset"]))
        
        # Add comment header
        header_comment = f"""# Query results generated at {datetime.now(timezone.utc).isoformat()}Z
# Nodes: {len(subgraph['nodes'])}, Edges: {len(subgraph['edges'])}, Statements: {len(subgraph['statements'])}
# Statements with provenance: {statements_with_provenance}
# Total triples: {len(rdf_graph)}

"""
        
        # Serialize to turtle with proper formatting
        turtle_content = rdf_graph.serialize(format='turtle')
        
        # Combine header and content
        return header_comment + turtle_content
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        return {
            "nodes": self.nodes.count_documents({}),
            "edges": self.edges.count_documents({}),
            "statements": self.statements.count_documents({}),
            "videos": self.videos.count_documents({})
        }
    
    def get_provenance_stats(self) -> Dict[str, Any]:
        """Get statistics about provenance data."""
        total_statements = self.statements.count_documents({})
        statements_with_offsets = self.statements.count_documents({
            "$and": [
                {"start_offset": {"$ne": None}},
                {"end_offset": {"$ne": None}}
            ]
        })
        
        # Sample a few statements with provenance
        sample_statements = list(self.statements.find({
            "start_offset": {"$ne": None}
        }).limit(3))
        
        return {
            "total_statements": total_statements,
            "statements_with_offsets": statements_with_offsets,
            "coverage_percentage": (statements_with_offsets / total_statements * 100) if total_statements > 0 else 0,
            "sample_statements": sample_statements
        }

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Query MongoDB graph and output RDF Turtle")
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--hops", type=int, default=2, help="Number of traversal hops (default: 2)")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--provenance", action="store_true", help="Show provenance statistics")
    
    args = parser.parse_args()
    
    try:
        # Initialize querier
        querier = GraphQuerier()
        
        if args.stats:
            stats = querier.get_stats()
            print("📊 Database Statistics:")
            for collection, count in stats.items():
                print(f"  {collection}: {count:,}")
            print()
        
        if args.provenance:
            prov_stats = querier.get_provenance_stats()
            print("🔍 Provenance Statistics:")
            print(f"  Total statements: {prov_stats['total_statements']:,}")
            print(f"  Statements with time offsets: {prov_stats['statements_with_offsets']:,}")
            print(f"  Coverage: {prov_stats['coverage_percentage']:.1f}%")
            
            if prov_stats['sample_statements']:
                print("\n  Sample statements with provenance:")
                for i, stmt in enumerate(prov_stats['sample_statements'], 1):
                    print(f"    {i}. {stmt.get('subject', 'N/A')} -> {stmt.get('predicate', 'N/A')}")
                    print(f"       Time: {stmt.get('start_offset', 'N/A')}s - {stmt.get('end_offset', 'N/A')}s")
                    print(f"       Video: {stmt.get('source_video', 'N/A')}")
            print()
        
        # Perform query
        print(f"🚀 Querying: '{args.query}' with {args.hops} hops")
        subgraph = querier.query_graph(args.query, args.hops)
        
        # Generate Turtle output
        turtle_output = querier.subgraph_to_turtle(subgraph)
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(turtle_output, encoding='utf-8')
            print(f"✅ Results saved to: {args.output}")
        else:
            print("\n" + "="*50)
            print("TURTLE OUTPUT:")
            print("="*50)
            print(turtle_output)
        
        # Summary
        statements_with_prov = sum(1 for stmt in subgraph['statements'] 
                                 if any(key in stmt for key in ["start_offset", "end_offset"]))
        
        print(f"\n📋 Query Summary:")
        print(f"  Query: '{args.query}'")
        print(f"  Hops: {args.hops}")
        print(f"  Nodes: {len(subgraph['nodes'])}")
        print(f"  Edges: {len(subgraph['edges'])}")
        print(f"  Statements: {len(subgraph['statements'])}")
        print(f"  Statements with provenance: {statements_with_prov}")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Set MONGODB_CONNECTION_STRING environment variable")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
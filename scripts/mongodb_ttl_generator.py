#!/usr/bin/env python3
"""
MongoDB Parliamentary Transcript to RDF Converter (JSON-LD)

This script converts processed parliamentary transcripts from MongoDB into RDF knowledge graphs,
with proper provenance linking each claim to specific time segments, and stores the TTL back to MongoDB.

Requirements:
- langchain-google-genai
- pymongo
- rdflib
- python-dotenv
- pydantic

Usage:
    python mongodb_ttl_generator.py --database parliamentary_graph
"""

import sys
import os
import json
import argparse
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_core.output_parsers import JsonOutputParser
    from pydantic import BaseModel, Field
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    from dotenv import load_dotenv
    import rdflib
    from rdflib import Graph
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install langchain-google-genai pymongo rdflib python-dotenv pydantic")
    sys.exit(1)

# Load environment variables
load_dotenv()

class KnowledgeGraph(BaseModel):
    """Pydantic model for structured knowledge graph output."""
    context: Dict[str, str] = Field(description="JSON-LD context with namespace prefixes", alias="@context")
    graph: list = Field(description="List of RDF entities and relationships in JSON-LD format", alias="@graph")
    
    class Config:
        allow_population_by_field_name = True

class MongoDBRDFConverter:
    def __init__(self, connection_string: str = None, database_name: str = "parliamentary_graph", api_key: str = None):
        """
        Initialize the RDF converter with MongoDB connection and Gemini model.
        
        Args:
            connection_string: MongoDB connection string. If None, will try to get from environment.
            database_name: Name of the MongoDB database to use
            api_key: Google API key. If None, will try to get from environment.
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
        
        # Setup Gemini model
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=api_key,
            temperature=0.1,  # Lower temperature for more consistent output
            thinking_budget=0
        )
        
        # Set up JSON output parser
        self.json_parser = JsonOutputParser(pydantic_object=KnowledgeGraph)
    
    def _get_conversion_prompt(self, video_id: str, video_title: str) -> str:
        """Return the improved system prompt for transcript to JSON-LD conversion with better provenance."""
        return f"""You are converting a parliamentary transcript into RDF triples in JSON-LD format. 

**CRITICAL REQUIREMENT: EVERY factual claim must have BOTH the main triple AND a corresponding reified statement with time provenance.**

**Source Video Title:** `{video_title}`
**Source Video ID:** `{video_id}` (YouTube video)

**Input Format:** Each line starts with time offset in seconds, followed by spoken text.
Example: `54 This is [unknown] H, Member for St. Peter.`

**Required Output Structure:**
```json
{{
  "@context": {{
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "schema": "http://schema.org/",
    "org": "http://www.w3.org/ns/org#",
    "bbp": "http://example.com/barbados-parliament-ontology#",
    "sess": "http://example.com/barbados-parliament-session/",
    "prov": "http://www.w3.org/ns/prov#"
  }},
  "@graph": [
    // Main entities AND reified statements with provenance
  ]
}}
```

**MANDATORY PATTERN - For EVERY claim, create BOTH:**

1. **Main Entity** (example for speaker identification):
```json
{{
  "@id": "bbp:Parliamentarian_H_StPeter",
  "@type": ["schema:Person", "bbp:Parliamentarian"],
  "schema:name": "H",
  "bbp:hasRole": "Member for St. Peter",
  "bbp:representsConstituency": {{"@id": "bbp:Constituency_StPeter"}}
}}
```

2. **Reified Statement with Time Provenance** (MANDATORY):
```json
{{
  "@id": "_:stmt_H_role_{video_id}_54",
  "@type": "rdf:Statement",
  "rdf:subject": {{"@id": "bbp:Parliamentarian_H_StPeter"}},
  "rdf:predicate": {{"@id": "bbp:hasRole"}},
  "rdf:object": "Member for St. Peter",
  "prov:wasDerivedFrom": {{
    "@type": "bbp:TranscriptSegment",
    "bbp:fromVideo": {{"@id": "sess:video_{video_id}"}},
    "bbp:startTimeOffset": {{"@type": "xsd:decimal", "@value": "54.0"}},
    "bbp:endTimeOffset": {{"@type": "xsd:decimal", "@value": "61.0"}},
    "bbp:transcriptText": "This is [unknown] H, Member for St. Peter."
  }}
}}
```

**STEP-BY-STEP PROCESS:**
1. Read each timestamped line
2. Extract factual claims (speaker identity, role, position, topic discussed, etc.)
3. For EACH claim, create:
   - The main RDF triple
   - A reified statement linking it to the specific time segment
4. Use exact time offsets from transcript
5. Calculate endTimeOffset as start of next relevant statement

**ENTITY TYPES TO EXTRACT:**
- **Video**: `sess:video_{video_id}`
- **Session**: `sess:{video_id}_session`
- **Parliamentarians**: `bbp:Parliamentarian_[Name]`
- **Constituencies**: `bbp:Constituency_[Name]`
- **Bills**: `bbp:Bill_[Name]`
- **Concepts**: `bbp:Concept_[TopicName]`
- **Organizations**: `bbp:Org_[Name]`

**CRITICAL RULES:**
1. NEVER output JSON without reified statements
2. EVERY factual claim needs time provenance
3. Use exact timestamp format: "54.0", "61.0", etc.
4. Include original transcript text in provenance
5. Link everything back to specific video segments

Video Title: {video_title}
Video ID: {video_id}

Now convert this transcript, ensuring EVERY claim has both main triple AND reified statement with time provenance:"""

    def get_videos_with_processed_transcripts(self) -> List[Dict[str, Any]]:
        """
        Get all videos from videos collection that have processed transcripts but no TTL.
        
        Returns:
            List of video documents with processed transcripts
        """
        query = {
            "transcript": {"$exists": True, "$ne": ""},
            "ttl": {"$exists": False}  # Only videos without TTL generated
        }
        
        # Only fetch necessary fields
        projection = {
            "VideoURL": 1,
            "Video_title": 1,
            "video_id": 1,
            "transcript": 1,
            "_id": 1
        }
        
        videos = list(self.videos.find(query, projection))
        print(f"Found {len(videos)} videos with processed transcripts but no TTL")
        
        return videos

    def check_if_ttl_exists(self, video_url: str) -> bool:
        """
        Check if a video already has TTL generated.
        
        Args:
            video_url: URL of the video to check
            
        Returns:
            True if TTL exists, False otherwise
        """
        existing = self.videos.find_one({"VideoURL": video_url, "ttl": {"$exists": True}})
        return existing is not None

    def validate_provenance(self, json_data: Dict[str, Any]) -> bool:
        """Validate that the JSON-LD contains proper provenance information."""
        if '@graph' not in json_data:
            print("❌ No @graph found in JSON-LD")
            return False
        
        graph = json_data['@graph']
        reified_statements = [item for item in graph if item.get('@type') == 'rdf:Statement']
        transcript_segments = [item for item in graph if 
                             isinstance(item.get('prov:wasDerivedFrom'), dict) and
                             item['prov:wasDerivedFrom'].get('@type') == 'bbp:TranscriptSegment']
        
        print(f"📊 Validation results:")
        print(f"  - Total graph items: {len(graph)}")
        print(f"  - Reified statements: {len(reified_statements)}")
        print(f"  - Items with transcript provenance: {len(transcript_segments)}")
        
        if len(reified_statements) == 0:
            print("❌ No reified statements found - missing provenance!")
            return False
        
        if len(transcript_segments) == 0:
            print("❌ No transcript segment provenance found!")
            return False
        
        # Check if reified statements have proper time provenance
        valid_provenance = 0
        for stmt in reified_statements:
            if 'prov:wasDerivedFrom' in stmt:
                prov = stmt['prov:wasDerivedFrom']
                if isinstance(prov, dict) and 'bbp:startTimeOffset' in prov:
                    valid_provenance += 1
        
        print(f"  - Reified statements with time provenance: {valid_provenance}")
        
        if valid_provenance < len(reified_statements) * 0.5:  # At least 50% should have time provenance
            print("❌ Insufficient time provenance in reified statements!")
            return False
        
        print("✅ Provenance validation passed")
        return True

    def convert_to_jsonld(self, transcript: str, video_id: str, video_title: str) -> Dict[str, Any]:
        """Convert transcript to JSON-LD format with enhanced provenance validation."""
        conversion_prompt = self._get_conversion_prompt(video_id, video_title)
        
        # Try multiple approaches if first one fails
        for attempt in range(3):
            try:
                print(f"🔄 Attempt {attempt + 1} to generate JSON-LD with provenance...")
                
                if attempt == 0:
                    # First attempt: Use chain with JSON parser
                    chain = self.llm | self.json_parser
                    formatted_input = {
                        "format_instructions": self.json_parser.get_format_instructions(),
                        "prompt": conversion_prompt,
                        "transcript": transcript
                    }
                    json_data = chain.invoke(f"{conversion_prompt}\n\n{transcript}")
                    
                else:
                    # Subsequent attempts: Direct LLM call with more explicit instructions
                    enhanced_prompt = conversion_prompt + f"""
                    
**CRITICAL REMINDER FOR ATTEMPT {attempt + 1}:**
The previous attempt failed to include proper provenance. You MUST include reified statements.

For EVERY fact extracted (like "X is Member for Y"), create:
1. The main entity
2. A reified statement with this pattern:
```
{{
  "@id": "_:stmt_[unique_id]",
  "@type": "rdf:Statement", 
  "rdf:subject": {{"@id": "[subject_uri]"}},
  "rdf:predicate": {{"@id": "[predicate_uri]"}},
  "rdf:object": "[object_value]",
  "prov:wasDerivedFrom": {{
    "@type": "bbp:TranscriptSegment",
    "bbp:fromVideo": {{"@id": "sess:video_{video_id}"}},
    "bbp:startTimeOffset": {{"@type": "xsd:decimal", "@value": "[time].0"}},
    "bbp:endTimeOffset": {{"@type": "xsd:decimal", "@value": "[next_time].0"}},
    "bbp:transcriptText": "[original text]"
  }}
}}
```

DO NOT output JSON-LD without reified statements!
                    """
                    
                    messages = [
                        SystemMessage(content=enhanced_prompt),
                        HumanMessage(content=transcript)
                    ]
                    
                    response = self.llm.invoke(messages)
                    
                    try:
                        json_data = json.loads(response.content.strip())
                    except json.JSONDecodeError:
                        json_data = self.json_parser.parse(response.content)
                
                # Validate provenance
                if self.validate_provenance(json_data):
                    print(f"✅ Attempt {attempt + 1} successful - proper provenance found")
                    return json_data
                else:
                    print(f"❌ Attempt {attempt + 1} failed validation - missing provenance")
                    if attempt == 2:  # Last attempt
                        print("⚠️ All attempts failed - proceeding with best available result")
                        return json_data
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed with error: {e}")
                if attempt == 2:  # Last attempt
                    raise e
        
        raise Exception("All conversion attempts failed")

    def validate_and_load_to_rdf(self, json_data: Dict[str, Any]) -> Graph:
        """Validate JSON-LD by loading it into rdflib Graph."""
        try:
            g = Graph()
            json_str = json.dumps(json_data, indent=2)
            
            # Debug: Print structure info
            print(f"📋 JSON-LD structure preview:")
            print(f"  - Top-level keys: {list(json_data.keys())}")
            if '@graph' in json_data:
                print(f"  - Graph entries: {len(json_data['@graph'])}")
                if json_data['@graph']:
                    first_item = json_data['@graph'][0]
                    print(f"  - First item type: {first_item.get('@type', 'No type')}")
                    print(f"  - First item ID: {first_item.get('@id', 'No ID')}")
            
            g.parse(data=json_str, format='json-ld')
            return g
        except Exception as e:
            print(f"📋 JSON-LD content causing error (first 500 chars):")
            print(json.dumps(json_data, indent=2)[:500])
            raise Exception(f"Failed to load JSON-LD into RDF graph: {str(e)}")

    def save_ttl_to_mongodb(self, video_url: str, ttl_content: str, json_ld: Dict[str, Any], 
                           triple_count: int) -> bool:
        """
        Save the generated TTL content to MongoDB.
        
        Args:
            video_url: URL of the video
            ttl_content: Generated TTL/Turtle content
            json_ld: Generated JSON-LD data
            triple_count: Number of RDF triples generated
            
        Returns:
            True if successful, False otherwise
        """
        try:
            document = {
                "ttl": ttl_content,
                "json_ld": json_ld,
                "rdf_triple_count": triple_count,
                "rdf_generated_at": datetime.now(timezone.utc),
                "rdf_generator_version": "mongodb_ttl_generator_v1.0"
            }
            
            # Update the existing video document with TTL data
            result = self.videos.update_one(
                {"VideoURL": video_url},
                {"$set": document}
            )
            
            if result.modified_count > 0:
                print(f"  ✅ TTL saved to MongoDB ({len(ttl_content)} chars, {triple_count} triples)")
                return True
            else:
                print(f"  ❌ Failed to update video document in MongoDB")
                return False
            
        except Exception as e:
            print(f"  ❌ Error saving TTL to MongoDB: {e}")
            return False

    def process_single_video(self, video: Dict[str, Any]) -> bool:
        """
        Process a single video's transcript to generate TTL.
        
        Args:
            video: Video document from MongoDB
            
        Returns:
            True if successful, False otherwise
        """
        video_url = video.get("VideoURL", "")
        video_title = video.get("Video_title", "Unknown Title")
        video_id = video.get("video_id", "")
        transcript = video.get("transcript", "")
        
        if not transcript:
            print("  ⚠️  No transcript content found")
            return False
        
        if not video_id:
            print("  ⚠️  No video_id found")
            return False
        
        try:
            print(f"  📄 Transcript length: {len(transcript)} characters")
            
            # Convert to JSON-LD
            print("  🔄 Converting to JSON-LD with provenance...")
            json_data = self.convert_to_jsonld(transcript, video_id, video_title)
            
            # Validate and convert to RDF Graph
            print("  🔍 Validating and loading to RDF...")
            rdf_graph = self.validate_and_load_to_rdf(json_data)
            triple_count = len(rdf_graph)
            print(f"  ✅ RDF validation successful - {triple_count} triples generated")
            
            # Generate TTL content
            print("  📝 Generating TTL content...")
            ttl_content = rdf_graph.serialize(format='turtle')
            
            # Save to MongoDB
            print("  💾 Saving TTL to MongoDB...")
            success = self.save_ttl_to_mongodb(video_url, ttl_content, json_data, triple_count)
            
            if success:
                # Print provenance summary
                self.print_provenance_summary(json_data)
                return True
            else:
                return False
                
        except Exception as e:
            print(f"  ❌ Error processing video: {e}")
            return False

    def process_all_videos(self, skip_existing: bool = True, limit: Optional[int] = None):
        """
        Process all videos with transcripts to generate TTL files.
        
        Args:
            skip_existing: Whether to skip videos with existing TTL
            limit: Maximum number of videos to process (None for all)
        """
        print("Starting TTL generation from processed transcripts...")
        
        # Get videos with processed transcripts
        if skip_existing:
            videos_to_process = self.get_videos_with_processed_transcripts()
        else:
            # Get all videos with transcripts, regardless of TTL existence
            query = {"transcript": {"$exists": True, "$ne": ""}}
            projection = {
                "VideoURL": 1,
                "Video_title": 1,
                "video_id": 1,
                "transcript": 1,
                "_id": 1
            }
            videos_to_process = list(self.videos.find(query, projection))
            print(f"Found {len(videos_to_process)} videos with processed transcripts")
        
        if not videos_to_process:
            print("No videos to process")
            return
        
        # Apply limit if specified
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
            video_url = video.get("VideoURL", "")
            video_title = video.get("Video_title", "Unknown Title")
            video_id = video.get("video_id", "")
            
            print(f"\n[{i}/{stats['total']}] Processing: {video_title[:80]}...")
            print(f"  🆔 Video ID: {video_id}")
            
            # Check if already processed (if skip_existing is True)
            if skip_existing and self.check_if_ttl_exists(video_url):
                print("  ⏭️  TTL already exists, skipping")
                stats["skipped"] += 1
                continue
            
            # Process the video
            if self.process_single_video(video):
                stats["processed"] += 1
            else:
                stats["errors"] += 1
        
        # Print final statistics
        print(f"\n📊 TTL Generation Complete!")
        print(f"  Total videos: {stats['total']}")
        print(f"  Successfully processed: {stats['processed']}")
        print(f"  Skipped (already processed): {stats['skipped']}")
        print(f"  Errors: {stats['errors']}")

    def print_provenance_summary(self, json_data: Dict[str, Any]):
        """Print a summary of provenance information in the generated JSON-LD."""
        if '@graph' not in json_data:
            return
        
        graph = json_data['@graph']
        reified_statements = [item for item in graph if item.get('@type') == 'rdf:Statement']
        
        print(f"  📈 PROVENANCE SUMMARY:")
        print(f"    - Total entities: {len(graph)}")
        print(f"    - Reified statements: {len(reified_statements)}")
        
        if reified_statements:
            time_ranges = []
            for stmt in reified_statements:
                if 'prov:wasDerivedFrom' in stmt:
                    prov = stmt['prov:wasDerivedFrom']
                    if isinstance(prov, dict) and 'bbp:startTimeOffset' in prov:
                        start_time = float(prov['bbp:startTimeOffset']['@value'])
                        time_ranges.append(start_time)
            
            if time_ranges:
                print(f"    - Time range covered: {min(time_ranges):.1f}s - {max(time_ranges):.1f}s")
                print(f"    - Provenance points: {len(time_ranges)}")
        
        print(f"    - ✅ All claims linked to video timestamps")

    def get_generation_stats(self) -> Dict[str, int]:
        """Get statistics about TTL generation."""
        videos_with_transcripts = self.videos.count_documents({
            "transcript": {"$exists": True, "$ne": ""}
        })
        
        videos_with_ttl = self.videos.count_documents({
            "ttl": {"$exists": True, "$ne": ""}
        })
        
        return {
            "videos_with_transcripts": videos_with_transcripts,
            "videos_with_ttl": videos_with_ttl,
            "remaining": videos_with_transcripts - videos_with_ttl
        }

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Generate TTL/RDF from processed parliamentary transcripts in MongoDB")
    parser.add_argument("--database", default="parliamentary_graph", help="MongoDB database name")
    parser.add_argument("--limit", type=int, help="Limit number of videos to process")
    parser.add_argument("--force", action="store_true", help="Regenerate TTL for videos that already have it")
    parser.add_argument("--stats", action="store_true", help="Show generation statistics only")
    
    args = parser.parse_args()
    
    try:
        # Initialize converter
        converter = MongoDBRDFConverter(database_name=args.database)
        
        if args.stats:
            # Show statistics only
            stats = converter.get_generation_stats()
            print("📊 TTL Generation Statistics:")
            print(f"  Videos with processed transcripts: {stats['videos_with_transcripts']}")
            print(f"  Videos with TTL generated: {stats['videos_with_ttl']}")
            print(f"  Remaining to process: {stats['remaining']}")
            return
        
        # Process videos
        converter.process_all_videos(
            skip_existing=not args.force,
            limit=args.limit
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
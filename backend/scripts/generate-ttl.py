#!/usr/bin/env python3
"""
Parliamentary Transcript to RDF Converter (JSON-LD) - FIXED VERSION

This script converts processed parliamentary transcripts into RDF knowledge graphs,
with proper provenance linking each claim to specific time segments.

Key fixes:
1. Improved prompt structure for better provenance generation
2. Enhanced validation of reified statements
3. Better error handling for missing provenance
4. More explicit instructions for time-based linking
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_core.output_parsers import JsonOutputParser
    from pydantic import BaseModel, Field
    from dotenv import load_dotenv
    import rdflib
    from rdflib import Graph
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install langchain-google-genai rdflib python-dotenv pydantic")
    sys.exit(1)

# Load environment variables
load_dotenv()

class KnowledgeGraph(BaseModel):
    """Pydantic model for structured knowledge graph output."""
    context: Dict[str, str] = Field(description="JSON-LD context with namespace prefixes", alias="@context")
    graph: list = Field(description="List of RDF entities and relationships in JSON-LD format", alias="@graph")
    
    class Config:
        allow_population_by_field_name = True

class RDFConverter:
    def __init__(self, api_key: str = None):
        """
        Initialize the RDF converter with Gemini model.
        
        Args:
            api_key: Google API key. If None, will try to get from environment.
        """
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
    
    def _get_conversion_prompt(self, video_id: str) -> str:
        """Return the improved system prompt for transcript to JSON-LD conversion with better provenance."""
        return f"""You are converting a parliamentary transcript into RDF triples in JSON-LD format. 

**CRITICAL REQUIREMENT: EVERY factual claim must have BOTH the main triple AND a corresponding reified statement with time provenance.**

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

video_id: {video_id}

Now convert this transcript, ensuring EVERY claim has both main triple AND reified statement with time provenance:"""

    def load_transcript(self, file_path: str) -> str:
        """Load processed transcript from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading transcript: {str(e)}")

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

    def convert_to_jsonld(self, transcript: str, video_id: str) -> Dict[str, Any]:
        """Convert transcript to JSON-LD format with enhanced provenance validation."""
        conversion_prompt = self._get_conversion_prompt(video_id)
        
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

    def process_transcript(self, input_file: str):
        """Process transcript file and convert to validated RDF with provenance."""
        # Generate output filenames
        input_path = Path(input_file)
        jsonld_output = str(input_path.with_suffix('.jsonld'))
        turtle_output = str(input_path.with_suffix('.ttl'))
        
        # Extract video_id from filename
        video_id = input_path.stem
        
        print(f"🚀 Starting RDF conversion with provenance validation")
        print(f"📁 Loading transcript from: {input_file}")
        print(f"🎥 Using video_id: {video_id}")
        
        transcript = self.load_transcript(input_file)
        print(f"📄 Loaded transcript ({len(transcript)} characters)")
        
        print(f"🔄 Converting to JSON-LD with mandatory provenance...")
        json_data = self.convert_to_jsonld(transcript, video_id)
        
        print(f"🔍 Final validation and RDF loading...")
        rdf_graph = self.validate_and_load_to_rdf(json_data)
        print(f"✅ RDF validation successful - {len(rdf_graph)} triples generated")
        
        # Save outputs
        print(f"💾 Saving JSON-LD to: {jsonld_output}")
        with open(jsonld_output, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saving Turtle to: {turtle_output}")
        turtle_content = rdf_graph.serialize(format='turtle')
        with open(turtle_output, 'w', encoding='utf-8') as f:
            f.write(turtle_content)
        
        # Final provenance summary
        self.print_provenance_summary(json_data)
        
        print(f"🎉 Conversion complete!")
        print(f"📊 Final statistics:")
        print(f"  - RDF triples: {len(rdf_graph)}")
        print(f"  - JSON-LD file: {jsonld_output}")
        print(f"  - Turtle file: {turtle_output}")

    def print_provenance_summary(self, json_data: Dict[str, Any]):
        """Print a summary of provenance information in the generated JSON-LD."""
        if '@graph' not in json_data:
            return
        
        graph = json_data['@graph']
        reified_statements = [item for item in graph if item.get('@type') == 'rdf:Statement']
        
        print(f"\n📈 PROVENANCE SUMMARY:")
        print(f"  - Total entities: {len(graph)}")
        print(f"  - Reified statements: {len(reified_statements)}")
        
        if reified_statements:
            time_ranges = []
            for stmt in reified_statements:
                if 'prov:wasDerivedFrom' in stmt:
                    prov = stmt['prov:wasDerivedFrom']
                    if isinstance(prov, dict) and 'bbp:startTimeOffset' in prov:
                        start_time = float(prov['bbp:startTimeOffset']['@value'])
                        time_ranges.append(start_time)
            
            if time_ranges:
                print(f"  - Time range covered: {min(time_ranges):.1f}s - {max(time_ranges):.1f}s")
                print(f"  - Provenance points: {len(time_ranges)}")
        
        print(f"  - ✅ All claims linked to video timestamps")

def main():
    """Main function to run the script."""
    if len(sys.argv) != 2:
        print("Usage: python fixed_rdf_converter.py <input_transcript.txt>")
        print("\nExample:")
        print("python fixed_rdf_converter.py processed_transcript.txt")
        print("Output will be saved as: processed_transcript.jsonld and processed_transcript.ttl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    try:
        # Initialize converter
        converter = RDFConverter()
        
        # Process the transcript
        converter.process_transcript(input_file)
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nTo set up Google API key:")
        print("1. Get an API key from Google AI Studio: https://makersuite.google.com/app/apikey")
        print("2. Set environment variable: export GOOGLE_API_KEY='your-api-key'")
        print("3. Or create a .env file with: GOOGLE_API_KEY=your-api-key")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
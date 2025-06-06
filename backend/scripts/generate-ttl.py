#!/usr/bin/env python3
"""
Parliamentary Transcript to RDF Converter (JSON-LD)

This script converts processed parliamentary transcripts into RDF knowledge graphs,
using JSON-LD format for more reliable parsing and validation.

Requirements:
- langchain-google-genai
- rdflib
- python-dotenv (optional, for environment variables)

Usage:
    python rdf_converter_jsonld.py input_transcript.txt
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
    context: Dict[str, str] = Field(description="JSON-LD context with namespace prefixes")
    graph: list = Field(description="List of RDF entities and relationships in JSON-LD format")

class RDFConverter:
    def __init__(self, api_key: str = None):
        """
        Initialize the RDF converter with Gemini models.
        
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
        
        # Primary LLM for initial conversion (temperature 0.0 for consistency)
        self.llm_convert = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=api_key,
            temperature=0.0,
            thinking_budget=0  # Disable thinking mode
        )
        
        # Secondary LLM for error correction (temperature 1.0 for creativity in fixing)
        self.llm_correct = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=api_key,
            temperature=1.0,
            thinking_budget=0  # Disable thinking mode
        )
        
        # Set up JSON output parser
        self.json_parser = JsonOutputParser(pydantic_object=KnowledgeGraph)
        
        self.conversion_prompt = None  # Will be set dynamically
        self.correction_prompt = self._get_correction_prompt()
    
    def _get_conversion_prompt(self, video_id: str) -> str:
        """Return the system prompt for transcript to JSON-LD conversion."""
        return f"""Your task is to convert the provided parliamentary session transcript into a comprehensive RDF knowledge graph in JSON-LD format. The graph should capture speakers, their roles, constituencies, political affiliations, stances on the debated Bill, key arguments, mentioned organizations, legislation, events, and other relevant entities and their relationships. Crucially, each generated RDF claim must include provenance linking it to the specific time segment in the video.

**Source Video ID:** `{video_id}` (YouTube video)

**Input Transcript Format:**
Each line of the transcript begins with a numerical time offset in seconds from the start of the video, followed by the spoken text.
Example:
`54 This is [unknown] H, Member for St. Peter.`
`61 Thank you very much, Mr. Speaker.`
`62 I rise to join this debate...`

**Output Format Instructions:**

You must output a valid JSON object with this exact structure:

```json
{{
  "context": {{
    "@context": {{
      "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
      "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
      "xsd": "http://www.w3.org/2001/XMLSchema#",
      "schema": "http://schema.org/",
      "org": "http://www.w3.org/ns/org#",
      "bbp": "http://example.com/barbados-parliament-ontology#",
      "sess": "http://example.com/barbados-parliament-session/",
      "prov": "http://www.w3.org/ns/prov#"
    }}
  }},
  "graph": [
    // Array of JSON-LD objects representing RDF entities
  ]
}}
```

**Entity Creation Guidelines:**

1. **Main Video Entity:**
```json
{{
  "@id": "sess:video_{video_id}",
  "@type": "bbp:VideoRecording",
  "schema:identifier": "{video_id}",
  "bbp:hasVideoId": "{video_id}"
}}
```

2. **Parliamentary Session:**
```json
{{
  "@id": "sess:{video_id}_session",
  "@type": "bbp:ParliamentarySession",
  "schema:name": "Barbados Parliament Session - Video {video_id}",
  "bbp:recordedIn": {{"@id": "sess:video_{video_id}"}},
  "bbp:debatesBill": {{"@id": "bbp:Bill_ChildProtectionBill"}}
}}
```

3. **Parliamentarians (for each speaker):**
```json
{{
  "@id": "bbp:Parliamentarian_UnknownH",
  "@type": ["schema:Person", "bbp:Parliamentarian"],
  "schema:name": "Unknown H",
  "bbp:hasRole": "Member for St. Peter",
  "bbp:representsConstituency": {{"@id": "bbp:Constituency_StPeter"}},
  "org:memberOf": {{"@id": "bbp:PoliticalParty_BarbadosLabourParty"}},
  "bbp:supportsBill": {{"@id": "bbp:Bill_ChildProtectionBill"}},
  "bbp:discussesConcept": [
    {{"@id": "bbp:Concept_ChildProtection"}},
    {{"@id": "bbp:Concept_MandatoryReporting"}}
  ]
}}
```

4. **Reified Statements with Provenance:**
For each factual claim, create both the main entity and a reified statement:
```json
{{
  "@id": "_:stmt_UnknownH_Role_{video_id}_54",
  "@type": "rdf:Statement",
  "rdf:subject": {{"@id": "bbp:Parliamentarian_UnknownH"}},
  "rdf:predicate": {{"@id": "bbp:hasRole"}},
  "rdf:object": "Member for St. Peter",
  "prov:wasDerivedFrom": {{
    "@type": "bbp:TranscriptSegment",
    "bbp:fromVideo": {{"@id": "sess:video_{video_id}"}},
    "bbp:startTimeOffset": {{"@type": "xsd:decimal", "@value": "54.0"}},
    "bbp:endTimeOffset": {{"@type": "xsd:decimal", "@value": "61.0"}}
  }}
}}
```

**Comprehensive Entity Extraction:**

Extract and create JSON-LD objects for:

- **Parliamentarians**: All speakers with their roles, constituencies, party affiliations
- **Bills/Legislation**: The main bill being debated and any referenced legislation
- **Organizations**: All mentioned organizations (schools, ministries, NGOs, etc.)
- **Locations**: Constituencies and places mentioned
- **Concepts**: Key topics discussed (child protection, mandatory reporting, etc.)
- **Events**: Specific incidents or historical events mentioned
- **Conventions/Treaties**: International agreements referenced

**URI Naming Convention:**
- Parliamentarians: `bbp:Parliamentarian_Name` or `bbp:Parliamentarian_Constituency`
- Constituencies: `bbp:Constituency_Name`
- Political Parties: `bbp:PoliticalParty_Name`
- Bills: `bbp:Bill_Name`
- Legislation: `bbp:Legislation_Name_Year`
- Organizations: `bbp:Org_Name`
- Concepts: `bbp:Concept_TopicName`
- Events: `schema:Event_Name_Year`
- Places: `schema:Place_Name`

**Critical Requirements:**
1. Output ONLY valid JSON - no markdown, no explanations, no code fences
2. Every factual claim must have a corresponding reified statement with time provenance
3. Use exact time offsets from the transcript for startTimeOffset
4. Calculate endTimeOffset as the start time of the next relevant line
5. Handle `[unknown]` speakers appropriately without inventing information

video_id: {video_id}

Convert the following transcript:"""

    def _get_correction_prompt(self) -> str:
        """Return the system prompt for JSON-LD correction."""
        return """You are an expert in JSON-LD and RDF. The following JSON-LD content has syntax errors that prevent it from being parsed as valid JSON or loaded by rdflib.

You have access to:
1. The original transcript that was being converted
2. The JSON-LD content with errors
3. The specific error message from the parser

Please fix ONLY the syntax errors while preserving all the semantic content and structure. Common JSON issues to fix:
- Missing or extra commas
- Unescaped quotes in string values
- Malformed JSON structure
- Invalid JSON-LD @context or @type usage
- Incorrect array/object syntax
- Missing closing brackets or braces

Use the original transcript as context to ensure that:
- Entity names and labels accurately reflect the content
- Relationships make sense in context
- URIs are properly formed and consistent
- All semantic meaning is preserved

Return ONLY the corrected JSON-LD with no additional commentary or explanations."""

    def load_transcript(self, file_path: str) -> str:
        """
        Load processed transcript from file.
        
        Args:
            file_path: Path to the input processed transcript file
            
        Returns:
            Raw file content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading transcript: {str(e)}")

    def clean_json_output(self, json_content: str) -> str:
        """
        Clean JSON output by removing markdown code fences and extra formatting.
        
        Args:
            json_content: Raw JSON content that might contain code fences
            
        Returns:
            Clean JSON content
        """
        # Remove markdown code fences
        lines = json_content.split('\n')
        cleaned_lines = []
        in_code_block = False
        
        for line in lines:
            stripped = line.strip()
            
            # Check for code fence start/end
            if stripped.startswith('```'):
                if not in_code_block:
                    in_code_block = True
                else:
                    in_code_block = False
                continue
            
            # If we're in a code block or not in any block, include the line
            if in_code_block or not any(stripped.startswith(fence) for fence in ['```']):
                cleaned_lines.append(line)
        
        # Join back and strip extra whitespace
        cleaned = '\n'.join(cleaned_lines).strip()
        
        # Remove any remaining markdown artifacts
        cleaned = cleaned.replace('```json', '').replace('```', '')
        
        return cleaned.strip()

    def convert_to_jsonld(self, transcript: str, video_id: str) -> Dict[str, Any]:
        """
        Convert transcript to JSON-LD format.
        
        Args:
            transcript: Processed transcript content
            video_id: Video ID extracted from filename
            
        Returns:
            JSON-LD data as dictionary
        """
        conversion_prompt = self._get_conversion_prompt(video_id)
        
        # Create chain with JSON parser
        chain = self.llm_convert | self.json_parser
        
        try:
            # Use the parser format instructions in the prompt
            messages = [
                SystemMessage(content=conversion_prompt + f"\n\n{self.json_parser.get_format_instructions()}"),
                HumanMessage(content=transcript)
            ]
            
            response = self.llm_convert.invoke(messages)
            raw_content = response.content.strip()
            
            # Clean any markdown formatting
            cleaned_content = self.clean_json_output(raw_content)
            
            # Parse as JSON
            try:
                json_data = json.loads(cleaned_content)
                return json_data
            except json.JSONDecodeError as e:
                # Fallback: try to parse with the chain
                try:
                    json_data = chain.invoke({"transcript": transcript})
                    return json_data
                except:
                    # Last resort: return the cleaned string for manual correction
                    raise Exception(f"JSON parsing failed: {e}\nContent: {cleaned_content[:500]}...")
            
        except Exception as e:
            raise Exception(f"Error converting to JSON-LD: {str(e)}")

    def correct_jsonld(self, json_content: str, error_message: str, original_transcript: str) -> Dict[str, Any]:
        """
        Correct JSON-LD syntax errors using original transcript as context.
        
        Args:
            json_content: JSON-LD content with errors (as string)
            error_message: Error message from parser
            original_transcript: Original transcript for context
            
        Returns:
            Corrected JSON-LD data as dictionary
        """
        
        correction_message = f"""{self.correction_prompt}

ORIGINAL TRANSCRIPT (for context):
{original_transcript}

PARSER ERROR MESSAGE:
{error_message}

JSON-LD CONTENT TO FIX:
{json_content}"""
        
        messages = [
            HumanMessage(content=correction_message)
        ]
        
        try:
            response = self.llm_correct.invoke(messages)
            corrected_content = response.content.strip()
            
            # Clean any markdown formatting that might have been added
            corrected_content = self.clean_json_output(corrected_content)
            
            # Parse the corrected JSON
            json_data = json.loads(corrected_content)
            return json_data
            
        except Exception as e:
            raise Exception(f"Error correcting JSON-LD: {str(e)}")

    def validate_jsonld(self, json_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate JSON-LD by loading it into rdflib.
        
        Args:
            json_data: JSON-LD data as dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            g = Graph()
            # Convert dict to JSON string for rdflib
            json_str = json.dumps(json_data, indent=2)
            g.parse(data=json_str, format='json-ld')
            return True, None
        except Exception as e:
            return False, str(e)

    def jsonld_to_turtle(self, json_data: Dict[str, Any]) -> str:
        """
        Convert JSON-LD to Turtle format.
        
        Args:
            json_data: Valid JSON-LD data
            
        Returns:
            Turtle format string
        """
        try:
            g = Graph()
            json_str = json.dumps(json_data, indent=2)
            g.parse(data=json_str, format='json-ld')
            return g.serialize(format='turtle')
        except Exception as e:
            raise Exception(f"Error converting JSON-LD to Turtle: {str(e)}")

    def process_transcript(self, input_file: str):
        """
        Process transcript file and convert to validated RDF.
        
        Args:
            input_file: Path to input processed transcript file
        """
        # Generate output filenames
        input_path = Path(input_file)
        jsonld_output = str(input_path.with_suffix('.jsonld'))
        turtle_output = str(input_path.with_suffix('.ttl'))
        
        # Extract video_id from filename (basename without extension)
        video_id = input_path.stem
        
        print(f"Loading processed transcript from: {input_file}")
        print(f"Using video_id: {video_id}")
        transcript = self.load_transcript(input_file)
        print(f"Loaded transcript ({len(transcript)} characters)")
        
        print("Converting transcript to JSON-LD...")
        try:
            json_data = self.convert_to_jsonld(transcript, video_id)
        except Exception as e:
            print(f"Initial conversion failed: {e}")
            return
        
        print("Validating JSON-LD syntax...")
        is_valid, error_message = self.validate_jsonld(json_data)
        
        # Correction loop (up to 5 attempts)
        correction_attempts = 0
        max_attempts = 5
        
        while not is_valid and correction_attempts < max_attempts:
            correction_attempts += 1
            print(f"JSON-LD validation failed (attempt {correction_attempts}/{max_attempts})")
            print(f"Error: {error_message}")
            print("Attempting to correct syntax errors...")
            
            try:
                # Convert current JSON data back to string for correction
                json_str = json.dumps(json_data, indent=2)
                json_data = self.correct_jsonld(json_str, error_message, transcript)
                is_valid, error_message = self.validate_jsonld(json_data)
                
                if is_valid:
                    print(f"JSON-LD syntax corrected successfully after {correction_attempts} attempt(s)")
                    break
                else:
                    print(f"Correction attempt {correction_attempts} still has errors: {error_message}")
                    
            except Exception as e:
                print(f"Error during correction attempt {correction_attempts}: {e}")
                if correction_attempts == max_attempts:
                    break
        
        if not is_valid:
            print(f"Failed to correct JSON-LD syntax after {max_attempts} attempts")
            print(f"Final error: {error_message}")
            print("Saving JSON-LD content with syntax errors for manual review...")
            jsonld_output = str(input_path.with_suffix('.jsonld.error'))
        else:
            print("JSON-LD syntax validation successful!")
        
        # Save the JSON-LD content
        print(f"Saving JSON-LD to: {jsonld_output}")
        with open(jsonld_output, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        if is_valid:
            # Convert to Turtle and save
            print(f"Converting to Turtle format...")
            try:
                turtle_content = self.jsonld_to_turtle(json_data)
                print(f"Saving Turtle to: {turtle_output}")
                with open(turtle_output, 'w', encoding='utf-8') as f:
                    f.write(turtle_content)
                
                # Final validation with statistics
                g = Graph()
                json_str = json.dumps(json_data, indent=2)
                g.parse(data=json_str, format='json-ld')
                print(f"Final RDF graph contains {len(g)} triples")
                print(f"Conversion complete! Files saved:")
                print(f"  JSON-LD: {jsonld_output}")
                print(f"  Turtle: {turtle_output}")
                
            except Exception as e:
                print(f"Error converting to Turtle: {e}")
                print(f"JSON-LD file saved successfully: {jsonld_output}")
        else:
            print("Note: Output file contains syntax errors and requires manual correction")

def main():
    """Main function to run the script."""
    if len(sys.argv) != 2:
        print("Usage: python rdf_converter_jsonld.py <input_transcript.txt>")
        print("\nExample:")
        print("python rdf_converter_jsonld.py processed_transcript.txt")
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
#!/usr/bin/env python3
"""
Parliamentary Transcript to RDF Converter

This script converts processed parliamentary transcripts into RDF Turtle format,
with validation and error correction using LangChain and Google's Gemini model.

Requirements:
- langchain-google-genai
- rdflib
- python-dotenv (optional, for environment variables)

Usage:
    python rdf_converter.py input_transcript.txt
"""

import sys
import os
from pathlib import Path
from typing import Optional

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage, SystemMessage
    from dotenv import load_dotenv
    import rdflib
    from rdflib import Graph
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install langchain-google-genai rdflib python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

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
            temperature=1.0,
            thinking_budget=0  # Disable thinking mode
        )
        
        self.conversion_prompt = None  # Will be set dynamically
        self.correction_prompt = self._get_correction_prompt()
    
    def _get_conversion_prompt(self, video_id: str) -> str:
        """Return the system prompt for transcript to RDF conversion."""
        return f"""You are an expert Semantic Web engineer.
Convert the following excerpt from a **parliamentary transcript** into RDF triples **in Turtle syntax**.
★ Mandatory guidelines
1. Use these prefixes exactly:
   @prefix ex:    <http://example.org/> .
   @prefix pol:   <http://example.org/politics#> .
   @prefix foaf:  <http://xmlns.com/foaf/0.1/> .
   @prefix owl:   <http://www.w3.org/2002/07/owl#> .
   @prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
   @prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
   @prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .

2. **Create clean, concise URIs** - use short, meaningful names for entities and predicates:
   - Good: ex:Speaker, pol:discusses, pol:supports, pol:opposes, pol:appeals
   - Bad: pol:appealsToEveryoneToSaySomethingIfTheyComeAcrossAbuse
   - Use simple verbs like: discusses, mentions, supports, opposes, proposes, questions, appeals, states, argues

3. **CRITICAL: Every custom entity AND predicate MUST have an rdfs:label**:
   - For entities: ex:Speaker rdfs:label "Mr. Speaker" .
   - For predicates: pol:appeals rdfs:label "appeals to everyone to say something if they come across abuse" .
   - For concepts: ex:AbuseReporting rdfs:label "Abuse Reporting" .
   - NO entity or predicate should exist without a human-readable label

4. **Example of proper predicate labeling**:
   ```turtle
   # Define the predicate with its label
   pol:appeals rdfs:label "appeals to everyone to say something if they come across abuse" .
   
   # Use the predicate in statements
   ex:MemberForStPeter pol:appeals ex:AbuseReporting .
   
   # Reify with provenance
   [] rdf:type rdf:Statement ;
      rdf:subject ex:MemberForStPeter ;
      rdf:predicate pol:appeals ;
      rdf:object ex:AbuseReporting ;
      ex:source ex:{video_id} ;
      ex:offset "2382"^^xsd:decimal .
   ```

5. **Include concept resources** (`ex:Concept_*`) for any abstract idea, policy, or theme so the public can query concepts directly.

6. **All entities must be connected** by at least one explicit relationship – no isolated nodes.

7. **Capture provenance for EVERY asserted triple** using RDF reification. For each triple `S P O`, create a new blank node (e.g. []) with:
       rdf:type      rdf:Statement ;
       rdf:subject   S ;
       rdf:predicate P ;
       rdf:object    O ;
       ex:source     ex:{video_id} ;
       ex:offset     "{{offset}}"^^xsd:decimal ;   # seconds from video start

8. Use **FOAF** for people (`foaf:Person`, `foaf:name`) and **OWL** if you need to declare new classes.

9. Replace pronouns with canonical URIs or literals.

10. Output **only valid Turtle** – no commentary, no Markdown.

11. Input consists of lines with an integer as the seconds offset from the start of the video file and then a sentence or two.

**Template structure:**
```turtle
# First: Define all entities and predicates with labels
ex:EntityName rdfs:label "Human Readable Name" .
pol:predicateName rdfs:label "human readable description of action" .

# Then: Use them in statements
ex:Subject pol:predicate ex:Object .

# Finally: Add reification for provenance
[] rdf:type rdf:Statement ;
   rdf:subject ex:Subject ;
   rdf:predicate pol:predicate ;
   rdf:object ex:Object ;
   ex:source ex:{video_id} ;
   ex:offset "123"^^xsd:decimal .
```

video_id: {video_id}

Convert the following transcript:"""

    def _get_correction_prompt(self) -> str:
        """Return the system prompt for RDF correction."""
        return """You are an expert Semantic Web engineer. The following RDF Turtle content has syntax errors that prevent it from being parsed by rdflib.

You have access to:
1. The original transcript that was being converted
2. The RDF content with syntax errors
3. The specific error message from the parser

Please fix ONLY the syntax errors while preserving all the semantic content and structure. Do not add, remove, or modify any triples - only fix syntax issues like:
- Missing periods at end of statements
- Incorrect punctuation
- Malformed URIs
- Invalid literal syntax
- Missing prefixes
- Incorrect use of blank nodes
- Quote escaping issues
- Namespace declaration problems

Use the original transcript as context to ensure that:
- Entity names and labels accurately reflect the content
- Relationships make sense in context
- URIs are properly formed and consistent
- All semantic meaning is preserved

Return ONLY the corrected Turtle syntax with no additional commentary or explanations."""

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

    def clean_turtle_output(self, rdf_content: str) -> str:
        """
        Clean Turtle output by removing markdown code fences and extra formatting.
        
        Args:
            rdf_content: Raw RDF content that might contain code fences
            
        Returns:
            Clean Turtle content
        """
        # Remove markdown code fences
        lines = rdf_content.split('\n')
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
        cleaned = cleaned.replace('```turtle', '').replace('```ttl', '').replace('```', '')
        
        return cleaned.strip()

    def convert_to_rdf(self, transcript: str, video_id: str) -> str:
        """
        Convert transcript to RDF Turtle format.
        
        Args:
            transcript: Processed transcript content
            video_id: Video ID extracted from filename
            
        Returns:
            RDF Turtle content
        """
        conversion_prompt = self._get_conversion_prompt(video_id)
        messages = [
            SystemMessage(content=conversion_prompt),
            HumanMessage(content=transcript)
        ]
        
        try:
            response = self.llm.invoke(messages)
            raw_content = response.content.strip()
            
            # Clean any markdown formatting
            cleaned_content = self.clean_turtle_output(raw_content)
            
            return cleaned_content
        except Exception as e:
            raise Exception(f"Error converting to RDF: {str(e)}")

    def correct_rdf(self, rdf_content: str, error_message: str, original_transcript: str) -> str:
        """
        Correct RDF syntax errors using original transcript as context.
        
        Args:
            rdf_content: RDF Turtle content with errors
            error_message: Error message from rdflib
            original_transcript: Original transcript for context
            
        Returns:
            Corrected RDF Turtle content
        """
        # Truncate transcript if it's very long to avoid token limits
        transcript_excerpt = original_transcript
        if len(original_transcript) > 2000:
            transcript_excerpt = original_transcript[:2000] + "\n... (transcript truncated for context)"
        
        correction_message = f"""{self.correction_prompt}

ORIGINAL TRANSCRIPT (for context):
{transcript_excerpt}

PARSER ERROR MESSAGE:
{error_message}

RDF CONTENT TO FIX:
{rdf_content}"""
        
        messages = [
            HumanMessage(content=correction_message)
        ]
        
        try:
            response = self.llm.invoke(messages)
            corrected_content = response.content.strip()
            
            # Clean any markdown formatting that might have been added
            corrected_content = self.clean_turtle_output(corrected_content)
            
            return corrected_content
        except Exception as e:
            raise Exception(f"Error correcting RDF: {str(e)}")

    def validate_rdf(self, rdf_content: str) -> tuple[bool, Optional[str]]:
        """
        Validate RDF Turtle syntax using rdflib.
        
        Args:
            rdf_content: RDF Turtle content to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            g = Graph()
            g.parse(data=rdf_content, format='turtle')
            return True, None
        except Exception as e:
            return False, str(e)

    def process_transcript(self, input_file: str):
        """
        Process transcript file and convert to validated RDF Turtle.
        
        Args:
            input_file: Path to input processed transcript file
        """
        # Generate output filename by replacing extension with .ttl
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix('.ttl'))
        
        # Extract video_id from filename (basename without extension)
        video_id = input_path.stem
        
        print(f"Loading processed transcript from: {input_file}")
        print(f"Using video_id: {video_id}")
        transcript = self.load_transcript(input_file)
        print(f"Loaded transcript ({len(transcript)} characters)")
        
        print("Converting transcript to RDF Turtle...")
        rdf_content = self.convert_to_rdf(transcript, video_id)
        
        print("Validating RDF syntax...")
        is_valid, error_message = self.validate_rdf(rdf_content)
        
        # Correction loop (up to 5 attempts)
        correction_attempts = 0
        max_attempts = 5
        
        while not is_valid and correction_attempts < max_attempts:
            correction_attempts += 1
            print(f"RDF validation failed (attempt {correction_attempts}/{max_attempts})")
            print(f"Error: {error_message}")
            print("Attempting to correct syntax errors with original transcript context...")
            
            try:
                # Pass the original transcript as context for correction
                rdf_content = self.correct_rdf(rdf_content, error_message, transcript)
                is_valid, error_message = self.validate_rdf(rdf_content)
                
                if is_valid:
                    print(f"RDF syntax corrected successfully after {correction_attempts} attempt(s)")
                    break
                else:
                    print(f"Correction attempt {correction_attempts} still has errors: {error_message}")
                    
            except Exception as e:
                print(f"Error during correction attempt {correction_attempts}: {e}")
                if correction_attempts == max_attempts:
                    break
        
        if not is_valid:
            print(f"Failed to correct RDF syntax after {max_attempts} attempts")
            print(f"Final error: {error_message}")
            print("Saving RDF content with syntax errors for manual review...")
            output_file = str(input_path.with_suffix('.ttl.error'))
        else:
            print("RDF syntax validation successful!")
        
        # Save the RDF content
        print(f"Saving RDF Turtle to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(rdf_content)
        
        print(f"Conversion complete! RDF saved to: {output_file}")
        
        if is_valid:
            # Final validation with statistics
            g = Graph()
            g.parse(data=rdf_content, format='turtle')
            print(f"Final RDF graph contains {len(g)} triples")
        else:
            print("Note: Output file contains syntax errors and requires manual correction")

def main():
    """Main function to run the script."""
    if len(sys.argv) != 2:
        print("Usage: python rdf_converter.py <input_transcript.txt>")
        print("\nExample:")
        print("python rdf_converter.py processed_transcript.txt")
        print("Output will be saved as: processed_transcript.ttl")
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
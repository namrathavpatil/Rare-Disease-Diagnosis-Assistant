import os
import json
import logging
from typing import List, Dict, Any
import anthropic

logger = logging.getLogger(__name__)

class LLMEntityExtractor:
    """
    Uses LLM (via Anthropic Claude) to extract medical entities from text.
    """
    def __init__(self, api_key: str = None, model_name: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name

        # Prompt template for entity extraction
        self.prompt_template = """
You are a medical entity extraction expert. Extract all disease names, drug names, and symptoms from the following text.

Return ONLY a valid JSON array of objects with 'text' and 'type' fields. The 'type' should be one of: 'disease', 'drug', 'symptom'.

Example output format:
[
    {{"text": "diabetes", "type": "disease"}},
    {{"text": "metformin", "type": "drug"}},
    {{"text": "high blood sugar", "type": "symptom"}}
]

Text to analyze: {text}

JSON response:"""

    def _call_claude(self, system: str, user_prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        """Helper to call Claude API."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text.strip()

    def _clean_json(self, result: str) -> str:
        """Strip markdown fences from JSON response."""
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        return result.strip()

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using LLM and return as a list of dicts.
        """
        try:
            prompt = self.prompt_template.format(text=text)

            result = self._call_claude(
                system="You are a medical entity extraction expert. Always respond with valid JSON only.",
                user_prompt=prompt,
                max_tokens=500,
                temperature=0.1
            )

            try:
                result = self._clean_json(result)
                entities = json.loads(result)

                valid_entities = []
                for entity in entities:
                    if isinstance(entity, dict) and 'text' in entity and 'type' in entity:
                        if entity['type'] in ['disease', 'drug', 'symptom']:
                            valid_entities.append({
                                'text': entity['text'].strip(),
                                'type': entity['type'],
                                'confidence': 0.9
                            })

                logger.info(f"Extracted {len(valid_entities)} entities using LLM")
                return valid_entities

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.error(f"Raw response: {result}")
                return []

        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return []

    def extract_entities_with_relationships(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and infer relationships using LLM.
        """
        try:
            relationship_prompt = f"""
You are a medical knowledge extraction expert. Analyze the following text and extract:
1. All disease names, drug names, and symptoms
2. Relationships between these entities

Return a JSON object with 'entities' and 'relationships' arrays.

Example format:
{{
    "entities": [
        {{"text": "diabetes", "type": "disease"}},
        {{"text": "metformin", "type": "drug"}},
        {{"text": "high blood sugar", "type": "symptom"}}
    ],
    "relationships": [
        {{"source": "diabetes", "target": "high blood sugar", "type": "has_symptom"}},
        {{"source": "metformin", "target": "diabetes", "type": "treats"}}
    ]
}}

Text to analyze: {text}

JSON response:"""

            result = self._call_claude(
                system="You are a medical knowledge extraction expert. Always respond with valid JSON only.",
                user_prompt=relationship_prompt,
                max_tokens=1000,
                temperature=0.1
            )

            result = self._clean_json(result)
            data = json.loads(result)

            entities = data.get('entities', [])
            relationships = data.get('relationships', [])

            valid_entities = []
            for entity in entities:
                if isinstance(entity, dict) and 'text' in entity and 'type' in entity:
                    if entity['type'] in ['disease', 'drug', 'symptom']:
                        valid_entities.append({
                            'text': entity['text'].strip(),
                            'type': entity['type'],
                            'confidence': 0.9
                        })

            valid_relationships = []
            for rel in relationships:
                if isinstance(rel, dict) and all(k in rel for k in ['source', 'target', 'type']):
                    source_type = None
                    target_type = None

                    for entity in valid_entities:
                        if entity['text'].lower() == rel['source'].lower():
                            source_type = entity['type']
                        if entity['text'].lower() == rel['target'].lower():
                            target_type = entity['type']

                    if source_type and target_type:
                        valid_relationships.append({
                            'source': rel['source'].strip(),
                            'target': rel['target'].strip(),
                            'type': rel['type'],
                            'source_type': source_type,
                            'target_type': target_type,
                            'confidence': 0.8
                        })

            return {
                'entities': valid_entities,
                'relationships': valid_relationships
            }

        except Exception as e:
            logger.error(f"Error extracting entities and relationships: {e}")
            return {'entities': [], 'relationships': []}

    def extract_graph_from_prompt(self, prompt: str) -> dict:
        """
        Given a prompt that asks for a knowledge graph in JSON format, call the LLM and parse the result.
        Returns a dict with 'nodes' and 'edges'.
        """
        try:
            result = self._call_claude(
                system="You are a medical knowledge graph builder. Always respond with valid JSON only. Keep the graph concise - max 30 nodes and 40 edges. Use short IDs.",
                user_prompt=prompt,
                max_tokens=4096,
                temperature=0.1
            )
            result = self._clean_json(result)
            data = json.loads(result)
            if not isinstance(data, dict) or "nodes" not in data or "edges" not in data:
                logger.error(f"LLM did not return a valid graph JSON: {data}")
                return {}
            return data
        except Exception as e:
            logger.error(f"Error extracting graph from LLM: {e}")
            return {}

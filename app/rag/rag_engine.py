import logging
import json
import os
from typing import List, Dict, Any
import anthropic
from app.config import settings
from app.core.knowledge_graph import KnowledgeGraph
from rag_ready_graph_builder import RAGReadyGraphBuilder

logger = logging.getLogger(__name__)

class RAGEngine:
    """Retrieval-Augmented Generation engine for medical Q&A."""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model_name = settings.model_name
        self.rag_builder = RAGReadyGraphBuilder()
        logger.info("RAG Engine initialized with Claude")

    def _call_claude(self, system: str, user_prompt: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
        """Helper to call Claude API."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text.strip()

    def retrieve_context(self, query: str, max_nodes: int = 5) -> (List[Dict[str, Any]], float):
        """Retrieve relevant context from the knowledge graph and return top similarity score as confidence."""
        relevant_nodes = self.knowledge_graph.search_nodes(query, limit=max_nodes)

        top_confidence = 0.0
        if relevant_nodes and 'similarity_score' in relevant_nodes[0]:
            top_confidence = float(relevant_nodes[0]['similarity_score'])

        context = []
        for node in relevant_nodes:
            node_data = {
                'id': node['id'],
                'type': node['type'],
                'name': node['name'],
                'metadata': node['metadata']
            }
            context.append(node_data)

        return context, top_confidence

    def retrieve_rag_ready_context(self, query: str, top_k: int = 5) -> (List[Dict[str, Any]], float):
        """Retrieve relevant context from the RAG-ready graph."""
        try:
            if not os.path.exists("rag_ready_graph.json"):
                logger.warning("RAG-ready graph not found")
                return [], 0.0

            with open("rag_ready_graph.json", "r") as f:
                rag_structure = json.load(f)

            chunks = rag_structure.get("chunks", [])
            if not chunks:
                logger.warning("No chunks found in RAG-ready graph")
                return [], 0.0

            self.rag_builder.build_faiss_index(chunks)
            relevant_chunks = self.rag_builder.retrieve_relevant_chunks(query, top_k)

            context = []
            top_confidence = 0.0

            for chunk in relevant_chunks:
                chunk_data = {
                    'id': chunk['id'],
                    'type': chunk['type'],
                    'source': chunk['source'],
                    'content': chunk['content'],
                    'retrieval_score': chunk.get('retrieval_score', 0.0),
                    'metadata': chunk.get('metadata', {})
                }
                context.append(chunk_data)

                if chunk.get('retrieval_score', 0.0) > top_confidence:
                    top_confidence = chunk.get('retrieval_score', 0.0)

            return context, top_confidence

        except Exception as e:
            logger.error(f"Error retrieving RAG-ready context: {e}")
            return [], 0.0

    def format_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Format the prompt for the LLM with retrieved context."""
        prompt = f"""You are a medical assistant specialized in rare diseases.\nYour responses should be accurate, evidence-based, and aligned with current medical guidelines.\n\nUse the following context to answer the question. If you cannot find the answer in the context, you may provide general medical knowledge, but clearly indicate that this information is not from the provided context.\n\nContext:\n"""
        for item in context:
            prompt += f"\n{item['type'].upper()}: {item['name']}\n"
            if 'metadata' in item and item['metadata']:
                for key, value in item['metadata'].items():
                    prompt += f"{key}: {value}\n"
        prompt += f"\nQuestion: {query}\nAnswer:"
        return prompt

    def format_rag_ready_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Format the prompt for the LLM with RAG-ready context."""
        prompt = (
            "You are a medical assistant. Use the following PubMed articles and Orphanet details to answer the question. "
            "Cite the source (PubMed or Orphanet) in your answer. If you use a PubMed article, cite its title. "
            "If you use Orphanet, say 'according to Orphanet...'.\n\n"
        )
        pubmed_chunks = [c for c in context if c.get('source') == 'pubmed']
        orphanet_chunks = [c for c in context if c.get('source') == 'orphanet']
        fda_chunks = [c for c in context if c.get('source') == 'fda']

        if pubmed_chunks:
            prompt += "PubMed Articles:\n"
            for i, chunk in enumerate(pubmed_chunks, 1):
                title = chunk['metadata'].get('title', 'No title')
                abstract = chunk['content']
                prompt += f"{i}. Title: {title}\n   Content: {abstract}\n"
        if orphanet_chunks:
            prompt += "\nOrphanet Details:\n"
            for chunk in orphanet_chunks:
                prompt += f"- {chunk['content']}\n"
        if fda_chunks:
            prompt += "\nFDA Drug Information:\n"
            for chunk in fda_chunks:
                prompt += f"- {chunk['content']}\n"

        prompt += f"\nQuestion: {query}\nAnswer (synthesize and cite sources):"
        return prompt

    def format_confidence_prompt(self, answer: str, context: List[Dict[str, Any]], query: str) -> str:
        """Prompt the LLM to self-assess its confidence in the answer."""
        return f"""Given the following context and answer, rate your confidence in the answer on a scale from 0 (not confident) to 1 (very confident).\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer: {answer}\n\nConfidence (0-1):"""

    def format_followup_prompt(self, answer: str, context: List[Dict[str, Any]], query: str) -> str:
        """Prompt the LLM to generate a follow-up question if confidence is low."""
        return f"""Given the following context and answer, if you are not confident in the answer, suggest a follow-up question to ask the user for clarification or more information.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer: {answer}\n\nFollow-up question (if needed):"""

    def generate_answer_with_rag_ready(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Generate an answer using RAG-ready graph data with LLM."""
        try:
            context, context_confidence = self.retrieve_rag_ready_context(query, top_k)

            if not context:
                return {
                    'answer': "I couldn't find any relevant information in the RAG-ready knowledge base to answer your question. Please try building the RAG-ready graph first or ask a different question.",
                    'context': [],
                    'confidence': 0.0,
                    'follow_up_question': None,
                    'method': 'rag_ready'
                }

            prompt = self.format_rag_ready_prompt(query, context)

            answer = self._call_claude(
                system="You are a comprehensive medical assistant with access to scientific research, clinical information, and drug safety data. Always prioritize information from the provided context and synthesize information from multiple sources when available.",
                user_prompt=prompt,
                max_tokens=2048,
                temperature=0.3
            )

            # Ask LLM to self-assess confidence
            confidence_prompt = self.format_confidence_prompt(answer, context, query)
            conf_result = self._call_claude(
                system="You are a medical assistant. Rate your confidence in the answer from 0 to 1. Respond with just the number.",
                user_prompt=confidence_prompt,
                max_tokens=50,
                temperature=0.1
            )

            try:
                llm_confidence = float(conf_result.split()[0])
            except Exception:
                llm_confidence = context_confidence

            follow_up_question = None
            if llm_confidence < 0.7:
                followup_prompt = self.format_followup_prompt(answer, context, query)
                follow_up_question = self._call_claude(
                    system="You are a medical assistant. Suggest a follow-up question if confidence is low.",
                    user_prompt=followup_prompt,
                    max_tokens=100,
                    temperature=0.3
                )

            return {
                'answer': answer,
                'context': context,
                'confidence': llm_confidence,
                'follow_up_question': follow_up_question,
                'method': 'rag_ready'
            }

        except Exception as e:
            logger.error(f"Error generating answer with RAG-ready: {str(e)}")
            return {
                'answer': "I encountered an error while processing your question with the RAG-ready system. Please try again.",
                'context': [],
                'confidence': 0.0,
                'follow_up_question': None,
                'method': 'rag_ready'
            }

    def generate_answer(self, query: str, max_context_nodes: int = 5) -> Dict[str, Any]:
        """Generate an answer using RAG, with confidence and follow-up question."""
        try:
            context, context_confidence = self.retrieve_context(query, max_context_nodes)
            if not context:
                return {
                    'answer': "I couldn't find any relevant information in the knowledge base to answer your question.",
                    'context': [],
                    'confidence': 0.0,
                    'follow_up_question': None
                }

            prompt = self.format_prompt(query, context)

            answer = self._call_claude(
                system="You are a medical assistant specialized in rare diseases. Always prioritize information from the provided context, but if it's insufficient, you may provide general medical knowledge while clearly indicating this.",
                user_prompt=prompt,
                max_tokens=2048,
                temperature=0.3
            )

            # Ask LLM to self-assess confidence
            confidence_prompt = self.format_confidence_prompt(answer, context, query)
            conf_result = self._call_claude(
                system="You are a medical assistant. Rate your confidence in the answer from 0 to 1. Respond with just the number.",
                user_prompt=confidence_prompt,
                max_tokens=50,
                temperature=0.1
            )

            try:
                llm_confidence = float(conf_result.split()[0])
            except Exception:
                llm_confidence = context_confidence

            follow_up_question = None
            if llm_confidence < 0.7:
                followup_prompt = self.format_followup_prompt(answer, context, query)
                follow_up_question = self._call_claude(
                    system="You are a medical assistant. Suggest a follow-up question if confidence is low.",
                    user_prompt=followup_prompt,
                    max_tokens=100,
                    temperature=0.3
                )

            return {
                'answer': answer,
                'context': context,
                'confidence': llm_confidence,
                'follow_up_question': follow_up_question
            }

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                'answer': "I encountered an error while processing your question. Please try again.",
                'context': [],
                'confidence': 0.0,
                'follow_up_question': None
            }

    def answer_question(self, query: str, use_rag_ready: bool = False) -> Dict[str, Any]:
        """Main method to answer a question using RAG."""
        logger.info(f"Processing question: {query}")

        if use_rag_ready:
            result = self.generate_answer_with_rag_ready(query)
        else:
            result = self.generate_answer(query)

        logger.info("Answer generated successfully")
        return result

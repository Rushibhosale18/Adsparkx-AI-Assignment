import os
import json

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class SupportAgent:
    def __init__(self, chroma_path="chroma_db"):
        self.chroma_path = chroma_path
        # Initialize embeddings (must match ingest.py)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embeddings)
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})
        
        # Initialize LLM
        # Assumes GEMINI_API_KEY is set in environment or .env
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        
    def detect_persona(self, user_message: str) -> str:
        """Detects if the user is a Technical Expert, Frustrated User, or Business Executive."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert customer persona classifier. Classify the user's message into exactly one of these three categories based on their language and intent: 'Technical Expert', 'Frustrated User', or 'Business Executive'.\n\n- Technical Expert: Uses technical jargon, asks about APIs, configurations, logs, or root causes.\n- Frustrated User: Uses emotional language, expresses anger or urgency, repeated complaints.\n- Business Executive: Focuses on impact, SLAs, billing, resolution time, outcome-oriented, uses minimal technical jargon.\n\nRespond with ONLY the category name. No other text."),
            ("user", "{user_message}")
        ])
        chain = prompt | self.llm
        result = chain.invoke({"user_message": user_message})
        content = result.content.strip()
        
        # Fallback to a default if the LLM output is weird
        valid_personas = ["Technical Expert", "Frustrated User", "Business Executive"]
        for p in valid_personas:
            if p.lower() in content.lower():
                return p
        return "Frustrated User" # Default fallback
        
    def check_escalation(self, user_message: str, context_docs, persona: str) -> (bool, str):
        """Determines if the issue should be escalated to a human."""
        # Rule 1: No relevant documents found
        if not context_docs:
            return True, "No relevant documentation found in the knowledge base."
            
        # Rule 2: Explicit escalation triggers
        escalation_keywords = ["cancel", "refund", "lawsuit", "legal", "manager", "human", "talk to someone"]
        if any(keyword in user_message.lower() for keyword in escalation_keywords):
            return True, "User requested human escalation or mentioned sensitive topics (billing/legal)."
            
        return False, ""

    def generate_response(self, user_message: str, persona: str, context_docs) -> str:
        """Generates a persona-adapted response based on retrieved context."""
        context_text = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in context_docs])
        
        system_instruction = """You are an AI customer support agent. Answer the user's query based strictly on the provided Context. 
Do NOT hallucinate information. If the answer is not in the Context, say 'I do not have enough information to answer that.'

You must adapt your tone and style based on the User Persona:
- Technical Expert: Provide a detailed, technical response. Include root cause analysis if applicable, step-by-step troubleshooting, and use technical terminology appropriately.
- Frustrated User: Be highly empathetic, use simple language, reassure the user, and provide action-oriented steps clearly. Apologize for the inconvenience.
- Business Executive: Be concise, focus on business impact and outcomes. Minimize technical jargon and provide clear, estimated resolution guidance.

Context:
{context}

User Persona: {persona}
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("user", "{user_message}")
        ])
        
        chain = prompt | self.llm
        result = chain.invoke({
            "context": context_text,
            "persona": persona,
            "user_message": user_message
        })
        
        return result.content
        
    def generate_handoff_summary(self, user_message: str, persona: str, context_docs) -> dict:
        """Generates a structured JSON handoff summary for the human agent."""
        doc_sources = list(set([doc.metadata.get('source', 'Unknown') for doc in context_docs]))
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant helping a human support agent. Generate a JSON summary of the escalation. Do not wrap the JSON in markdown formatting (like ```json), just output the raw JSON object.\n\nThe JSON must have these keys:\n- 'persona': (string) The user's detected persona.\n- 'issue': (string) A concise 1-sentence summary of the user's issue.\n- 'documents_used': (list of strings) The source documents retrieved.\n- 'attempted_steps': (list of strings) What the automated agent attempted to do or would have done.\n- 'recommendation': (string) Suggested next step for the human agent.\n"),
            ("user", "User Message: {user_message}\nPersona: {persona}\nSources: {sources}")
        ])
        
        chain = prompt | self.llm
        result = chain.invoke({
            "user_message": user_message,
            "persona": persona,
            "sources": json.dumps(doc_sources)
        })
        
        content = result.content.strip()
        # Clean up potential markdown formatting
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        try:
            summary = json.loads(content)
        except json.JSONDecodeError:
            summary = {
                "persona": persona,
                "issue": user_message,
                "documents_used": doc_sources,
                "attempted_steps": ["Automated RAG retrieval"],
                "recommendation": "Review user query manually."
            }
        return summary

    def process_query(self, user_message: str):
        # 1. Detect Persona
        persona = self.detect_persona(user_message)
        
        # 2. Retrieve Context
        docs = self.retriever.invoke(user_message)
        
        # 3. Check Escalation
        should_escalate, reason = self.check_escalation(user_message, docs, persona)
        
        if should_escalate:
            handoff_summary = self.generate_handoff_summary(user_message, persona, docs)
            return {
                "persona": persona,
                "sources": [doc.metadata.get('source', 'Unknown') for doc in docs],
                "response": "I apologize, but this issue requires a human specialist. I am escalating your case now.",
                "escalated": True,
                "escalation_reason": reason,
                "handoff_summary": handoff_summary
            }
            
        # 4. Generate Response
        response = self.generate_response(user_message, persona, docs)
        
        # 5. Fallback escalation if LLM says it doesn't know
        if "do not have enough information" in response.lower() or "don't have enough information" in response.lower():
            handoff_summary = self.generate_handoff_summary(user_message, persona, docs)
            return {
                "persona": persona,
                "sources": [doc.metadata.get('source', 'Unknown') for doc in docs],
                "response": "I couldn't find the exact answer in my knowledge base. Let me connect you with a human agent.",
                "escalated": True,
                "escalation_reason": "LLM indicated lack of information.",
                "handoff_summary": handoff_summary
            }
            
        return {
            "persona": persona,
            "sources": list(set([doc.metadata.get('source', 'Unknown') for doc in docs])),
            "response": response,
            "escalated": False,
            "escalation_reason": None,
            "handoff_summary": None
        }

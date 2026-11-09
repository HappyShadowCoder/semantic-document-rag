from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

PERSONA_PROMPTS = {
    "Strict Fact-Based": (
        "You are an objective document assistant. Answer questions ONLY using the verified context provided below.\n"
        "Never extrapolate, speculate, or introduce external knowledge.\n"
        "If the answer cannot be found in the context, reply exactly:\n"
        "'I don't have enough information in this document.'\n"
        "Always cite relevant page numbers when referencing facts."
    ),
    "Executive Summary": (
        "You are an executive research analyst. Provide concise, high-level summaries and key bullet points based strictly on the context.\n"
        "Focus on metrics, decisions, risks, and conclusions found in the text.\n"
        "If information is missing, state it clearly."
    ),
    "Detailed Explainer": (
        "You are an instructional tutor and thorough document analyst. Provide detailed, step-by-step explanations\n"
        "grounded solely in the provided context, breaking down complex concepts while citing page numbers."
    )
}

def get_rag_prompt(persona: str = "Strict Fact-Based", doc_name: str = "", total_pages: int = 0):
    persona_text = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["Strict Fact-Based"])
    
    doc_info = f"Document: {doc_name} ({total_pages} pages)" if doc_name else "Uploaded Document"

    system_message = f"""{persona_text}

---
CONTEXT METADATA:
{doc_info}

RULES:
1. Base all answers strictly on the provided Context.
2. If lists or tables appear in the context, preserve their completeness.
3. Include page citations like [Page X] where appropriate.
4. If the context does not contain the answer, do not invent anything. State clearly that the document lacks the information.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "Context:\n{context}\n\nQuestion: {input}")
    ])
    return prompt

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Given a conversation history and a follow-up question, rephrase the follow-up question into an independent standalone question that can be understood without the conversation history. Do NOT answer the question, only reformulate it if needed, otherwise return it as is."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from app.config import Config

class LLMService:
    def __init__(self, vector_store):
        # Initialize the LLM using the updated langchain-openai syntax
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo", # or "gpt-4o-mini"
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Base retriever from your vector store
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Dictionary to store chat histories by session ID
        self.session_store = {}
        
        # Initialize the full conversation RAG chain
        self.rag_chain = self._build_conversational_rag()

    def _get_session_history(self, session_id: str):
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()
        return self.session_store[session_id]

    def _build_conversational_rag(self):
        # 1. Setup the history-aware retriever prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # 2. Setup the QA response generation prompt
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # 3. Combine them into a final chain
        base_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # 4. Wrap with session history manager
        return RunnableWithMessageHistory(
            base_rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def ask(self, session_id: str, question: str) -> str:
        """Call this method to query the RAG system with context memory."""
        config = {"configurable": {"session_id": session_id}}
        response = self.rag_chain.invoke({"input": question}, config=config)
        return response["answer"]
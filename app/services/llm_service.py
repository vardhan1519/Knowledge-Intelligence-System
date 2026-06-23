from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from app.config import Config


print(Config.OPENAI_API_KEY)
class LLMService:
    def __init__(self, vector_store):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=Config.OPENAI_API_KEY
        )
        
#Server
from fastapi import FastAPI

#Manage query
import asyncio
from typing import AsyncIterable
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import CallbackManager

#Manage .env 
import os
from dotenv import load_dotenv, find_dotenv

#Class
from pydantic import BaseModel

#Manage data and temporary File
import gzip
import base64
import tempfile

#Manage Pinecone -> vector database, Embeddings for indexing
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader  

#Manage LangChain memory
from langchain.memory import ConversationSummaryBufferMemory

#Manage langChain template
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage 

#Manage LangChain LLM
from langchain.llms import Cohere
from langchain.chat_models import ChatCohere
from langchain.chains import LLMChain

#Manage Cohere tokenize
from cohere.client import Client


#Launch server
app = FastAPI()

#Manage API KEY
load_dotenv(find_dotenv(), override=True)
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')  
PINECONE_ENV = os.environ.get('PINECONE_ENV')
PINECONE_INDEX = os.environ.get('PINECONE_INDEX')

#Const for memory
N_TOKENS = 4000

#Manage tokenize of Cohere
co = Client(COHERE_API_KEY)
tokenizer = co.tokenize

# Manage fundamental element 
llm = Cohere(cohere_api_key=COHERE_API_KEY)
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

#Manage global variable of the server
vectorstore = None
chat_llm = None
callback = None

#Manage global state of the server
index_created = False
llm_created = False

#Manage chat history of the LLM
chat_history = []


#Manage data inside Query
class Question(BaseModel):
    question: str
    top_research: int

class DocumentStr(BaseModel):
    content: str

class Options(BaseModel):
    temperature: float


#Create the index inside Pinecone
def create_index(file_path):
  
  index_name = PINECONE_INDEX

  # Delete all previous indexes
  indexes = pinecone.list_indexes()
  for i in indexes:
    pinecone.delete_index(i)
  
  #Splitt the document into chunks
  splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
  doc = TextLoader(file_path).load() 
  chunks = splitter.split_documents(doc)

  # create the index
  pinecone.create_index(index_name, dimension=4096, metric='cosine')
  return Pinecone.from_documents(chunks, embeddings, index_name=index_name)

#Create the llm that will be used for the RAG
def create_chat_llm(temperature):

  global callback
  callback = AsyncIteratorCallbackHandler()

  global chat_llm
  global llm

  chat_llm = ChatCohere(
    streaming=True,
    verbose=False,
    temperature=temperature,
    callback_manager=CallbackManager([callback]),
    memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=4096), #Cohere(cohere_api_key=COHERE_API_KEY)
    cohere_api_key=COHERE_API_KEY)
  
  global llm_created
  llm_created = True



#Before sending the qestion to the LLM, we will update the query with the chat history in order to be able to ask on previous question
def contextualize_question(question: str):
    global chat_history
    
    template = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is.
    
    chat_history:
    {chat_history}

    Question:
    {question}"""

    prompt = PromptTemplate(
        input_variables=['chat_history', 'question'],
        template=template
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    contextualize_question = llm_chain.run({'chat_history': chat_history, 'question': question})

    #print(contextualize_question)
    return contextualize_question

# clean up the response of retrieve
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

#genrating the final prompt before sending to the LLM
def final_augment_prompt(contestualized_question: str, top_research: int):
    
    global vectorstore
    global chat_history


    # get top X results from knowledge base
    results = vectorstore.similarity_search(contestualized_question, k=top_research)
    print(results)

    # clean upfrom the results
    source_knowledge = format_docs(results) 
    
    # feed into the final prompt
    '''
      of context AND with the history to answer the question at the end.
      History:
      {chat_history}
    '''

    final_prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Contexts:
    {source_knowledge}

    Query: {contestualized_question}"""
    #print(final_prompt)

    return final_prompt

# generate the response of the question
async def send_question_rag(question: str, top_research: int) -> AsyncIterable[str]:

  if llm_created :
    
    #Question updated
    contestualize_question = contextualize_question(question)
    #Final prompt that we will give to the LLM
    final_prompt = final_augment_prompt(contestualize_question, top_research)
    prompt = HumanMessage(content=final_prompt)

    #
    task = asyncio.create_task(
          chat_llm.agenerate(messages=[[prompt]])
      )

    try:
          async for token in callback.aiter():
              yield token
    except Exception as e:
          print(f"Caught exception: {e}")
    finally:
          callback.done.set()

    response = await task

    #update the chat history
    global chat_history
    chat_history.extend([HumanMessage(content=question), AIMessage(content=response.generations[0][0].text)] )
    #manage_history(N_TOKENS)
    #print(chat_history)

    
  else:
     yield "LLM not generated"


#IN PROGRESS
def manage_history(n_tokens):
    global chat_history
    
    total_tokens = 0

    for generation in reversed(chat_history):
      
      print("GENERATION  :  ", generation)
      for message in reversed(generation):
        
        if isinstance(message, HumanMessage):
          tokens = tokenizer.encode(message.content)
          
        elif isinstance(message, AIMessage):
          tokens = message.tokens
        
        total_tokens += len(tokens)  
        if total_tokens >= n_tokens:
          return chat_history[:chat_history.index(generation)+1]

    return chat_history

@app.get("/get_index")
def get_index():
    global index_created
    if index_created:
       return True
    elif len(pinecone.list_indexes()) > 0 :
       return True
    return False

@app.get("/get_llm")
def get_llm():
    global llm_created
    return llm_created



@app.post("/create_index")
async def index_document(document: DocumentStr):
    
    global vectorstore
    global index_created

    decoded = base64.b64decode(document.content) # bytes
    decompressed = gzip.decompress(decoded) # bytes
    
    # Write bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        fp.write(decompressed) 
        file_path = fp.name

    # create the index 
    vectorstore = create_index(file_path)
    #print(vectorstore)

    # Delete temporary file
    os.remove(file_path)

    index_created = True
    
    return True


@app.post("/create_llm")
async def create_llm(options : Options):
   create_chat_llm(options.temperature)
   return True
   


@app.post("/stream_chat")
async def stream_chat(question: Question):
  generator = send_question_rag(question.question,question.top_research)
  return StreamingResponse(generator, media_type="text/event-stream")
























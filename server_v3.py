#Server
from fastapi import FastAPI

#Manage query
import asyncio
from typing import AsyncIterable
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler

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
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
 

#Manage LangChain memory
from langchain.memory import ConversationSummaryBufferMemory

#Manage langChain template
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage 

#Manage LangChain LLM
from langchain.llms import Cohere
from langchain.chat_models import ChatCohere
from langchain.chains import LLMChain

#Launch server
app = FastAPI()

#Manage API KEY
load_dotenv(find_dotenv(), override=True)
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')  
PINECONE_ENV = os.environ.get('PINECONE_ENV')
PINECONE_INDEX = os.environ.get('PINECONE_INDEX')

# Manage fundamental element 
llm = Cohere(cohere_api_key=COHERE_API_KEY) # set up a basic Cohere LLM 
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY) # bridge between human language and AI language
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV) # set up the Pincone connexion

#Manage global variable of the server
vectorstore = None # manage the vector db of our document
chat_llm = None # the main LLM for the conversation
callback = None # manage the stream answer
conversation_summary_buffer_memory = None # manage the history of the conversation

#Manage global state of the server
index_created = False
llm_created = False
mutex = asyncio.Lock()

#Manage chat history of the LLM
chat_history = []
summary_chat_history=""


#Manage data inside Query
class Question(BaseModel):
    question: str
    top_research: int
    temperature: float

class DocumentStr(BaseModel):
    content: str
    chunk_size: int

class Options(BaseModel):
    temperature: float


#Create the index inside Pinecone
def create_index(data, chunk_size): #file_path
  
  index_name = PINECONE_INDEX

  # Delete all previous indexes
  indexes = pinecone.list_indexes()
  for i in indexes:
    pinecone.delete_index(i)
  
  #Splitt the document into chunks
  # chunk_size : maximum size of text chunk in number of characters
  # chunk_overlap : pecifies the number of overlapping characters between adjacent chunks.
  # if chunk 1 ends at character 100, chunk 2 will start at character 
  # Similarity metric 
  # Distance measure used to compare vectors
  # 'cosine' measures the cosine similarity between vectors
  splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
  chunks = splitter.split_documents(data)

  # create the index
  # dimension : number of vectors for the vector space
  pinecone.create_index(index_name, dimension=4096, metric='cosine')

  global vectorstore 
  # Index the text chunks into Pinecone 
  vectorstore = Pinecone.from_documents(chunks, embeddings, index_name=index_name)


#Add document inside Pinecone
def document_to_index(data, chunk_size): #file_path
  
  index_name = PINECONE_INDEX
  
  #Splitt the document into chunks
  # chunk_size : maximum size of text chunk in number of characters
  # chunk_overlap : pecifies the number of overlapping characters between adjacent chunks.
  # if chunk 1 ends at character 100, chunk 2 will start at character 
  # Similarity metric 
  # Distance measure used to compare vectors
  # 'cosine' measures the cosine similarity between vectors
  splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
  chunks = splitter.split_documents(data)

  global vectorstore 
  # Index the text chunks into Pinecone 
  vectorstore.add_documents(chunks)

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(document_bytes, file_path):

    # Determine the file extension based on the first 4 bytes of the decompressed bytes object
    file_type = document_bytes[:4]

    if file_type == b'%PDF':
        # Load the PDF bytes object
        loader = PyPDFLoader(file_path) 
    elif file_type == b'PK\x03\x04':
        # Load the DOCX bytes object
        loader = Docx2txtLoader(file_path) 
    elif isinstance(file_type, bytes):
        # Load the TXT bytes object
        loader = TextLoader(file_path) 
    else:
        return None
    
    data = loader.load()
    return data


#Create the llm that will be used for the RAG
def create_chat_llm(temperature=0.75):

  # Retrieve the pinecone index direclty to Pinecone server without uploading one more time
  global index_created
  if not index_created:
      global vectorstore
      vectorstore = Pinecone.from_existing_index(PINECONE_INDEX,embeddings)
      
  global chat_llm
  global llm
  global conversation_summary_buffer_memory

  # verbose : Whether to print out response text
  # temperature : A non-negative float that tunes the degree of randomness in generation.
  # Higher values give more diverse, creative responses
  # Lower values give more focused, deterministic responses
  chat_llm = ChatCohere(
    streaming=True,
    verbose=False,
    temperature=temperature,
    cohere_api_key=COHERE_API_KEY)
  # combines the two ideas. It keeps a buffer of recent interactions in memory, 
  # but rather than just completely flushing old interactions it compiles them into a summary and uses both. 
  # It uses token length rather than number of interactions to determine when to flush interactions.
  conversation_summary_buffer_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=3000)

  global llm_created
  llm_created = True



#Before sending the qestion to the LLM, we will update the query with the chat history in order to be able to ask on previous question
def contextualize_question(question: str):
    global chat_history
    global conversation_summary_buffer_memory

    template = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. 
    Do NOT responding with clarifying questions or sentence and keep the focus on providing the requested information. \
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is. \
    
    chat_history:
    {chat_history}

    Question:
    {question}"""

    prompt = PromptTemplate(
        input_variables=['chat_history', 'question'],
        template=template
    )

    # load the history
    history = conversation_summary_buffer_memory.load_memory_variables({})
    #print(history)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    contextualize_question = llm_chain.run({'chat_history': history, 'question': question}) #chat_history
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
    final_prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Contexts:
    {source_knowledge}

    Query: {contestualized_question}"""
    #print(final_prompt)

    return final_prompt, results

# generate the response of the question
async def send_question_rag(question: str, top_research: int, temperature: float) -> AsyncIterable[str]:

  if llm_created :
    
    global callback
    # Callback to handle streaming response
    callback = AsyncIteratorCallbackHandler()

    global chat_llm
    # Set callback on LLM
    chat_llm.callbacks = [callback]

    #Set temperature on LLM
    chat_llm.temperature = temperature
    #print("chat_llm.temperature : ", chat_llm.temperature)

    # Contextualize the question
    contestualize_question = contextualize_question(question)

    #Final prompt that we will give to the LLM
    final_prompt, results = final_augment_prompt(contestualize_question, top_research)
    
    #Send the retrieving chunks
    report = []
    index = 1
    for doc in results:
      report.append(f"chunk {index} : {doc.page_content}\n\n")
      index += 1
    result = "\n".join(report).strip()
    yield result
    

    prompt = HumanMessage(content=final_prompt)

    # Generate response asynchronously - run in the background
    task = asyncio.create_task(
          chat_llm.agenerate([[prompt]]) 
      )

    try:  
          # Stream response
          # First of all, our LLM is streaming his response so we will get continious tokens through our callback
          # When a new token is generated and send it to us via the callback of our LLM
          # We will yield (=send) it to the client side, the request of the client is in streaming. 

          # callback.aiter() streams each token as they are generated
          async for token in callback.aiter():
              # sends each token to the client side since the request is streaming
              yield token
    except Exception as e:
          print(f"Caught exception: {e}")
    finally:
          callback.done.set()
    # Get full response when done
    response = await task
    task.cancel() # cancel task

    #update the chat history
    global chat_history
    global conversation_summary_buffer_memory
    chat_history.extend([HumanMessage(content=question), AIMessage(content=response.generations[0][0].text)] )
    conversation_summary_buffer_memory.save_context({"input": question}, {"output": response.generations[0][0].text})
    
  else:
     yield "LLM not generated"

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


@app.post("/index_document")
async def index_document(document: DocumentStr):
    #global vectorstore
    global index_created

    decoded = base64.b64decode(document.content) # bytes
    decompressed = gzip.decompress(decoded) # bytes
    #print(decompressed)

    # Write bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        fp.write(decompressed) 
        file_path = fp.name
        
        data = load_document(decompressed, file_path)

    # Create or update the index using the data
    if not index_created:
        create_index(data, document.chunk_size)
    else:
        document_to_index(data, document.chunk_size)

    # Delete temporary file
    os.remove(file_path)

    index_created = True
    
    return True
'''
@app.post("/create_llm")
async def create_llm(options : Options):
   create_chat_llm(options.temperature)
   return True
'''
   


@app.get("/stream_chat")
async def stream_chat(question: Question):
  async with mutex:

    if not get_llm():
        create_chat_llm()
    generator = send_question_rag(question.question,question.top_research, question.temperature)
    return StreamingResponse(generator, media_type="text/event-stream")
























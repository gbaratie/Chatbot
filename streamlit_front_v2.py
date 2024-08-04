import requests

import gzip
import base64
import json

import streamlit as st

index_created = False
URL = "http://localhost:8000/"

def print_request(r, *args, **kwargs):
  print(r.request.body) #verify the request

def getIndexCreated():
   url_get_index = f"{URL}get_index"
   return requests.get(url_get_index) #verify if the index inside Pinecone is created

st.title("💬 Chatbot") 


st.sidebar.header('Workflow')

st.sidebar.markdown('## Step 1 - upload the document')

# Upload document
doc_file = st.sidebar.file_uploader("Upload a document") 
if doc_file : 
  #Transform the content of the document before sending to the server
  document_str = doc_file.read() # already in bytes
  compressed = gzip.compress(document_str) 
  b64_encoded = base64.b64encode(compressed) # bytes object
  data = {"content": b64_encoded.decode()} # bytes in string
  #print(data)
  #Send the document
  url_create_index = f"{URL}create_index"
  response = requests.post(url_create_index, data=json.dumps(data)) # VERIFY REQUEST : , hooks={'response': print_request}
  
  if response :
      
      index_created = True
      st.sidebar.write("<p style='color:#CCFFCC'>Index created ! Now generate the LLM - PLEASE don't forget to remove the file uploaded</p>", unsafe_allow_html=True)
  else :
     st.sidebar.write("<p style='color:#FF0038'> ERROR please RETRY</p>", unsafe_allow_html=True)
 
st.sidebar.markdown('## Step 2 - Generate the LLM') 
temperature = st.sidebar.slider("Temperature : controls the 'randomness' of the chatbot's responses during text generation. Higher values give more diverse, creative responses. Lower values give more focused, deterministic responses.", min_value=0.01, max_value=1.0, value=0.01)

if st.sidebar.button('Generate LLM'):
   if getIndexCreated() :
      url_create_llm = f"{URL}create_llm"
      data = {"temperature": temperature}
      response = requests.post(url_create_llm, data=json.dumps(data))
      st.sidebar.write("<p style='color:#CCFFCC'>LLM generated !</p>", unsafe_allow_html=True)
      
   else :
    st.sidebar.write("<p style='color:#FF0038'>An index needs to be created</p>", unsafe_allow_html=True)


st.sidebar.markdown('## Step 3 - The number of N most relevant chunks of information in the vector database')
top_research = st.sidebar.slider("You can change it after each question ", min_value=1, max_value=50, value=1)

    

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])



cache_answer = None
if prompt := st.chat_input():

  st.session_state.messages.append({"role": "user", "content": prompt})
  st.chat_message("user").write(prompt)

  res_box = st.empty()
  
  report = []

  url = f"{URL}stream_chat"
    
  data = {"question": prompt, "top_research": top_research}

  headers = {"Content-type": "application/json"}

  #with st.chat_message("assistant"):
  # stream = true : stream and process the response from the API in chunks rather than loading the entire response into memory at once.
  # iter_content(): iterate over the response data in chunks of predefined sizes (1024 bytes). We can process each chunk separately rather than the whole response together.
  with requests.get(url, data=json.dumps(data), headers=headers, stream=True) as r: # VERIFY REQUEST : , hooks={'response': print_request}
      for chunk in r.iter_content(1024):
              
        #join method to concatenate the elements of the list 
        # into a single string, 
        # then strip out any empty strings
        report.append(chunk.decode('utf-8'))
        result = "".join(report).strip()    
        res_box.chat_message("assistant").write(f'*{result}*')

        cache_answer = result

  st.session_state.messages.append({"role": "assistant", "content": cache_answer})






        
        
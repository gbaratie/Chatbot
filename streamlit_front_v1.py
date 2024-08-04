import requests

import gzip
import base64
import json

import streamlit as st

index_created = False
URL = "http://localhost:8000/"

def print_request(r, *args, **kwargs):
  print(r.request.body)

def getIndexCreated():
   url_get_index = f"{URL}get_index"
   return requests.get(url_get_index)

st.header("Chatbot")

st.sidebar.header('Options')

temperature = st.sidebar.slider("Temperature", min_value=0.01, max_value=1.0, value=0.01)

if st.sidebar.button('Generate LLM'):
   if getIndexCreated() :
      url_create_llm = f"{URL}create_llm"
      data = {"temperature": temperature}
      response = requests.post(url_create_llm, data=json.dumps(data))
      st.write("<p style='color:#CCFFCC'>LLM generated !</p>", unsafe_allow_html=True)
      
   else :
    st.write("<p style='color:#FF0038'>An index needs to be created</p>", unsafe_allow_html=True)
    

# Upload document
doc_file = st.file_uploader("Upload a document") 
if doc_file : 
  #Transform the content of the document before sending to the server
  document_str = doc_file.read()
  compressed = gzip.compress(document_str) 
  b64_encoded = base64.b64encode(compressed)
  data = {"content": b64_encoded.decode()}

  #Send the document
  url_create_index = f"{URL}create_index"
  response = requests.post(url_create_index, data=json.dumps(data)) # VERIFY REQUEST : , hooks={'response': print_request}
  
  if response :
      
      index_created = True
      st.write("<p style='color:#CCFFCC'>Index created ! Now generate the LLM - PLEASE don't forget to remove the file uploaded</p>", unsafe_allow_html=True)
  else :
     st.write("<p style='color:#FF0038'> ERROR please RETRY</p>", unsafe_allow_html=True)
     


question = st.text_input("Enter your question :")

top_research = st.slider("Top research", min_value=1, max_value=100, value=1)

if st.button("Ask Chatbot"):
    
    res_box = st.empty()
    report = []

    st.markdown("----")
    
    url = f"{URL}stream_chat"
    
    data = {"question": question, "top_research": top_research}

    headers = {"Content-type": "application/json"}

    with requests.post(url, data=json.dumps(data), headers=headers, stream=True) as r: # VERIFY REQUEST : , hooks={'response': print_request}
        for chunk in r.iter_content(1024):
            
            #join method to concatenate the elements of the list 
            # into a single string, 
            # then strip out any empty strings
            report.append(chunk.decode('utf-8'))
            result = "".join(report).strip()    
            res_box.markdown(f'*{result}*')
    st.markdown("----")




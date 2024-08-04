# Launch Chatbot Application

## Pre-requisites
- Python and pyvenv (or Python3)
- API KEY for Cohere LLM (https://dashboard.cohere.com/)
- API KEY for Pinecone  (https://app.pinecone.io/)
- Name of your Pinecone environment of your API KEY
- Put a name for your Pinecone index (whatever you want)


## Create Cohere API KEY
You can get one by signing up https://app.cohere.io/dashboard. Visit the Cohere dashboard to retrieve your API key. If you haven't previously connected a session, you should see it in the main screen. Otherwise, you can find it in the settings page.

## Create Pinecone API KEY

https://youtu.be/_gC9wWWBjmY?si=Z06gAoH7miIhoEKk

## Steps  
1. Create Python virtual environment
    ```
    python3 -m venv {name of your venv}
    ```
    Or
    ```
    python -m venv {name of your venv}
    ```
    
2. Activate virtual environment
    ```
    source {name of your venv}/bin/activate
    ```

You should see the name of your environment on the left of your terminal

3. Install requirements 
    ```
    pip install -r requirements_v1.txt
    ```
    or
    ```
    pip install -r requirements_v2.py
    ```
    or
    ```
    pip install -r requirements_v3.py
    ```
4. Configure .env file
    - Add Cohere API key
    - Add Pinecone API key
    - Add Pinecone environment of your index
    
5. Launch API server

Be sure to launch under your python environment
You should see the name of your environment on the left of your terminal

    ``` 
    uvicorn server_v1:app --reload
    ```
    or
    ``` 
    uvicorn server_v2:app --reload
    ```
    or
    ``` 
    uvicorn server_v3:app --reload
    ```
6. Launch frontend

Be sure to launch under your python environment
You should see the name of your environment on the left of your terminal

    ```
    streamlit run streamlit_front_v1.py
    ```
    or
    ```
    streamlit run streamlit_front_v2.py
    ```
     or
    ```
    streamlit run streamlit_front_v3.py
    ```
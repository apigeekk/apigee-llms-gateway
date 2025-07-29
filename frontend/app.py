import streamlit as st
import logging
import requests
from functools import partial
import config  # Import your config file

# Set page config
st.set_page_config(page_title="LLMs Gateway Demo", layout="wide")

# Custom CSS to style the app
st.markdown("""
<style>
    body {
        background-color: #ffffff;
        color: #333333;
    }
    .stApp {
        background-color: #fff;
    }
    .main {
        background-color: #e0e0e0;
    }
    .stTextInput, .stTextArea {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    .stTextArea:focus {
        border-color: #4285F4 !important;
        box-shadow: 0 0 0 1px #4285F4 !important;
    }
    .stButton>button {
        background-color: #4285F4;
        color: #ffffff;
        font-size: 16px;
        padding: 10px 24px;
    }
    .stSelectbox {
        background-color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }

    .stTextArea:hover, .stButton>button:hover {
        border-color: #4285F4 !important;
        box-shadow: 0 0 0 1px #4285F4 !important;
    }
    h1 {
        color: #1a73e8 !important;
        font-weight: bold;
    }

</style>
""", unsafe_allow_html=True)

# Main content
st.markdown("<h1 style='text-align: center;'> LLMs Gateway Demo (powered by Apigee)</h1>", unsafe_allow_html=True)

# LLM selection
llm_options = ["gemini-1.5-flash", "gpt-4o-mini", "claude-3-5-sonnet-20240620"] 
selected_llm = st.selectbox("Select an LLM", llm_options, key="llm_select")

# Application Settings
api_product_options = ["Basic LLM API Product", "Advanced LLM API Product"]
selected_api_product = st.selectbox("Select an API Product", api_product_options, key="api_product_select")

# User input
user_input = st.text_area(
    "Enter your prompt here...",
    height=300,
    key="standard_textarea"
)

# Submit button
submit_button = st.button("Submit")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def call_api(prompt, model, api_key, api_endpoint, path_suffix):
    """
    Calls the Apigee endpoint to interact with the specified LLM model.
    """
    url = api_endpoint + path_suffix
    headers = {"apikey": api_key}
    data = {"prompt": prompt, "model": model}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # This will raise an HTTPError if the status code is 4xx or 5xx

        result = response.json()
        return result

    except requests.exceptions.HTTPError as http_err:
        try:
            error_message = response.json().get("error", http_err)  # Try to extract the error message
        except ValueError:
            error_message = f"API call failed >> {error_message}"  # Use a generic message if JSON decoding fails
        logging.error(f"HTTP error occurred: {http_err}, Message: {error_message}")
        st.error(f"HTTP error occurred: check the logs for details.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"API call failed: {e}")
        st.error(f"API call failed: check the logs for details.")
        return None


if submit_button and user_input and len(user_input) > 2:
    model_type = selected_llm.split("-")[0]
    model_paths = {
        "gemini": "/gemini",
        "gpt": "/openai",
        "claude": "/anthropic"
    }

    path_suffix = model_paths.get(model_type)

    # Get API keys and endpoint from config.py
    api_endpoint = config.apigee_endpoint
    if selected_api_product == "Basic LLM API Product":
        api_key = config.apigee_api_key_basic
    elif selected_api_product == "Advanced LLM API Product":
        api_key = config.apigee_api_key_secure
    else:
        st.error(f"API key for {selected_api_product} is missing. Please check your config.py file.")
        api_key = None

    # Proceed if the API key is available
    if api_key and api_endpoint:
        with st.spinner(f"Generating response using {model_type}..."):
            model_response = call_api(user_input, selected_llm, api_key, api_endpoint, path_suffix)
            if model_response:
                st.markdown(f"**Total tokens used:** {model_response.get('tokens_count', 'N/A')}")
                st.code(model_response['response'], language="markdown")
    elif not api_endpoint:
        st.error("Apigee Endpoint URL is missing. Please check your config.py file.")
elif submit_button:
    st.warning("Please enter a prompt before submitting.")


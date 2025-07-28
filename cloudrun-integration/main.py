import os
from flask import Flask, request, jsonify
import google.generativeai as genai
from openai import OpenAI
import anthropic

import requests
import logging
import google.auth.transport.requests

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/gemini', methods=['POST'])
def gemini_request_handler():
    """
    Handles requests to the /gemini endpoint for interacting with the Gemini API.
    Expects prompt and model in the JSON payload from the frontend.
    Expects API key, max_output_tokens, and model_armor_template_url as HTTP headers
    injected by Apigee.
    Returns:
        A JSON response with the Gemini model's output and token count,
        or an error message if the API call fails.
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Extract all parameters from their expected sources
        prompt = request_data.get('prompt')
        model = request_data.get('model')

        max_output_tokens = request.headers.get('X-Max-Output-Tokens')
        model_armor_template_url = request.headers.get('X-Model-Armor-Template-Url')
        api_key = request.headers.get('api_key')

        # Consolidated validation for all required parameters
        missing_params = []
        if not prompt: missing_params.append('prompt (JSON)')
        if not model: missing_params.append('model (JSON)')
        if not max_output_tokens: missing_params.append('X-Max-Output-Tokens (Header)')
        if not model_armor_template_url: missing_params.append('X-Model-Armor-Template-Url (Header)')
        if not api_key: missing_params.append('api_key (Header)')

        if missing_params:
            if 'api_key (Header)' in missing_params:
                logger.error(f"API key header missing for Gemini. Ensure Apigee is injecting it correctly. Missing: {missing_params}")
                return jsonify({"error": "Authentication required: API key not provided by proxy."}), 401
            else:
                logger.error(f"Missing required parameters in Gemini request: {missing_params}")
                return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400

        # Prompt Sanitization check (commented out as per original, but functional)
        # model_armor_response = handle_prompt_sanitization(prompt, model_armor_template_url)
        # if model_armor_response:
        #    return model_armor_response

        # Get the response from Gemini
        response = get_gemini_response(
            prompt, model, api_key, max_output_tokens
        )
        return response
    except Exception as e:
        logger.error(f"Error in Gemini request handler: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error processing Gemini request: {str(e)}"}), 500


def get_gemini_response(prompt, model, api_key, max_tokens):
    """
    Sends a request to the Gemini API with the given prompt and parameters.
    Args:
        prompt: The text prompt to send to the Gemini model.
        model: The name of the Gemini model to use.
        api_key: The API key for authenticating with the Gemini API.
        max_tokens: The maximum number of tokens to generate in the response.
    Returns:
        A JSON response with the generated text and token count,
        or an error message if the API call fails.
    """
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": int(max_tokens),
        "response_mime_type": "text/plain",
    }

    try:
        model_instance = genai.GenerativeModel( # Renamed 'model' to 'model_instance' to avoid shadowing
            model_name=model,
            generation_config=generation_config,
        )
        response = model_instance.generate_content(prompt)
        usage_metadata = response.usage_metadata

        resp = {
            "prompt": prompt,
            "response": response.text,
            "tokens_count": usage_metadata.total_token_count
        }
        return jsonify(resp)

    except Exception as e:
        logger.error(f"Error calling Gemini API in get_gemini_response: {e}", exc_info=True)
        return jsonify({"error": f"Error calling Gemini API: {str(e)}"}), 500

@app.route('/openai', methods=['POST'])
def gpt_request_handler():
    """
    Handles requests to the /openai endpoint for interacting with OpenAI's GPT models.
    Expects prompt and model in the JSON payload from the frontend.
    Expects API key, max_output_tokens, and model_armor_template_url as HTTP headers
    injected by Apigee.
    Returns:
        A JSON response with the GPT model's output and token usage,
        or an error message if the API call fails.
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Extract all parameters from their expected sources
        prompt = request_data.get('prompt')
        model = request_data.get('model')

        max_output_tokens = request.headers.get('X-Max-Output-Tokens')
        model_armor_template_url = request.headers.get('X-Model-Armor-Template-Url')
        api_key = request.headers.get('api_key')

        # Consolidated validation for all required parameters
        missing_params = []
        if not prompt: missing_params.append('prompt (JSON)')
        if not model: missing_params.append('model (JSON)')
        if not max_output_tokens: missing_params.append('X-Max-Output-Tokens (Header)')
        if not model_armor_template_url: missing_params.append('X-Model-Armor-Template-Url (Header)')
        if not api_key: missing_params.append('api_key (Header)')

        if missing_params:
            if 'api_key (Header)' in missing_params:
                logger.error(f"API key header missing for OpenAI. Ensure Apigee is injecting it correctly. Missing: {missing_params}")
                return jsonify({"error": "Authentication required: API key not provided by proxy."}), 401
            else:
                logger.error(f"Missing required parameters in OpenAI request: {missing_params}")
                return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400

        # Prompt Sanitization check (commented out as per original, but functional)
        # model_armor_response = handle_prompt_sanitization(prompt, model_armor_template_url)
        # if model_armor_response:
        #    return model_armor_response

        response = get_chatgpt_response(prompt, model, api_key, max_output_tokens)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error calling ChatGPT API: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error processing ChatGPT request: {str(e)}"}), 500


def get_chatgpt_response(prompt, model, api_key, max_tokens):
    """
    Sends a request to the OpenAI ChatGPT API with the given prompt and parameters.
    Args:
        prompt: The text prompt to send to the GPT model.
        model: The name of the GPT model to use.
        api_key: The API key for authenticating with the OpenAI API.
        max_tokens: The maximum number of tokens to generate in the response.
    Returns:
        A dictionary containing the generated text and token usage.
    """
    client = OpenAI(api_key=api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=model,
        max_tokens=int(max_tokens)
    )

    message = chat_completion.choices[0].message.content
    total_tokens = chat_completion.usage.total_tokens

    resp = {
            "prompt": prompt,
            "response": message,
            "tokens_count": total_tokens
    }
    return resp


@app.route('/anthropic', methods=['POST'])
def claude_request_handler():
    """
    Handles requests to the /anthropic endpoint for interacting with Anthropic's Claude models.
    Expects prompt and model in the JSON payload from the frontend.
    Expects API key, max_output_tokens, and model_armor_template_url as HTTP headers
    injected by Apigee.
    Returns:
        A JSON response with the Claude model's output and token usage,
        or an error message if the API call fails.
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Extract all parameters from their expected sources
        prompt = request_data.get('prompt')
        model = request_data.get('model')

        max_output_tokens = request.headers.get('X-Max-Output-Tokens')
        model_armor_template_url = request.headers.get('X-Model-Armor-Template-Url')
        api_key = request.headers.get('api_key')

        # Consolidated validation for all required parameters
        missing_params = []
        if not prompt: missing_params.append('prompt (JSON)')
        if not model: missing_params.append('model (JSON)')
        if not max_output_tokens: missing_params.append('X-Max-Output-Tokens (Header)')
        if not model_armor_template_url: missing_params.append('X-Model-Armor-Template-Url (Header)')
        if not api_key: missing_params.append('api_key (Header)')

        if missing_params:
            if 'api_key (Header)' in missing_params:
                logger.error(f"API key header missing for Anthropic. Ensure Apigee is injecting it correctly. Missing: {missing_params}")
                return jsonify({"error": "Authentication required: API key not provided by proxy."}), 401
            else:
                logger.error(f"Missing required parameters in Anthropic request: {missing_params}")
                return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400

        # Prompt Sanitization check (commented out as per original, but functional)
        # model_armor_response = handle_prompt_sanitization(prompt, model_armor_template_url)
        # if model_armor_response:
        #    return model_armor_response

        response = get_calude_response(prompt, model, api_key, max_output_tokens)
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error calling Claude API: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error processing Claude request: {str(e)}"}), 500


def get_calude_response(prompt, model, api_key, max_tokens):
    """
    Sends a request to the Anthropic Claude API with the given prompt and parameters.
    Args:
        prompt: The text prompt to send to the Claude model.
        model: The name of the Claude model to use.
        api_key: The API key for authenticating with the Anthropic API.
        max_tokens: The maximum number of tokens to generate in the response.
    Returns:
        A dictionary containing the generated text and token usage.
    """
    client = anthropic.Anthropic(
        api_key=api_key,
    )
    message = client.messages.create(
        model=model,
        max_tokens= int(max_tokens),
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    resp = {
            "prompt": prompt,
            "response": message.content[0].text,
            "tokens_count": int (message.usage.input_tokens) + int(message.usage.output_tokens),
    }
    
    return resp
    
@app.route('/sanitizePrompt', methods=['POST'])
def prompt_sanitization_handler():
    """
    Sanitizes a user prompt using the ModelArmor API.
    Receives a JSON payload containing the prompt.
    model_armor_template_url is expected from an HTTP header or JSON payload.
    Returns:
      A JSON response with the sanitization outcome ('MATCH_FOUND' or 'NO_MATCH_FOUND') 
      and a corresponding message.
    """
    try:
        logger.info("*******Request ******")
        logger.info(request)
        
        logger.info("*******Request Data ******")
        logger.info(request.get_json())
        logger.info("*************************")

        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        prompt = request_data.get('prompt')
        # This endpoint's model_armor_template_url source might vary.
        # Prioritize header if Apigee injects it for this specific endpoint.
        # Fallback to payload if it's sometimes sent directly.
        model_armor_template_url = request.headers.get('X-Model-Armor-Template-Url') or request_data.get('model_armor_template_url')
        
        # Validate required parameters for this handler
        missing_params = []
        if not prompt: missing_params.append('prompt (JSON)')
        if not model_armor_template_url: missing_params.append('X-Model-Armor-Template-Url (Header/JSON)')
        
        if missing_params:
             logger.error(f"Missing required parameters for sanitizePrompt: {missing_params}")
             return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400
           
        logger.info("********Inputs******")
        logger.info(prompt)
        logger.info(model_armor_template_url)
        
           
        logger.info("*******Sending request to ModelArmor API:******")
        logger.info(f"  prompt: {prompt}")
        logger.info(f"  model_armor_template_url: {model_armor_template_url}")

        
        id_token = get_id_token() # Uses Google-managed auth, which is appropriate for GCP services
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}'
        }
        modelarmor_request_data = {'user_prompt_data': {'text': prompt}}

        logger.info("Sending request to ModelArmor API:")
        logger.info(f"  URL: {model_armor_template_url}")
        logger.info(f"  Headers: {headers}")
        logger.info(f"  Data: {modelarmor_request_data}")

        modelarmor_response = requests.post(model_armor_template_url, headers=headers, json=modelarmor_request_data)
        modelarmor_response.raise_for_status()

        response_data = modelarmor_response.json()
        logger.info(f"ModelArmor response: {response_data}")
        
        result = {}
        if response_data.get('sanitizationResult', {}).get('filterMatchState') == 'MATCH_FOUND':
            matching_filters = find_matching_filter_types(response_data)
            
            result = {
                "verdict": "MATCH_FOUND",
                "message": f"Prompt failed sanity check due to the following filters: {', '.join(matching_filters)}"
            }
        else:
            result = {
                "verdict": "NO_MATCH_FOUND",
                "message": "Prompt passed sanity check."
            }
            
        return result

    except Exception as e:
        logger.error(f"Error interacting with ModelArmor: {e}", exc_info=True)
        result = {
                "verdict": "ERROR",
                "message": f"Error interacting with ModelArmor : {e}"
            }
        return result

def find_matching_filter_types(response_data):
    """
    Parses the ModelArmor API response to identify the types of filters that matched the prompt.
    Args:
      response_data: The JSON response data from the ModelArmor API.
    Returns:
      A list of filter type values that resulted in a match, or an empty list if no match is found 
      or the response data is invalid.
    """
    matching_filters = []
    try:
        sanitization_result = response_data['sanitizationResult']
        for filter_result in sanitization_result['filterResults']:
            for filter_name, filter_data in filter_result.items():
                if 'inspectResult' in filter_data:
                    filter_data = filter_data['inspectResult']
                if filter_data.get('matchState') == 'MATCH_FOUND':
                    filter_type = None
                    if 'raiFilterTypeResults' in filter_data:
                        for rai_type_result in filter_data['raiFilterTypeResults']:
                            if rai_type_result.get('matchState') == 'MATCH_FOUND':
                                filter_type = rai_type_result.get('filterType')
                                break
                    if not filter_type:
                        filter_type = filter_name.replace('FilterResult', '')
                    matching_filters.append(filter_type)
        return matching_filters
    except (KeyError, TypeError) as e:
        logger.error(f"Error parsing ModelArmor response data format: {e}", exc_info=True)
        return []


def get_id_token():
    """
    Retrieves the ID token for the current Cloud Run service account 
    to authenticate requests to the ModelArmor API.
    """
    creds, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds.token

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


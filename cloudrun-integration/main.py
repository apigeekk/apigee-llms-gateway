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
    Expects API key and max_output_tokens as HTTP headers injected by Apigee.
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        prompt = request_data.get('prompt')
        model = request_data.get('model')
        max_output_tokens = request.headers.get('X-Max-Output-Tokens')
        api_key = request.headers.get('X-Api-Key')

        missing_params = []
        if not prompt: missing_params.append('prompt (JSON)')
        if not model: missing_params.append('model (JSON)')
        if not max_output_tokens: missing_params.append('X-Max-Output-Tokens (Header)')
        if not api_key: missing_params.append('X-Api-Key (Header)')

        if missing_params:
            if 'X-Api-Key (Header)' in missing_params:
                logger.error(f"API key header missing for Gemini. Ensure Apigee is injecting it correctly. Missing: {missing_params}")
                return jsonify({"error": "Authentication required: API key not provided by proxy."}), 401
            else:
                logger.error(f"Missing required parameters in Gemini request: {missing_params}")
                return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400

        response = get_gemini_response(prompt, model, api_key, max_output_tokens)
        return response
    except Exception as e:
        logger.error(f"Error in Gemini request handler: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error processing Gemini request: {str(e)}"}), 500


def get_gemini_response(prompt, model, api_key, max_tokens):
    """
    Sends a request to the Gemini API with the given prompt and parameters.
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
        model_instance = genai.GenerativeModel(
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
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        prompt = request_data.get('prompt')
        model = request_data.get('model')
        max_output_tokens = request.headers.get('X-Max-Output-Tokens')
        api_key = request.headers.get('X-Api-Key')

        missing_params = []
        if not prompt: missing_params.append('prompt (JSON)')
        if not model: missing_params.append('model (JSON)')
        if not max_output_tokens: missing_params.append('X-Max-Output-Tokens (Header)')
        if not api_key: missing_params.append('X-Api-Key (Header)')

        if missing_params:
            if 'X-Api-Key (Header)' in missing_params:
                logger.error(f"API key header missing for OpenAI. Ensure Apigee is injecting it correctly. Missing: {missing_params}")
                return jsonify({"error": "Authentication required: API key not provided by proxy."}), 401
            else:
                logger.error(f"Missing required parameters in OpenAI request: {missing_params}")
                return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400

        response = get_chatgpt_response(prompt, model, api_key, max_output_tokens)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error calling ChatGPT API: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error processing ChatGPT request: {str(e)}"}), 500


def get_chatgpt_response(prompt, model, api_key, max_tokens):
    """
    Sends a request to the OpenAI ChatGPT API with the given prompt and parameters.
    """
    client = OpenAI(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=int(max_tokens)
    )

    message = chat_completion.choices[0].message.content
    total_tokens = chat_completion.usage.total_tokens
    return {
        "prompt": prompt,
        "response": message,
        "tokens_count": total_tokens
    }


@app.route('/anthropic', methods=['POST'])
def claude_request_handler():
    """
    Handles requests to the /anthropic endpoint for interacting with Anthropic's Claude models.
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        prompt = request_data.get('prompt')
        model = request_data.get('model')
        max_output_tokens = request.headers.get('X-Max-Output-Tokens')
        api_key = request.headers.get('X-Api-Key')

        missing_params = []
        if not prompt: missing_params.append('prompt (JSON)')
        if not model: missing_params.append('model (JSON)')
        if not max_output_tokens: missing_params.append('X-Max-Output-Tokens (Header)')
        if not api_key: missing_params.append('X-Api-Key (Header)')

        if missing_params:
            if 'X-Api-Key (Header)' in missing_params:
                logger.error(f"API key header missing for Anthropic. Ensure Apigee is injecting it correctly. Missing: {missing_params}")
                return jsonify({"error": "Authentication required: API key not provided by proxy."}), 401
            else:
                logger.error(f"Missing required parameters in Anthropic request: {missing_params}")
                return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400

        response = get_calude_response(prompt, model, api_key, max_output_tokens)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error calling Claude API: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error processing Claude request: {str(e)}"}), 500


def get_calude_response(prompt, model, api_key, max_tokens):
    """
    Sends a request to the Anthropic Claude API with the given prompt and parameters.
    """
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=int(max_tokens),
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "prompt": prompt,
        "response": message.content[0].text,
        "tokens_count": int(message.usage.input_tokens) + int(message.usage.output_tokens),
    }


@app.route('/sanitizePrompt', methods=['POST'])
def prompt_sanitization_handler():
    """
    Sanitizes a user prompt using the ModelArmor API.
    """
    try:
        request_data = request.get_json()
        prompt = request_data['prompt']
        model_armor_template_url = request_data['model_armor_template_url']

        id_token = get_id_token()
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}'
        }
        modelarmor_request_data = {'user_prompt_data': {'text': prompt}}
        modelarmor_response = requests.post(model_armor_template_url, headers=headers, json=modelarmor_request_data)
        modelarmor_response.raise_for_status()

        response_data = modelarmor_response.json()
        if response_data.get('sanitizationResult', {}).get('filterMatchState') == 'MATCH_FOUND':
            matching_filters = find_matching_filter_types(response_data)
            return {
                "verdict": "MATCH_FOUND",
                "message": f"Prompt failed sanity check due to the following filters: {', '.join(matching_filters)}"
            }
        else:
            return {"verdict": "NO_MATCH_FOUND", "message": "Prompt passed sanity check."}
    except Exception as e:
        return {"verdict": "ERROR", "message": f"Error interacting with ModelArmor : {e}"}


def find_matching_filter_types(response_data):
    """
    Parses the ModelArmor API response to identify the types of filters that matched the prompt.
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
    except (KeyError, TypeError):
        return []


def get_id_token():
    """
    Retrieves the ID token for the current Cloud Run service account.
    """
    creds, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds.token


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

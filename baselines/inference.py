import json
import os
import time
import logging
from functools import wraps
from dotenv import load_dotenv
from together import Together
# Import provider SDKs
import openai
import requests
import anthropic
from google import genai
from sglang.utils import terminate_process, wait_for_server
from sglang.utils import launch_server_cmd
from transformers import AutoTokenizer
from openai.types.shared_params import Reasoning
# Load environment variables from .env file with override
load_dotenv(override=True)

# Together API does not have an official SDK, use requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_on_error(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self.lock = None
        try:
            from threading import Lock
            self.lock = Lock()
        except ImportError:
            self.lock = None

    def acquire(self):
        if self.lock:
            self.lock.acquire()
        try:
            now = time.time()
            time_passed = now - self.last_update
            if time_passed >= 60:
                self.tokens = self.requests_per_minute
                self.last_update = now
            if self.tokens > 0:
                self.tokens -= 1
                return True
            wait_time = 60 - time_passed
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_update = time.time()
            self.tokens = self.requests_per_minute - 1
            return True
        finally:
            if self.lock:
                self.lock.release()

def rate_limit(requests_per_minute: int):
    limiter = RateLimiter(requests_per_minute)
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.acquire()
            return func(*args, **kwargs)
        return wrapper
    return decorator

class AIClient:
    def generate_completion(self, prompt, **kwargs):
        raise NotImplementedError

class OpenAIClient(AIClient):
    def __init__(self, api_key=None, model_name=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model_name = model_name

    @retry_on_error()
    def generate_completion(self, prompt, **kwargs):
        model = kwargs.get("model", self.model_name or "gpt-3.5-turbo")
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 2048)
        n = kwargs.get("n", 1)
        messages = kwargs.get("messages")
        presence_penalty = kwargs.get("presence_penalty", 0.0)
        frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        enable_thinking = kwargs.get("enable_thinking", False)
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        responses = []
        response_kwargs = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_tokens,
        }
        if model in ["gpt-5.1", "gpt-5.2"]:
            response_kwargs["reasoning"] = {"effort": "none"} if not enable_thinking else {"effort": "low"}
        
        if "o3" in model or "o4" in model: 
            response_kwargs["reasoning"] = {"effort": "low"} 
        
        for _ in range(n):
            response = self.client.responses.create(**response_kwargs)
            responses.append(response.output_text)
        print(responses)
        return responses

class TogetherClient(AIClient):
    def __init__(self, api_key=None, model_name=None):
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together API key not found")
        # Using Together's official client library instead of direct API calls
        self.client = Together(api_key=self.api_key)
        self.model_name = model_name
    @retry_on_error()
    def generate_completion(self, prompt, **kwargs):
        model = kwargs.get("model", self.model_name)
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 2048)
        n = kwargs.get("n", 1)
        messages = kwargs.get("messages")
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": model,
            "messages": messages,
            "n": n
        }
        if "o4" in model:
            # payload["max_completion_tokens"] = max_tokens
            # o4 does not support temperature
            pass
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
        response = self.client.chat.completions.create(**payload)
        return [choice.message.content for choice in response.choices]

class ClaudeClient(AIClient):
    def __init__(self, api_key=None, model_name=None):
        if anthropic is None:
            raise ImportError("Please install the 'anthropic' package for Claude support.")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not found")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model_name = model_name

    @retry_on_error()
    def generate_completion(self, prompt, **kwargs):
        model = kwargs.get("model", self.model_name or "claude-3-opus-20240229")
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 2048)
        n = kwargs.get("n", 1)
        messages = kwargs.get("messages")
        if messages is not None:
            # Convert OpenAI-style messages to Anthropic format
            system_prompt = ""
            user_content = ""
            for m in messages:
                if m["role"] == "system":
                    system_prompt = m["content"]
                elif m["role"] == "user":
                    user_content += m["content"] + "\n"
            prompt = user_content.strip()
        else:
            system_prompt = ""
        completions = []
        for _ in range(n):
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}]
            )
            completions.append(response.content[0].text if hasattr(response.content[0], 'text') else response.content[0]["text"])
            print(completions[-1])
        return completions

class GeminiClient(AIClient):
    def __init__(self, api_key=None, model_name=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Google Gemini API key not found")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    @retry_on_error()
    @rate_limit(requests_per_minute=15)  # Gemini's rate limit is 15 RPM
    def generate_completion(self, prompt, **kwargs):
        completions = []
        model = kwargs.get("model", self.model_name or "gemini-pro")
        for _ in range(kwargs.get("n", 1)):
            if messages := kwargs.get("messages"):
                # Convert messages to content format
                contents = []
                for msg in messages:
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append({"role": role, "parts": [{"text": msg["content"]}]})
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=kwargs.get("max_tokens", 2048),
                        temperature=kwargs.get("temperature", 0.0),
                    )
                )
            else:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=kwargs.get("max_tokens", 2048),
                        temperature=kwargs.get("temperature", 0.0),
                    )
                )
            completions.append(response.text)
                
        return completions


class SGLangClient(AIClient):
    def __init__(self, model_name, sglang_addr):
        self.sglang_addr = sglang_addr
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)

    def generate_completion(self, prompt, **kwargs):
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 2048)
        n = kwargs.get("n", 1)
        enable_thinking = kwargs.get("enable_thinking", False)

        
        sampling_params = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "n": n,
        }
        
        # Add optional sampling parameters if they differ from defaults
        has_thinking_mode = self.model_name in ['Qwen/Qwen3-8B', 'Qwen/Qwen3-4B', 'Qwen/Qwen3-14B', 'Qwen/Qwen3-32B']
        if 'messages' in kwargs:
            if has_thinking_mode:
                prompt = self.tokenizer.apply_chat_template(kwargs['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
            else:
                prompt = self.tokenizer.apply_chat_template(kwargs['messages'], tokenize=False, add_generation_prompt=True)
        else:
            if has_thinking_mode:
                prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
            else:
                prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        
        json_data = {
            "text": prompt,
            "sampling_params": sampling_params,
        }
        response = requests.post(
            f"{self.sglang_addr}/generate",
            json=json_data,
        )
        
        out = response.json()
        completions = []
        if n > 1:
            for i in range(n):
                llm_resp = out[i]['text']
                completions.append(llm_resp)
        else:
            completions.append(out['text'])
        return completions

    

    

def get_client(client_type, api_key=None, model_name=None, sglang_addr = None):
    client_type = client_type.lower()
    if client_type == "openai":
        return OpenAIClient(api_key, model_name)
    elif client_type == "together":
        return TogetherClient(api_key, model_name)
    elif client_type in ["claude", "anthropic"]:
        return ClaudeClient(api_key, model_name)
    elif client_type == "gemini":
        return GeminiClient(api_key, model_name)
    elif client_type == "sglang":
        return SGLangClient(model_name, sglang_addr)
    else:
        raise ValueError(f"Unsupported client type: {client_type}") 
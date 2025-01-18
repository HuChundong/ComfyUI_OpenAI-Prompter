import os
import json
import openai
from openai import OpenAI
from typing import List, Dict

__reload_libs__ = True

class OpenAIPromptGenerator:
    _instance = None
    _models_cache = None
    _client_cache = None
    
    def __init__(self):
        if OpenAIPromptGenerator._models_cache is None:
            self.client = None
            self.available_models = []
            self._initialize_client()
            # Cache both the models and the client
            OpenAIPromptGenerator._models_cache = self.available_models
            OpenAIPromptGenerator._client_cache = self.client
        else:
            # Restore both from cache
            self.available_models = OpenAIPromptGenerator._models_cache
            self.client = OpenAIPromptGenerator._client_cache
            
    @classmethod
    def INPUT_TYPES(cls):
        try:
            if cls._instance is None:
                cls._instance = cls()
            return {
                "required": {
                    "model": (cls._instance.available_models,),
                    "prompt_context": ("STRING", {
                        "multiline": True,
                        "default": "Generate a detailed prompt for an image generation AI"
                    }),
                    "additional_instructions": ("STRING", {
                        "multiline": True,
                        "default": "give it without any prefix, or things like \"here the prompt for:\". just give the image prompt"
                    }),
                    "max_tokens": ("INT", {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 256
                    }),
                    "temperature": ("FLOAT", {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1
                    }),
                    "seed": ("INT", {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "step": 1
                    })
                }
            }
        except Exception as e:
            print(f"Error initializing INPUT_TYPES: {str(e)}")
            return {
                "required": {
                    "error": ("STRING", {"default": "Error: OpenAI API key not found or invalid"})
                }
            }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_prompt",)  # Changed to match the key used in generate_prompt method
    FUNCTION = "generate_prompt"
    CATEGORY = "prompt"  # Changed to match README's category description

    def _initialize_client(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.available_models = ["Error: OpenAI API key not configured"]
            return
        
        try:
            # Check if we're using OpenAI v1.0.0 or newer
            if hasattr(openai, 'OpenAI'):
                self.client = OpenAI(api_key=api_key)
            else:
                # For older versions of the OpenAI package
                openai.api_key = api_key
                self.client = openai
            
            try:
                # Handle both new and old API versions
                if hasattr(self.client, 'models'):
                    models = self.client.models.list()
                else:
                    models = self.client.Model.list()
            except openai.AuthenticationError as auth_err:
                self.available_models = ["Error: Invalid API key"]
                return
            except openai.PermissionError as perm_err:
                self.available_models = ["Error: Permission denied"]
                return
            
            self.available_models = [
                model.id for model in models.data
                if "gpt" in model.id.lower()  # Filter for GPT models
            ]
            
            if not self.available_models:
                # Fallback to common models
                self.available_models = ["gpt-3.5-turbo"]
                
        except openai.APIError as api_err:
            self.available_models = ["Error: OpenAI API error occurred"]
        except openai.APIConnectionError as conn_err:
            self.available_models = ["Error: Could not connect to OpenAI"]
        except Exception as e:
            self.available_models = ["Error: Could not fetch models"]

    def generate_prompt(self, model: str, prompt_context: str, additional_instructions: str, max_tokens: int, temperature: float, seed: int):
        result = self._generate_prompt_internal(model, prompt_context, additional_instructions, max_tokens, temperature, seed)
        if isinstance(result, tuple) and len(result) == 1 and isinstance(result[0], str):
            return result
        elif isinstance(result, str):
            return (result,)
        else:
            error_message = "⚠️ Invalid response format"
            return (error_message,)
        
    def _generate_prompt_internal(self, model: str, prompt_context: str, additional_instructions: str, max_tokens: int, temperature: float, seed: int):
        if any(error in model for error in ["Error:", "quota exceeded"]):
            return ("⚠️ Please configure a valid OpenAI API key with available credits",)
            
        try:
            generated_prompt = ""
            # For OpenAI v1.0.0+
            if hasattr(self.client, 'chat'):
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a creative assistant that generates detailed, descriptive prompts for image generation AI. Focus on visual details, style, mood, and composition."},
                        {"role": "user", "content": f"{prompt_context}\n\n{additional_instructions}"}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed
                )
                generated_prompt = response.choices[0].message.content.strip()
                return (generated_prompt,)
                
            # For older versions
            else:
                response = self.client.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a creative assistant that generates detailed, descriptive prompts for image generation AI. Focus on visual details, style, mood, and composition."},
                        {"role": "user", "content": f"{prompt_context}\n\n{additional_instructions}"}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                generated_prompt = response.choices[0].message['content'].strip()
                return (generated_prompt,)
            
        except openai.RateLimitError as e:
            error_message = ("⚠️ OpenAI API quota exceeded. Please visit https://platform.openai.com/account/billing "
                           "to add credits to your account or check your payment method.")
            self.available_models = ["Error: Quota exceeded - Add credits to continue"]
            return (str(error_message),)
        except openai.APIError as e:
            if "insufficient_quota" in str(e):
                error_message = "⚠️ OpenAI API quota exceeded. Please check your billing details at https://platform.openai.com/account/billing"
            else:
                error_message = f"⚠️ OpenAI API Error: {str(e)}"
            return (error_message,)
        except Exception as e:
            error_message = f"⚠️ Error: {str(e)}"
            return (error_message,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "OpenAI Prompt Generator": OpenAIPromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAI Prompt Generator": "OpenAI Prompt Generator"
}

def reset_cache():
    OpenAIPromptGenerator._instance = None
    OpenAIPromptGenerator._models_cache = None
    OpenAIPromptGenerator._client_cache = None

reset_cache()  # Reset cache when module is reloaded 
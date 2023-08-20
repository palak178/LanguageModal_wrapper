import openai
import tiktoken
import time
from typing import List, Union
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

# Utility functions
def count_tokens(text, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def handle_error(exception, retry_count, max_retry_attempts, retry_wait_time):
    if isinstance(exception, openai.error.APIError):
        response = exception.response
        if response.status_code in {429, 500, 503} and retry_count < max_retry_attempts:
            print(f"Retrying after {retry_wait_time} seconds...")
            time.sleep(retry_wait_time)
            return True
        elif isinstance(exception, openai.error.InvalidRequestError):
            print("Invalid request error encountered:", response)
            return True
    return False

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatMessageEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ChatMessage):
            return obj.__dict__  # Convert ChatMessage object to a dictionary
        elif isinstance(obj, list) and all(isinstance(item, ChatMessage) for item in obj):
            return [item.__dict__ for item in obj]  # Convert list of ChatMessage objects to a list of dictionaries
        return super().default(obj)


class ChatModelWrapper:
    def __init__(self, api_key, use_memory=True, max_completion_token=2000, model_name="gpt-3.5-turbo"):
        openai.api_key = api_key
        self.memory = []
        self.max_completion_token = max_completion_token
        self.model_name = model_name
        self.use_memory = use_memory

    def _manage_memory(self, current_prompt_content, max_tokens):
        total_required_token = self.max_completion_token - (count_tokens(current_prompt_content) + max_tokens)
        combined_memory_tokens = sum(count_tokens(msg.content) for msg in self.memory)
        removed_messages = []

        while combined_memory_tokens > total_required_token:
            removed_message = self.memory.pop(0)
            combined_memory_tokens -= count_tokens(removed_message.content)
            removed_messages.append(removed_message)

    def _generate_prompt(self, messages, max_tokens):
        if self.use_memory:
            current_prompt_content = " ".join(message.content for message in messages)
            self._manage_memory(current_prompt_content, max_tokens)
            final_messages = self.memory + messages
        else:
            final_messages = messages
        return final_messages

    def _chat_completion(self, messages: List[ChatMessage], max_tokens: int = 128, **kwargs) -> openai.ChatCompletion:
        retry_count = 0

        total_tokens = sum(count_tokens(msg.content) for msg in messages)
        if total_tokens + max_tokens > self.max_completion_token:
            return "Error: Total tokens exceed the limit."

        while retry_count < kwargs.get("max_retry_attempts", 3):
            try:
                json_string = json.dumps(messages, cls=ChatMessageEncoder)
                # Remove 'prompt' from kwargs before passing to create method
                kwargs.pop('prompt', None)
                response = openai.ChatCompletion.create(
                    model=kwargs.get("model", self.model_name),
                    messages=json.loads(json_string),
                    max_tokens=max_tokens,
                    **kwargs
                )
                if response.choices[0].finish_reason == "incomplete":
                    # Token limit exceeded, split the conversation and retry
                    print("Token limit exceeded. Splitting conversation and retrying...")
                    split_chunks = self.split_long_conversation(messages, max_tokens)
                    responses = []
                    for chunk in split_chunks:
                        response_chunk = self._chat_completion(chunk, **kwargs)
                        if response_chunk:
                            responses.append(response_chunk.choices[0].message)
                    return responses
                else:
                    return response

            except openai.error.APIError as e:
                if handle_error(e, retry_count, kwargs.get("max_retry_attempts", 3), kwargs.get("retry_wait_time", 60)):
                    print("Error in _chat_completion: ", e)
                    retry_count += 1
                else:
                    return None

    def generate_response(self, messages: List[ChatMessage], max_tokens: int = 128, **kwargs) -> openai.ChatCompletion:
        if len(messages) == 0:
            return "Error: No input messages."

        prompt = self._generate_prompt(messages, max_tokens)

        response = self._chat_completion(prompt, **kwargs)

        if isinstance(response, list):
            combined_response = " ".join(response)
            response = openai.ChatCompletion(choices=[openai.ChatCompletionChoice(message=combined_response)])

        try:
            if response and response.choices and len(response.choices) > 0:
                if self.use_memory:
                    self.memory.extend(messages)
                    response_dict = {
                        "role": "assistant",
                        "content": response.choices[0].message.content
                    }
                    self.memory.append(response_dict)
                return response
        except:
            return response

    def set_model(self, model_name):
        self.model_name = model_name

    def set_memory_usage(self, use_memory):
        self.use_memory = use_memory

    def prioritize_messages(self, messages: List[ChatMessage]):
        # Sort messages based on timestamp or relevance score
        # Ensure that higher-priority messages come first in the list
        sorted_messages = sorted(messages, key=lambda msg: msg.timestamp, reverse=True)
        return sorted_messages

    def split_long_conversation(self, messages: List[ChatMessage], max_tokens_per_chunk):
        split_chunks = []
        current_chunk = []
        current_chunk_tokens = 0

        for msg in messages:
            msg_tokens = count_tokens(msg.content)
            if current_chunk_tokens + msg_tokens <= max_tokens_per_chunk:
                current_chunk.append(msg)
                current_chunk_tokens += msg_tokens
            else:
                split_chunks.append(current_chunk)
                current_chunk = [msg]
                current_chunk_tokens = msg_tokens

        if current_chunk:
            split_chunks.append(current_chunk)

        return split_chunks


# Completion Model Wrapper
class CompletionModelWrapper:
    def __init__(self, api_key, use_memory=True, max_completion_token=3000, model_name="text-davinci-003"):
        openai.api_key = api_key
        self.memories = []
        self.max_completion_token = max_completion_token
        self.completion_model_name = model_name
        self.use_memory = use_memory
        self.max_retry_attempts = 3

    def _manage_memory(self, current_prompt, max_tokens):
        number_of_token_in_current_prompt = count_tokens(current_prompt, model_name=self.completion_model_name)
        total_memory_tokens = sum(
            count_tokens(memory["USER"], model_name=self.completion_model_name) + count_tokens(memory["AI"],
                                                                                               model_name=self.completion_model_name)
            for memory in self.memories)

        while total_memory_tokens > self.max_completion_token - (number_of_token_in_current_prompt + max_tokens):
            removed_memory = self.memories.pop(0)
            total_memory_tokens -= count_tokens(removed_memory["USER"],
                                                model_name=self.completion_model_name) + count_tokens(
                removed_memory["AI"], model_name=self.completion_model_name)

    def _format_conversation(self, current_prompt):
        if self.use_memory:
            conversation_series = "\n".join([f"User: {memory['USER']}\nAI: {memory['AI']}" for memory in self.memories])
            conversation_series += f"\nUser: {current_prompt}\nAI:"
            return conversation_series
        else:
            return current_prompt

    def _completion(self, prompt: str, max_tokens: int = 2000, temperature=1.0, **kwargs) -> openai.Completion:
        prompt_with_memory = self._format_conversation(prompt)
        retry_count = 0
        while retry_count < self.max_retry_attempts:
            try:
                response = openai.Completion.create(
                    model=self.completion_model_name,
                    prompt=prompt_with_memory,
                    max_tokens=max_tokens,
                    temperature=temperature,  # Control randomness of output
                    **kwargs
                )
                return response
            except openai.error.OpenAIError as e:
                if handle_error(e.response, retry_count, self.max_retry_attempts, kwargs.get("retry_wait_time", 60)):
                    retry_count += 1
                else:
                    return None

    def generate_response(self, prompt: str, max_tokens: int = 2000, temperature=1.0, **kwargs) -> openai.Completion:
        res = self._completion(prompt, max_tokens, temperature, **kwargs)
        if res:
            memory = {
                "USER": prompt,
                "AI": res.choices[0].text.strip()
            }
            self.memories.append(memory)
            self._manage_memory(prompt, max_tokens)  # Dynamic memory management
        return res

    def set_model(self, model_name):
        self.model_name = model_name

    def set_memory_usage(self, use_memory):
        self.use_memory = use_memory


class LLMWrapper:
    def __init__(self, api_key, model_type, use_memory=True, max_chat_completion_token=3000, model_name="gpt-3.5-turbo",
                 completion_model_name="text-davinci-003"):
        self.api_key = api_key
        self.model_type = model_type
        self.use_memory = use_memory
        self.max_chat_completion_token = max_chat_completion_token
        self.model_name = model_name
        self.completion_model_name = completion_model_name
        self.chat_wrapper = ChatModelWrapper(self.api_key, self.use_memory, self.max_chat_completion_token,
                                             self.model_name)
        self.completion_wrapper = CompletionModelWrapper(self.api_key, self.use_memory, self.max_chat_completion_token,
                                                         self.completion_model_name)

    def generate_response(self, messages_or_prompt, max_tokens: int = 2000, **kwargs) -> Union[
        openai.ChatCompletion, openai.Completion, str]:
        if isinstance(messages_or_prompt, str):
            prompt = messages_or_prompt
            messages = [ChatMessage(role="user", content=prompt)]
        else:
            messages = messages_or_prompt
            prompt = messages[-1].content

        if self.model_type == "Chat":
            res = self.chat_wrapper.generate_response(messages, max_tokens, **kwargs)
            return res.choices[0].message
        elif self.model_type == "Completion":
            res = self.completion_wrapper.generate_response(prompt, max_tokens, **kwargs)
            return res.choices[0].text.strip()
        else:
            return "Invalid model_type specified."

    def get_conversation_sequence(self, messages, max_tokens):
        sequence = []
        role_mapping = {"user": "User:", "assistant": "AI:"}
        total_tokens = 0
        for msg in messages:
            msg_tokens = count_tokens(msg.content)
            if total_tokens + msg_tokens <= max_tokens:
                sequence.append(f"{role_mapping[msg.role]} {msg.content}")
                total_tokens += msg_tokens
            else:
                break
        return "\n".join(sequence)

    def format_prompt(self, conversation_history):
        formatted_prompt = f"{conversation_history}\nUser:"
        return formatted_prompt

    def generate_response_with_dynamic_prompt(self, messages_or_prompt, max_tokens: int = 2000, **kwargs) -> Union[
        openai.ChatCompletion, openai.Completion, str]:
        if isinstance(messages_or_prompt, str):
            prompt = messages_or_prompt
            messages = [ChatMessage(role="user", content=prompt)]
        else:
            messages = messages_or_prompt
            prompt = messages[-1].content

        conversation_history = self.get_conversation_sequence(messages, max_tokens)
        formatted_prompt = self.format_prompt(conversation_history)
        return self.generate_response(formatted_prompt, max_tokens)

# Usage
llm_wrapper = LLMWrapper(API_KEY, model_type="Chat")

messages = [
    ChatMessage(role="user", content="Hello, how are you?"),
    ChatMessage(role="assistant", content="I'm doing well, thank you!")
]

response = llm_wrapper.generate_response_with_dynamic_prompt(messages, max_tokens=100)
print("Generated Response:", response)

llm_wrapper = LLMWrapper(API_KEY, model_type="Completion")

prompt = "Once upon a time in a village, far away"
response = llm_wrapper.generate_response_with_dynamic_prompt(prompt)
print("Generated Response:", response)

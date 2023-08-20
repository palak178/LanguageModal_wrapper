# LLM Wrapper 

## Introduction
The Language Model (LLM) Wrapper is a Python script designed to simplify interactions with OpenAI's language models, specifically the Chat and Completion models. It offers an easy-to-use interface for handling conversations and text completions while managing conversation history, token limits, and error handling.

## Features
- **Rate Limit and Token Limit Handling**: The wrapper automates the management of rate limit and token limit errors by re-pipelining data to the language model, ensuring a smooth experience.
- **Conversation Sequence and History**: The implementation maintains a chronological sequence of the conversation, including user inputs, prompts, and history. This sequence provides context and facilitates memory management.
- **Dynamic Conversation History Management**: The wrapper ensures that the conversation history adheres to token limits. Older entries can be intelligently removed to accommodate new messages.
- **Prompt Formatting and Variable Update**: The prompt is consistently formatted at every conversation step. Variables within the prompt are updated to reflect the latest input, ensuring precise context for the language model.

## Installation
1. Clone or download the repository to your local machine.
2. Install the required dependencies using the following command:
```
pip install openai tiktoken pydantic python-dotenv
```
3. Obtain your OpenAI API key and set it as an environment variable named `OPENAI_API_KEY`.
4. Open the `llm_wrapper.py` script and customize the settings to suit your use case if necessary.
5. You can now utilize the `LLMWrapper` class to generate responses for your conversations and prompts.

## Implementation
The wrapper implementation is organized into three primary classes:

### ChatModelWrapper
This class facilitates chat-based conversations and includes the following capabilities:
- Conversation history management
- Memory management to comply with token limits
- Handling rate limit errors and token limit errors

### CompletionModelWrapper
This class interacts with text completion models and provides the following features:
- Memory management to preserve context for text completions
- Dynamic memory management to adjust memory based on token limits

### LLMWrapper
The `LLMWrapper` class serves as the main interface, offering the choice between chat-based models and text completion models. It provides methods to generate responses.

## Conclusion

The Language Model (LLM) Wrapper simplifies language model interactions by handling rate limit errors, managing conversation history, and dynamically formatting prompts. With specialized classes for chat-based models and text completions, the wrapper streamlines integration into applications, enabling context-aware responses. Enhance your projects with the power of language models using the LLM Wrapper.



### Thank you





import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient
from pygls.server import LanguageServer
from lsprotocol import types 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)

checkpoint = "bigcode/starcoder2-3b"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# in-process processing
#device = "cpu" # for CPU usage or "cuda" for GPU usage
#model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# inferrence
# https://huggingface.co/docs/api-inference/getting-started

# local inference
# docker run -p 8080:80 -v $PWD/data:/data -e HUGGING_FACE_HUB_TOKEN=<YOUR BIGCODE ENABLED TOKEN> -d  ghcr.io/huggingface/text-generation-inference:latest --model-id bigcode/starcoder2-15b --max-total-tokens 8192
# https://github.com/huggingface/text-generation-inference

# remote inference 
# https://huggingface.co/docs/huggingface_hub/guides/inference
hf_client = InferenceClient(
    checkpoint,
    token=os.getenv("HF_TOKEN"),
)

logger = logging.getLogger('lsp-server')
server = LanguageServer("lsp-server", "v0.1")

@server.feature(types.TEXT_DOCUMENT_COMPLETION)
def completions(params: types.CompletionParams):
    logging.info(f"Completions called with: {params}")
    items = []
    document = server.workspace.get_text_document(params.text_document.uri)
    logging.info(f"Document recieved: {document}")

    document_lines = document.lines
    current_line_index = params.position.line
    token_limit = 1000

    def count_tokens(text):
        return len(tokenizer.encode(text))

    prefix = []
    suffix = []
    current_line = document_lines[current_line_index]
    current_tokens = count_tokens(current_line)

    # Initialize pointers for prefix and suffix
    prefix_index = current_line_index - 1
    suffix_index = current_line_index + 1

    while current_tokens < token_limit:
        if prefix_index >= 0:
            line = document_lines[prefix_index]
            line_tokens = count_tokens(line)
            if current_tokens + line_tokens <= token_limit:
                prefix.insert(0, line)
                current_tokens += line_tokens
                prefix_index -= 1
            else:
                break

        if suffix_index < len(document_lines):
            line = document_lines[suffix_index]
            line_tokens = count_tokens(line)
            if current_tokens + line_tokens <= token_limit:
                suffix.append(line)
                current_tokens += line_tokens
                suffix_index += 1
            else:
                break

        # Break if both pointers are out of bounds
        if prefix_index < 0 and suffix_index >= len(document_lines):
            break
    
    prefix = "\n".join(prefix)
    suffix = "\n".join(suffix)
    
    fim_input_text = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{current_line.strip()}"
    logger.info(f"Input: {fim_input_text}")

    fim_output_text = hf_client.text_generation(fim_input_text)
    logger.info(f"Output: {fim_output_text}")

    #fim_inputs = tokenizer.encode(fim_input_text, return_tensors="pt").to(device)
    #fim_outputs = model.generate(fim_inputs)
    #fim_output_text = tokenizer.decode(fim_outputs[0])

    items = [
        types.CompletionItem(label=fim_output_text)
    ]
    return types.CompletionList(is_incomplete=False, items=items)

def main():
    try:
        logger.info("Starting server")
        # server.start_tcp('127.0.0.1', 5000)
        server.start_io()
    except Exception as e:
        logger.error(f"Server encountered an error: {e}")
    finally:
        logger.info("Stopping server")
        logging.shutdown()

if __name__ == "__main__":
    main()
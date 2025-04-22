import torch
from model import LlamaModel
import utils


def main():
    # Load the model and tokenizer, and set the device
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = utils.init_tokenizer("model/tokenizer.model")
    model = LlamaModel("model").to(device)
    assert model.model is not None

    # The famous question from Hitchhiker's Guide to the Galaxy
    prompt = (
        "the answer to the ultimate question of life, the universe, and everything is "
    )

    # Tokenize the prompt - make sure we're using the same format as in correct.py
    tokens = tokenizer.encode(prompt)
    tokens = [model.bos_token] + tokens

    print(f"Tokens: {tokens}")
    print(f"Decoded tokens: {[tokenizer.decode([token]) for token in tokens]}")
    
    # Move tokens to the same device as the model
    token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)

    # Generate embeddings
    embeddings = model.generate(token_tensor)

    # Get the last token's embedding
    last_token_embedding = embeddings[-1]

    # Project to vocabulary space
    logits = torch.matmul(last_token_embedding, model.model["output.weight"].T.to(device)) # . to(device) is important for GPU compatibility

    # Apply temperature scaling
    logits = logits / 1.0  # You can adjust temperature here

    # Get the most likely next token
    next_token = torch.argmax(logits)
    next_token_id = int(next_token.item())
    next_token_text = tokenizer.decode([next_token_id])

    print(f"\nPrompt: {prompt}")
    print(f"Model's continuation: {next_token_text}")
    print(f"Next token ID: {next_token_id}")

    # Print top 5 most likely tokens for debugging
    top_k = 5
    top_logits, top_indices = torch.topk(logits, top_k)
    print("\nTop 5 predictions:")
    for i in range(top_k):
        token = tokenizer.decode([int(top_indices[i].item())])
        print(f"{i+1}. {token} (logit: {top_logits[i].item():.2f})")


if __name__ == "__main__":
    main()

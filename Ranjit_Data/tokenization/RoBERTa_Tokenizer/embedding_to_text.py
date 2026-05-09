from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn.functional as F
import numpy as np


class EmbeddingToText:

    def __init__(self, model_name="roberta-base"):

        # Load tokenizer + model
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)

        # Vocabulary embedding matrix
        # Shape: [vocab_size, hidden_size]
        self.embedding_matrix = self.model.embeddings.word_embeddings.weight

    def load_embedding(self, embedding_file="embedding.npy"):

        embedding_array = np.load(embedding_file)

        # Convert to torch tensor
        embedding_tensor = torch.tensor(embedding_array)

        return embedding_tensor

    def embedding_to_tokens(self, embedding_tensor):

        predicted_tokens = []

        print("\nDECODING EMBEDDINGS TO TOKENS\n")

        for i, token_embedding in enumerate(embedding_tensor):

            # Add batch dimension
            token_embedding = token_embedding.unsqueeze(0)

            # Compute cosine similarity against all vocab embeddings
            similarities = F.cosine_similarity(
                token_embedding,
                self.embedding_matrix,
                dim=1
            )

            # Get best matching token
            best_token_id = torch.argmax(similarities).item()

            # Convert ID to token
            token = self.tokenizer.convert_ids_to_tokens(best_token_id)

            predicted_tokens.append(token)

            # print(f"Token {i + 1}: {token}")
            # print(f"Token ID: {best_token_id}")
            # print(f"Similarity Score: {similarities[best_token_id].item():.4f}")
            # print("-" * 50)

        return predicted_tokens

    def tokens_to_text(self, tokens):

        # Convert tokens back to text
        text = self.tokenizer.convert_tokens_to_string(tokens)

        return text

    def run(self, embedding_file="embedding.npy"):

        # Load embedding
        embedding_tensor = self.load_embedding(embedding_file)

        # print("Loaded Embedding Shape:")
        # print(embedding_tensor.shape)

        # Decode embeddings
        tokens = self.embedding_to_tokens(embedding_tensor)

        # Convert to text
        reconstructed_text = self.tokens_to_text(tokens)

        print("\n\nRECONSTRUCTED TEXT\n")
        print(reconstructed_text)
        with open("ddd.txt", encoding="utf-8") as f:
            f.write(reconstructed_text)


if __name__ == "__main__":

    decoder = EmbeddingToText()

    decoder.run("embedding.npy")
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np


class RobertaEmbeddingExtractor:

    def __init__(self, model_name="roberta-base"):

        # Load tokenizer + model
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)

        # Embedding layer
        self.embedding_layer = self.model.embeddings.word_embeddings

    def process_text(self, text, output_file="embedding.npy"):

        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")

        # Token IDs
        input_ids = inputs["input_ids"][0]

        # Tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # Get embeddings for entire sequence
        with torch.no_grad():
            full_embeddings = self.embedding_layer(input_ids)

        # Convert to numpy
        embedding_array = full_embeddings.numpy()

        # Save embeddings
        np.save(output_file, embedding_array)

        print("\nTOKEN INFORMATION\n")

        # for i, token in enumerate(tokens):

        #     print(f"Token {i + 1}: {token}")
        #     print(f"Token ID : {input_ids[i].item()}")
        #     print(f"Embedding Shape : {embedding_array[i].shape}")

        #     print("-" * 50)

        print("\nFULL EMBEDDING SAVED SUCCESSFULLY")
        print(f"Saved File : {output_file}")

        print("\nEmbedding Tensor Shape:")
        print(embedding_array.shape)

    def load_embedding(self, file_path="embedding.npy"):

        embedding = np.load(file_path)

        print("\nLOADED EMBEDDING\n")
        print(embedding)

        print("\nShape:")
        print(embedding.shape)

        return embedding

    def run(self):

        text = input("Enter a sentence: ")

        self.process_text(text)


if __name__ == "__main__":

    extractor = RobertaEmbeddingExtractor()

    extractor.run()
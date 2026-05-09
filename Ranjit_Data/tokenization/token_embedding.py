# import torch
# from transformers import RobertaTokenizer, RobertaModel

# # Load tokenizer + model
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# model = RobertaModel.from_pretrained("roberta-base")

# # Example sentence
# text = input("Enter a the line: ")

# # Tokenize
# inputs = tokenizer(text, return_tensors="pt")

# # Get token ids
# input_ids = inputs["input_ids"][0]
# # Get embedding layer
# embedding_layer = model.embeddings.word_embeddings

# print("\nTOKEN ↔ EMBEDDING\n")

# for token_id in input_ids:

#     # Convert ID -> token
#     token = tokenizer.convert_ids_to_tokens(token_id.item())

#     # Get embedding vector
#     embedding = embedding_layer(token_id)

#     print(f"Token: {token}")
#     print(f"Token ID: {token_id.item()}")

#     # show first 8 dimensions only
#     print("Embedding[:8]:")
#     print(embedding[:8])

#     print("-" * 50)


from transformers import RobertaTokenizer, RobertaModel
from transformers.utils import logging



class RobertaEmbeddingExtractor:

    def __init__(self, model_name="roberta-base"):

        # Load tokenizer + model
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)

        # Embedding layer
        self.embedding_layer = self.model.embeddings.word_embeddings

    def process_text(self, text, show_dims=8):

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")

        # Get token ids
        input_ids = inputs["input_ids"][0]

        print("\nTOKEN ↔ EMBEDDING\n")

        for token_id in input_ids:

            # Token
            token = self.tokenizer.convert_ids_to_tokens(token_id.item())

            # Embedding
            embedding = self.embedding_layer(token_id)

            print(f"Token: {token}")
            print(f"Token ID: {token_id.item()}")

            print(f"Embedding[:{show_dims}]:")

            print(embedding[:show_dims])

            print("-" * 50)

    def run(self):

        text = input("Enter a sentence: ")

        self.process_text(text)
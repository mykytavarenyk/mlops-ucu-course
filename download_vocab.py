from transformers import BertTokenizer

# Download the tokenizer, which includes the vocab.txt file
tokenizer = BertTokenizer.from_pretrained("inference/bert-base-uncased")

# Save the vocab.txt file to the desired directory
tokenizer.save_pretrained("bert-base-uncased")

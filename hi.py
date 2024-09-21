import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import json

# Load and preprocess the Tweet Emotions Dataset
class TweetEmotionDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Set pad_token to eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet_text = self.data.loc[idx, "content"]
        sentiment = self.data.loc[idx, "sentiment"]
        inputs = self.tokenizer(f"Tweet: {tweet_text} | Sentiment: {sentiment}", 
                                return_tensors="pt", padding="max_length", 
                                truncation=True, max_length=128)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return input_ids, attention_mask

# Load complex_reasoning.json
class ComplexReasoningDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Set pad_token to eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming the keys in your JSON file are "prompt" and "solution"
        prompt = self.data[idx]["id"]
        solution = self.data[idx]["conversations"]
        inputs = self.tokenizer(f"Prompt: {prompt} | Solution: {solution}", 
                                return_tensors="pt", padding="max_length", 
                                truncation=True, max_length=128)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return input_ids, attention_mask

# Prepare Datasets
def prepare_datasets():
    # Load the Tweet Emotions dataset
    tweet_emotions_dataset = TweetEmotionDataset("tweet_emotions.csv")
    
    # Load the complex reasoning dataset
    complex_reasoning_dataset = ComplexReasoningDataset("complex_reasoning.json")
    
    return tweet_emotions_dataset, complex_reasoning_dataset

# Training Loop
# Training Loop
def train_model():
    tweet_emotions_dataset, complex_reasoning_dataset = prepare_datasets()
    
    print(f"Tweet Emotions Dataset Length: {len(tweet_emotions_dataset)}")
    print(f"Complex Reasoning Dataset Length: {len(complex_reasoning_dataset)}")

    # Combine the datasets into a single loader (for simplicity)
    combined_dataset = torch.utils.data.ConcatDataset([tweet_emotions_dataset, 
                                                       complex_reasoning_dataset])
    data_loader = DataLoader(combined_dataset, batch_size=4, shuffle=True)

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)  # Adjusted learning rate
    model.train()

    # Training loop
    epochs = 1  # Set to 1 for testing
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}")
        total_loss = 0
        for batch_idx, (input_ids, attention_mask) in enumerate(data_loader):
            print(f"Processing Batch {batch_idx + 1}/{len(data_loader)}")

            input_ids, attention_mask = input_ids, attention_mask  # No device transfer

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Dummy loss calculation for demonstration
            loss = torch.nn.functional.cross_entropy(logits.view(-1, model.config.vocab_size), input_ids.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}], Loss: {loss.item():.4f}")

        average_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}] Average Loss: {average_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "gpt2_emotion_reasoning_model.pt")
    print("Model training complete and saved.")

if __name__ == "__main__":
    train_model()

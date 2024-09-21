# Load the model
model = GPT2EmotionClassifier()
model.load_state_dict(torch.load("gpt2_emotion_instruct_reasoning_model.pt"))
model.eval()  # Set the model to evaluation mode
# Example of generating a response
prompt = "What are the benefits of regular exercise?"
chain_of_thought_response = generate_chain_of_thought_response(model, prompt)
print(chain_of_thought_response)

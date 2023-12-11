# Import necessary libraries
import random
import json
import torch

# Import functions from other modules
from model import ChatModel
from Nlp_toolkit import bag_of_words, tokenize

# Set device to use either CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a function to get random responses from an intent
def get_random_responses(intent_data, num_responses=3):
  """
  This function randomly selects a set of responses from the given intent.

  Args:
    intent_data: A dictionary containing information about the intent.
    num_responses: The number of responses to select (default: 3).

  Returns:
    A list of randomly chosen responses.
  """
  # Get all responses for the intent
  responses = intent_data["responses"]

  # Limit the number of responses to return
  num_responses = min(num_responses, len(responses))

  # Choose random responses
  selected_responses = random.sample(responses, k=num_responses)

  return selected_responses

# Load intents data from JSON file
intents_data = None
with open(r"roadmap/intents.json", "r") as json_file:
  intents_data = json.load(json_file)

# Load trained model data
model_data = None
with open(r"chat_model.pth", "rb") as model_file:
  model_data = torch.load(model_file)

# Extract information from the loaded model data
input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
all_words = model_data["all_words"]
tags = model_data["tags"]
model_state = model_data["model_state"]

# Initialize and load the chatbot model
chat_model = ChatModel(input_size, hidden_size, output_size).to(device)
chat_model.load_state_dict(model_state)
chat_model.eval()

# Define bot name
bot_name = "NinjaBot"

# Start chat loop
print("Let's chat! (Type 'quit' to exit)")
while True:
  # Get user input
  user_input = input("You: ")

  # Check for quit command
  if user_input.lower() == "quit":
    break

  # Preprocess user input
  tokenized_input = tokenize(user_input)
  input_bag_of_words = bag_of_words(tokenized_input, all_words)
  input_bag_of_words = input_bag_of_words.reshape(1, input_bag_of_words.shape[0])
  input_tensor = torch.from_numpy(input_bag_of_words).to(device)

  # Predict the intent of the user's message
  output = chat_model(input_tensor)
  _, predicted_class = torch.max(output, dim=1)
  predicted_tag = tags[predicted_class.item()]

  # Calculate the probability of the predicted intent
  probabilities = torch.softmax(output, dim=1)
  predicted_probability = probabilities[0][predicted_class.item()]

  # Check if the model is confident in its prediction
  if predicted_probability.item() > 0.75:
    # Find the corresponding intent data based on the predicted tag
    for intent in intents_data["intents"]:
      if intent["tag"] == predicted_tag:
        # Get appropriate responses based on the intent type
        if predicted_tag in ["greeting", "thanks"]:
          responses = get_random_responses(intent, num_responses=1)
        else:
          # Sort responses by their order in the original data
          responses = sorted(intent["responses"], key=lambda x: intent["responses"].index(x))
          # Get a subset of random responses
          responses = get_random_responses(intent, num_responses=4)

        # Display bot responses
        for i, response in enumerate(sorted(responses), start=1):
          print(f"{bot_name}: {response}")

  # If the model is not confident, tell the user it doesn't understand
  else:
    print(f"{bot_name}: I'm not sure I understand.")

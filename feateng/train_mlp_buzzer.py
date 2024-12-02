import torch
from mlp_buzzer import MLPBuzzer
from gpr_guesser import GprGuesser  # Import the GprGuesser

# Example Parameters
filename = "mlp_buzzer"
run_length = 100
num_guesses = 1
hidden_dims = [128, 64]
learning_rate = 0.001

# Instantiate the MLPBuzzer
mlp_buzzer = MLPBuzzer(
    filename=filename,
    run_length=run_length,
    num_guesses=num_guesses,
    hidden_dims=hidden_dims,
    learning_rate=learning_rate,
)

# Example question and run_text
question = {
    "id": 1,
    "text": "This scientist developed the theory of relativity."
}
run_text = "This scientist developed the theory of relativity."
guess_history = [{}]


# Provide a valid cache filename
guesser_model_train = "../models/buzztrain_gpt4o_cache"
guesser_model_test = "../models/buzzdev_gpt4o_cache"

# Instantiate the GprGuesser with the required cache_filename
gpr_guesser_train = GprGuesser(cache_filename=guesser_model_train)

# Add the GprGuesser to the buzzer
mlp_buzzer.add_guesser("gpr_guesser", gpr_guesser_train, primary_guesser=True)

# Featurize the question
guess, features = mlp_buzzer.featurize(question=question, run_text=run_text, guess_history=guess_history)

# Define labels (dummy binary labels)
batch_size = 1  # Single question
features = torch.randn(batch_size, len(features))  # Dummy feature data
labels = torch.randint(0, 2, (batch_size, 1)).float()  # Dummy binary labels

# Training
loss = mlp_buzzer.train_on_batch(features, labels)
print(f"Training loss: {loss}")

# Save the trained model
mlp_buzzer.save()

# Load the model for inference
mlp_buzzer.load()

# Prediction
predictions = mlp_buzzer.predict(features)
print(f"Predictions: {predictions}")


# import torch
# from torch.utils.data import DataLoader
# from mlp_buzzer import BuzzerMLP, BuzzLoss
# from torch.optim import Adam

# # MPS device setup
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# # Define model, loss, and optimizer
# input_dim = 100  # Example feature size
# hidden_dims = [128, 64]  # Example hidden layer sizes
# output_dim = 1  # Binary classification
# model = BuzzerMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim).to(device)
# criterion = BuzzLoss().to(device)
# optimizer = Adam(model.parameters(), lr=1e-3)

# # Dummy dataset and dataloader
# features = torch.randn(1000, input_dim)
# labels = torch.randint(0, 2, (1000,)).float()
# dataloader = DataLoader(list(zip(features, labels)), batch_size=32, shuffle=True)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0.0
#     for batch_features, batch_labels in dataloader:
#         batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

#         # Forward pass
#         confidences = model(batch_features)
#         loss = criterion(confidences, batch_labels)

#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# torch.save(model.state_dict(), "mlp_buzzer.pth")
# print("Model saved as mlp_buzzer.pth")


import pickle
import torch
import torch.nn as nn

# Define a utility to infer architecture from state_dict
def infer_architecture(state_dict):
    hidden_dims = []
    input_dim = None
    for key, tensor in state_dict.items():
        if "weight" in key:  # Look at weight matrices
            if input_dim is None:
                input_dim = tensor.shape[1]  # First layer input dimension
            hidden_dims.append(tensor.shape[0])  # Layer output dimension
    return input_dim, hidden_dims[:-1]  # Exclude the final output layer

# Define the MLP model structure
class SampleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(SampleMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)  # Directly define as layers
    
    def forward(self, x):
        return self.layers(x)

# Load the model from .pkl file
def load_model(file_path, input_dim, hidden_dims):
    with open(file_path, 'rb') as f:
        state_dict = pickle.load(f)

    # Recreate the model structure
    model = SampleMLP(input_dim, hidden_dims)

    # Remap keys to match the model's architecture
    remapped_state_dict = {f"layers.{k}": v for k, v in state_dict.items()}
    model.load_state_dict(remapped_state_dict)
    model.eval()  # Set the model to evaluation mode
    return model

# Analyze model weights
def analyze_model_weights(model):
    analysis = {}
    for name, param in model.named_parameters():
        analysis[name] = {
            "shape": tuple(param.shape),
            "sample_values": param.detach().cpu().numpy().flatten()[:5].tolist()  # Display first 5 values
        }
    return analysis

# Test inference
def test_inference(model, input_dim):
    dummy_input = torch.rand((1, input_dim))  # Create a random input tensor
    output = model(dummy_input)
    return output

# Example usage
if __name__ == "__main__":
    # Path to your .pkl file
    model_file = "models/mlp_no_features.model.pkl"  # Replace with the actual file path

    # Load the state_dict and infer architecture
    with open(model_file, 'rb') as f:
        state_dict = pickle.load(f)
    input_dim, hidden_dims = infer_architecture(state_dict)
    print(f"Inferred Input Dimension: {input_dim}")
    print(f"Inferred Hidden Dimensions: {hidden_dims}")

    # Load the model
    model = load_model(model_file, input_dim, hidden_dims)

    # Analyze weights
    weight_analysis = analyze_model_weights(model)
    print("\nModel Weights Analysis:")
    for layer, info in weight_analysis.items():
        print(f"{layer}: Shape={info['shape']}, Sample Values={info['sample_values']}")

    # Test inference
    output = test_inference(model, input_dim)
    print(f"\nSample Output from Inference: {output}")

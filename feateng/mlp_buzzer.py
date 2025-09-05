import pickle
import torch
import torch.nn as nn
from buzzer import Buzzer  # Base class for buzzers

class MLPBuzzer(Buzzer):
    def __init__(self, filename, run_length, num_guesses, hidden_dims, learning_rate=1e-3, device=None):
        """
        Initializes the MLP-based buzzer.
        """
        super().__init__(filename, run_length, num_guesses)
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.loss_function = BuzzLoss()  # Use BuzzLoss here

    def _initialize_model(self, input_dim):
        """
        Dynamically initializes the MLP model with custom weight initialization.
        """
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Final layer for binary output
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers).to(self.device)

        # Apply custom weight and bias initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                nn.init.uniform_(m.bias, -0.01, 0.01)  # Small random bias initialization

        self.model.apply(init_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def featurize(self, question, run_text, guess_history, guesses=None):
        """
        Featurize question data and initialize the model dynamically if required.
        """
        guess, features = super().featurize(question, run_text, guess_history, guesses)

        # Separate numerical features from categorical features
        numerical_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        categorical_features = {k: v for k, v in features.items() if isinstance(v, str)}

        # Log feature variability
        feature_values = list(numerical_features.values())
        print(f"Numerical Feature values: {feature_values}")  # Log raw numerical features
        if len(set(feature_values)) <= 1:  # Check if all numerical features are constant
            print("Warning: Numerical features may lack variability!")

        # Normalize numerical features
        if len(numerical_features) > 0:
            min_val, max_val = min(feature_values), max(feature_values)
            normalized_features = {k: (v - min_val) / (max_val - min_val + 1e-8) for k, v in numerical_features.items()}
        else:
            normalized_features = numerical_features  # If no numerical features, skip normalization

        # Combine normalized numerical features with categorical features
        combined_features = {**normalized_features, **categorical_features}

        if self.model is None:
            self._initialize_model(input_dim=len(combined_features))

        return guess, combined_features


    def train(self):
        """
        Train the MLP model using features and labels.
        """
        X = Buzzer.train(self)  # Get features
        self._initialize_model(input_dim=X.shape[1])
        features = torch.tensor(X.toarray(), dtype=torch.float32).to(self.device)
        labels = torch.tensor(self._correct, dtype=torch.float32).unsqueeze(1).to(self.device)

        for epoch in range(10):  # Train for 10 epochs
            self.model.train()
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.loss_function(predictions, labels)

            # Log loss and gradient stats
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient stats for {name}: Mean={param.grad.mean().item()}, Std={param.grad.std().item()}")

            loss.backward()
            self.optimizer.step()

    def predict(self, features):
        """
        Predict buzz decisions for a batch of input features.
        """
        self.model.eval()
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(features)
        print(f"Predictions: {predictions}")  # Log predictions for debugging
        return (predictions > 0.3).float()  # Use a lower threshold

    def save(self):
        """
        Save the MLP model and parent state.
        """
        Buzzer.save(self)
        with open(f"{self.filename}.model.pkl", "wb") as f:
            pickle.dump(self.model.state_dict(), f)

    def load(self):
        """
        Load the MLP model and parent state.
        """
        Buzzer.load(self)
        with open(f"{self.filename}.model.pkl", "rb") as f:
            self.model.load_state_dict(pickle.load(f))


class BuzzLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, confidences, accuracies):
        """
        Args:
            confidences: Tensor of shape (batch_size, T), where T is the number of timesteps.
            accuracies: Tensor of shape (batch_size, T), binary (1 if correct, 0 otherwise).
        Returns:
            Loss: Scalar, negative system score.
        """
        batch_size, T = confidences.size()
        buzz_probs = torch.zeros_like(confidences)
        system_scores = torch.zeros(batch_size, device=confidences.device)

        # Calculate buzz probabilities and system scores
        for t in range(T):
            if t == 0:
                buzz_probs[:, t] = confidences[:, t]
            else:
                cumulative_no_buzz = torch.prod(1 - confidences[:, :t], dim=1)
                buzz_probs[:, t] = confidences[:, t] * cumulative_no_buzz

            # Add score contribution from current timestep
            system_scores += buzz_probs[:, t] * accuracies[:, t]

        # Ensure the final timestep confidence becomes 1.0 if it isn't already
        final_timestep_correction = 1.0 - torch.sum(buzz_probs, dim=1, keepdim=True)
        buzz_probs[:, -1] += final_timestep_correction.squeeze()
        system_scores += final_timestep_correction.squeeze() * accuracies[:, -1]

        return -torch.mean(system_scores)  # Negative system score


# class BuzzLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, confidences, accuracies):
#         """
#         Custom loss function for MLP buzzer.
#         Args:
#             confidences: Tensor of shape (batch_size, T), where T is the number of timesteps.
#             accuracies: Tensor of shape (batch_size, T), binary (1 if correct, 0 otherwise).
#         Returns:
#             Loss: Scalar, negative system score.
#         """
#         batch_size, T = confidences.size()
#         buzz_probs = torch.zeros_like(confidences)
#         system_scores = torch.zeros(batch_size, device=confidences.device)

#         for t in range(T):
#             if t == 0:
#                 buzz_probs[:, t] = confidences[:, t]
#             else:
#                 cumulative_no_buzz = torch.prod(1 - confidences[:, :t], dim=1)
#                 buzz_probs[:, t] = confidences[:, t] * cumulative_no_buzz

#             system_scores += buzz_probs[:, t] * accuracies[:, t]

#         return -torch.mean(system_scores)


# import pickle
# import torch
# import torch.nn as nn
# from buzzer import Buzzer  # Base class for buzzers

# class MLPBuzzer(Buzzer):
#     def __init__(self, filename, run_length, num_guesses, hidden_dims, learning_rate=1e-3, device=None):
#         """
#         Initializes the MLP-based buzzer.
#         """
#         super().__init__(filename, run_length, num_guesses)
#         self.hidden_dims = hidden_dims
#         self.learning_rate = learning_rate
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = None
#         self.optimizer = None
#         self.loss_function = BuzzLoss()

#     def _initialize_model(self, input_dim):
#         """
#         Dynamically initializes the MLP model.
#         """
#         layers = []
#         prev_dim = input_dim
#         for hidden_dim in self.hidden_dims:
#             layers.append(nn.Linear(prev_dim, hidden_dim))
#             layers.append(nn.ReLU())
#             prev_dim = hidden_dim
#         layers.append(nn.Linear(prev_dim, 1))
#         layers.append(nn.Sigmoid())

#         self.model = nn.Sequential(*layers).to(self.device)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

#     def featurize(self, question, run_text, guess_history, guesses=None):
#         """
#         Featurize question data and initialize the model dynamically if required.
#         """
#         guess, features = super().featurize(question, run_text, guess_history, guesses)
#         if self.model is None:
#             self._initialize_model(input_dim=len(features))
#         return guess, features

#     def train(self):
#         """
#         Train the MLP model using features and labels.
#         """
#         X = Buzzer.train(self)  # Get features
#         self._initialize_model(input_dim=X.shape[1])
#         features = torch.tensor(X.toarray(), dtype=torch.float32).to(self.device)
#         labels = torch.tensor(self._correct, dtype=torch.float32).unsqueeze(1).to(self.device)

#         for epoch in range(10):  # Train for 10 epochs
#             self.model.train()
#             self.optimizer.zero_grad()
#             predictions = self.model(features)
#             loss = self.loss_function(predictions, labels)

#             # Log loss
#             print(f"Epoch {epoch+1}, Loss: {loss.item()}")

#             loss.backward()
#             self.optimizer.step()

#     def predict(self, features):
#         """
#         Predict buzz decisions for a batch of input features.
#         """
#         self.model.eval()
#         features = torch.tensor(features, dtype=torch.float32).to(self.device)
#         with torch.no_grad():
#             predictions = self.model(features)
#         return (predictions > 0.5).float()  # Apply threshold for binary output

#     def save(self):
#         """
#         Save the MLP model and parent state.
#         """
#         Buzzer.save(self)
#         with open(f"{self.filename}.model.pkl", "wb") as f:
#             pickle.dump(self.model.state_dict(), f)

#     def load(self):
#         """
#         Load the MLP model and parent state.
#         """
#         Buzzer.load(self)
#         with open(f"{self.filename}.model.pkl", "rb") as f:
#             self.model.load_state_dict(pickle.load(f))


# class BuzzLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, predictions, labels):
#         """
#         Custom loss function for MLP buzzer.
#         """
#         return nn.BCELoss()(predictions, labels)  # Binary cross-entropy loss



# import pickle
# import torch
# import torch.nn as nn
# from buzzer import Buzzer  # Import the base Buzzer class

# class MLPBuzzer(Buzzer):
#     def __init__(self, filename, run_length, num_guesses, hidden_dims, learning_rate=1e-3):
#         """
#         Initializes the MLP-based buzzer, extending the Buzzer class.
#         Args:
#             hidden_dims: List of hidden layer dimensions.
#             run_length: Length of each question segment (run).
#             num_guesses: Number of guesses to evaluate at each run.
#             learning_rate: Learning rate for the optimizer.
#         """
#         super().__init__(filename=filename, run_length=run_length, num_guesses=num_guesses)
#         self.hidden_dims = hidden_dims
#         self.learning_rate = learning_rate
#         self.model = None  # Model will be initialized dynamically

#     def _initialize_model(self, input_dim):
#         """
#         Dynamically initializes the MLP model based on the input feature dimension.
#         Args:
#             input_dim: Dimension of the input features.
#         """
#         layers = []
#         prev_dim = input_dim
        
#         for hidden_dim in self.hidden_dims:
#             layers.append(nn.Linear(prev_dim, hidden_dim))
#             layers.append(nn.ReLU())
#             prev_dim = hidden_dim
        
#         layers.append(nn.Linear(prev_dim, 1))  # Output layer for binary classification
#         layers.append(nn.Sigmoid())  # Sigmoid for confidence score

#         self.model = nn.Sequential(*layers)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
#         self.loss_function = BuzzLoss()

#     def featurize(self, question, run_text, guess_history, guesses=None):
#         """
#         Overridden featurization method to generate features for the MLP model.
#         Dynamically initializes the model if it hasn't been initialized yet.
#         Args:
#             question: The question object.
#             run_text: The portion of the question text available so far.
#             guess_history: History of guesses made so far.
#             guesses: Precomputed guesses (optional). If None, will be computed.
#         """
#         # Ensure run_text is valid
#         if run_text is None:
#             raise ValueError("run_text cannot be None. Please provide valid text to featurize.")

#         # Call the parent featurize method
#         guess, features = super().featurize(question, run_text, guess_history, guesses)

#         # Initialize the model if not already initialized
#         if self.model is None:
#             self._initialize_model(input_dim=len(features))

#         return guess, features


    
#     def save(self):
#         Buzzer.save(self)
#         with open("%s.model.pkl" % self.filename, 'wb') as outfile:
#             pickle.dump(self._classifier, outfile)

#     def load(self):
#         Buzzer.load(self)
#         with open("%s.model.pkl" % self.filename, 'rb') as infile:
#             self._classifier = pickle.load(infile)


#     def train_on_batch(self, features, labels):
#         """
#         Train the MLP model on a single batch of data.
#         Args:
#             features: Tensor of input features.
#             labels: Tensor of binary labels (correct/incorrect guesses).
#         """
#         if self.model is None:
#             raise ValueError("Model not initialized. Ensure features are passed through `featurize` first.")
        
#         self.model.train()
#         self.optimizer.zero_grad()
#         features = features.to(next(self.model.parameters()).device)
#         labels = labels.to(next(self.model.parameters()).device)

#         confidences = self.model(features)
#         loss = self.loss_function(confidences, labels)
#         loss.backward()
#         self.optimizer.step()

#         return loss.item()

#     def predict(self, features):
#         """
#         Predict buzz decisions for a batch of input features.
#         Args:
#             features: Tensor of input features.
#         Returns:
#             Tensor of predicted probabilities.
#         """
#         if self.model is None:
#             raise ValueError("Model not initialized. Ensure features are passed through `featurize` first.")
        
#         self.model.eval()
#         with torch.no_grad():
#             features = features.to(next(self.model.parameters()).device)
#             confidences = self.model(features)
#         return confidences

# class BuzzLoss(nn.Module):
#     def __init__(self):
#         super(BuzzLoss, self).__init__()

#     def forward(self, confidences, accuracies):
#         """
#         Args:
#             confidences: Tensor of shape (batch_size, T), where T is the number of timesteps.
#             accuracies: Tensor of shape (batch_size, T), binary (1 if correct, 0 otherwise).
#         Returns:
#             Loss: Scalar, negative system score.
#         """
#         batch_size, T = confidences.size()
#         buzz_probs = torch.zeros_like(confidences)
#         system_scores = torch.zeros(batch_size, device=confidences.device)

#         for t in range(T):
#             if t == 0:
#                 buzz_probs[:, t] = confidences[:, t]
#             else:
#                 cumulative_no_buzz = torch.prod(1 - confidences[:, :t], dim=1)
#                 buzz_probs[:, t] = confidences[:, t] * cumulative_no_buzz
            
#             system_scores += buzz_probs[:, t] * accuracies[:, t]
        
#         return -torch.mean(system_scores)

# if __name__ == "__main__":
#     print("MLPBuzzer class defined, extending the Buzzer base class.")

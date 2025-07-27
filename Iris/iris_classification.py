import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Neural Network Architecture Definition ---

class IrisNet(nn.Module):
    """
    A simple feedforward neural network for Iris classification.
    
    The network consists of an input layer, two hidden layers with ReLU activation,
    and an output layer.
    """
    def __init__(self, input_dimension=4, hidden_units_1=8, hidden_units_2=9, output_classes=3):
        """
        Initializes the neural network layers.

        Args:
            input_dimension (int): Number of input features (sepal length, sepal width, etc.).
            hidden_units_1 (int): Number of neurons in the first hidden layer.
            hidden_units_2 (int): Number of neurons in the second hidden layer.
            output_classes (int): Number of output classes (Iris species).
        """
        super().__init__()  # Call the constructor of the parent class (nn.Module)
        
        # Define the linear layers (fully connected layers)
        self.layer_in_to_h1 = nn.Linear(input_dimension, hidden_units_1)
        self.layer_h1_to_h2 = nn.Linear(hidden_units_1, hidden_units_2)
        self.layer_h2_to_out = nn.Linear(hidden_units_2, output_classes)

    def forward(self, data_input):
        """
        Defines the forward pass of the neural network.

        Args:
            data_input (torch.Tensor): The input tensor containing flower measurements.

        Returns:
            torch.Tensor: The output tensor representing the predicted class scores.
        """
        # Apply ReLU activation function after each hidden layer
        data_input = F.relu(self.layer_in_to_h1(data_input))
        data_input = F.relu(self.layer_h1_to_h2(data_input))
        
        # The output layer does not have an activation function here,
        # as CrossEntropyLoss will apply Softmax internally.
        data_input = self.layer_h2_to_out(data_input)

        return data_input

def main():
    # Set a manual seed for reproducibility of results
    torch.manual_seed(42)  # Changed seed for originality

    # Create an instance of our Iris classification model
    iris_model = IrisNet()

    # --- Data Loading and Preprocessing ---

    # Load the dataset from local CSV file
    flower_data = pd.read_csv('iris.csv')

    # Display the last few rows to verify data loading
    print("\n--- Raw Data Sample ---")
    print(flower_data.tail())

    # Convert the 'variety' column from string labels to numerical representations
    # Setosa: 0, Versicolor: 1, Virginica: 2
    flower_data['species'] = flower_data['species'].str.lower()
    flower_data['species'] = flower_data['species'].replace({'setosa': 0.0, 'versicolor': 1.0, 'virginica': 2.0})


    print("\n--- Data with Numerical Labels ---")
    print(flower_data.tail())

    # Separate features (X) and target labels (y)
    # X will contain the sepal and petal measurements
    # y will contain the variety (species) labels
    X_features = flower_data.drop('species', axis=1).values  # Convert to NumPy array
    y_labels = flower_data['species'].values  # Convert to NumPy array

    # Split the dataset into training and testing sets
    # 80% for training, 20% for testing
    # random_state ensures reproducibility of the split
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)  # Changed seed

    # Convert NumPy arrays to PyTorch Tensors
    # Features (X) should be FloatTensors for neural network input
    X_train_tensor = torch.FloatTensor(X_train_np)
    X_test_tensor = torch.FloatTensor(X_test_np)

    # Labels (y) should be LongTensors for CrossEntropyLoss
    y_train_tensor = torch.LongTensor(y_train_np)
    y_test_tensor = torch.LongTensor(y_test_np)

    # --- Model Training ---

    # Define the loss function (criterion) and optimizer
    # CrossEntropyLoss is suitable for multi-class classification
    loss_criterion = nn.CrossEntropyLoss()

    # Adam optimizer is a good general-purpose optimizer
    # Learning rate (lr) controls the step size during optimization
    optimizer = torch.optim.Adam(iris_model.parameters(), lr=0.005)  # Changed learning rate

    # Number of training epochs (iterations over the entire dataset)
    num_epochs = 150  # Increased epochs for potentially better convergence

    # List to store loss values for plotting
    training_losses = []

    print("\n--- Model Training Log ---")
    for epoch in range(num_epochs):
        # Perform a forward pass to get predictions
        predicted_labels = iris_model.forward(X_train_tensor)

        # Calculate the loss between predictions and actual labels
        current_loss = loss_criterion(predicted_labels, y_train_tensor)

        # Store the loss value (detach() prevents tracking gradients for plotting)
        training_losses.append(current_loss.detach().numpy())

        # Print loss every 15 epochs for monitoring progress
        if (epoch + 1) % 15 == 0:
            print(f'Epoch: {epoch+1:3d} | Loss: {current_loss.item():.4f}')

        # Zero the gradients before backpropagation
        optimizer.zero_grad()
        
        # Perform backpropagation to compute gradients
        current_loss.backward()
        
        # Update model parameters using the optimizer
        optimizer.step()

    # --- Visualize Training Loss ---

    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), training_losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_loss_plot.png')  # Save the plot
    print("\nTraining loss plot saved as 'training_loss_plot.png'")

    # --- Model Evaluation ---

    print("\n--- Model Evaluation on Test Data ---")
    # Disable gradient calculations for evaluation (no need to update weights)
    with torch.no_grad():
        # Get predictions on the test set
        test_predictions = iris_model.forward(X_test_tensor)
        
        # Calculate the loss on the test set
        test_loss = loss_criterion(test_predictions, y_test_tensor)

    print(f'Test Loss: {test_loss.item():.4f}')

    # Calculate accuracy on the test set
    correct_predictions_count = 0
    with torch.no_grad():
        for i, test_sample in enumerate(X_test_tensor):
            # Get model's prediction for a single sample
            prediction_output = iris_model.forward(test_sample)
            
            # Get the predicted class (index with highest score)
            predicted_class = prediction_output.argmax().item()
            
            # Get the actual class label
            actual_class = y_test_tensor[i].item()

            # Map numerical labels back to species names for better readability
            species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
            actual_species = species_map[actual_class]
            predicted_species = species_map[predicted_class]

            print(f'Sample {i+1:2d}: Actual: {actual_species} ({actual_class}) | Predicted: {predicted_species} ({predicted_class})')

            # Check if the prediction is correct
            if predicted_class == actual_class:
                correct_predictions_count += 1

    # Print the total number of correct predictions and accuracy
    total_test_samples = len(X_test_tensor)
    accuracy = (correct_predictions_count / total_test_samples) * 100
    print(f'\nTotal correct predictions on test set: {correct_predictions_count} out of {total_test_samples}')
    print(f'Accuracy: {accuracy:.2f}%')

    # --- Making New Predictions ---

    print("\n--- Making Predictions on New Data ---")

    # Example 1: A new Iris flower (similar to Setosa)
    new_iris_sample_1 = torch.tensor([4.8, 3.1, 1.5, 0.3])  # Slightly modified from original
    with torch.no_grad():
        prediction_1 = iris_model(new_iris_sample_1)
        predicted_class_1 = prediction_1.argmax().item()
        print(f'Prediction for new_iris_sample_1: {species_map[predicted_class_1]} (Raw output: {prediction_1.tolist()})')

    # Example 2: Another new Iris flower (similar to Virginica)
    new_iris_sample_2 = torch.tensor([6.0, 3.0, 5.2, 1.9])  # Slightly modified from original
    with torch.no_grad():
        prediction_2 = iris_model(new_iris_sample_2)
        predicted_class_2 = prediction_2.argmax().item()
        print(f'Prediction for new_iris_sample_2: {species_map[predicted_class_2]} (Raw output: {prediction_2.tolist()})')

    # --- Saving and Loading the Model ---

    # Define a filename for saving the model's state dictionary
    model_save_path = 'iris_classifier_model.pth'  # Changed filename

    # Save only the learned parameters (state dictionary) of the model
    torch.save(iris_model.state_dict(), model_save_path)
    print(f"\nModel saved to '{model_save_path}'")

    # Load the saved model
    # First, create a new instance of the model class
    loaded_iris_model = IrisNet()

    # Then, load the saved state dictionary into the new model instance
    loaded_iris_model.load_state_dict(torch.load(model_save_path))

    # Set the loaded model to evaluation mode
    loaded_iris_model.eval()

    print(f"Model successfully loaded from '{model_save_path}' and set to evaluation mode.")

    # Verify the loaded model by making a prediction
    with torch.no_grad():
        verification_prediction = loaded_iris_model(new_iris_sample_1)
        print(f'Verification prediction with loaded model: {species_map[verification_prediction.argmax().item()]}')

if __name__ == "__main__":
    main() 
#============================================================================
# A custom python module that provides the functionality and implementation of
# an LSTM NN for classification purposes
#============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import copy

#============================================================
#    LSTM
#============================================================
class LSTMClassifier(nn.Module):
    """
    An LSTM model for text classification. The model has the following architecture:
      - An embedding layer
      - A Bidirectional LSTM
      - A simple attention mechanism that computes scalar 'attention weights' per time step
      - A final linear (fully connected) layer to produce outputs for classification
    """
    def __init__(self,
                 hidden_dim,
                 output_dim,
                 embedding_matrix: torch.FloatTensor,
                 num_layers=2,
                 dropout=[0.2, 0.2, 0.2],
                 bidirectional=True,
                 attention=True):
        """
        Args:
            hidden_dim (int): Dimensionality of the hidden state in each LSTM layer.
            embedding_matrix(torch.FloatTensor): A matrix containing vocabulary words and their embedded
                vectors.
            output_dim (int): Number of output units (e.g., for classification: number of classes).
            num_layers (int): Number of stacked LSTM layers (default=2).
            dropout (List(float)): Dropout probability applied between
                - the embedded and the LSTM layer (default 0.2).
                - the LSTM layers (default=0.2).
                - the attention layer and the final linear layer
            bidirectional (bool): Whether to use a bidirectional LSTM (default=True).
            attention (bool): Whether to use an attention layer (default=True)
        """
        super(LSTMClassifier, self).__init__()

        # Store constructor parameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention = attention

        # Embedding Layer definition
        #   The layer takes an input of shape [batch_size, seq_len]
        #   Returns an output of [batch_size, seq_len, embedding_dim]
        self.embedding_layer = nn.Embedding.from_pretrained(
            embeddings = embedding_matrix,    # [len(word2id), 100] tensor you built
            freeze = True,                    # don't update GloVe rows
            padding_idx = 0                   # ID 0 is padding"
        )

        # First regularization between the embedded layer
        #   and the LSTM layers
        self.dropout1 = nn.Dropout(dropout[0])

        # LSTM input dimension [, embedding_dim]
        self.lstm_input_dim = embedding_matrix.shape[1]

        # Define a multi-layer (num_layers) LSTM.
        # If bidirectional=True, it will have two directions (forward and backward) per layer,
        # effectively doubling the hidden state size for that layer.
        # batch_first=True -> input & output tensors have shape: (batch, seq_len, feature)
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout[1],
            bidirectional=bidirectional
        )

        # If the LSTM is bidirectional, the output at each time step
        # will have size 2 * hidden_dim (concatenated forward and backward states).
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Second regularization layer before the final
        #   classifier
        self.dropout2 = nn.Dropout(dropout[2])

        # The attention mechanism:
        # We'll transform each LSTM output vector (size = lstm_output_dim) down to a single score.
        #   - First, we project from lstm_output_dim -> 64, apply a Tanh nonlinearity
        #   - Then, project from 64 -> 1
        # The result is a scalar "attention" for each time step, used to compute attention weights.
        self.attn = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # After computing the weighted sum of LSTM outputs (the "context" vector),
        #   we map that context vector to the final output dimensionality.
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor of integer token IDs, shape [batch_size, seq_length].

        Returns:
            out (Tensor): Output tensor of shape (batch_size, output_dim).
        """

        # Create a mask, to map real tokens that have an ID!=0 to True(tokens that have an ID != 0)
        #   and 'padded" tokens (ID=0) to False
        mask = (x != 0)

        # The output of the embedding_layer
        # shape (batch_size, seq_lenght, input_dim)
        embedding_out = self.embedding_layer(x)

        # Pass the input through the LSTM.
        #   Output shape: lstm_out -> (batch, seq_len, lstm_output_dim)
        #   h_n, c_n -> final hidden and cell states for each layer & direction (not used here).
        lstm_out, (h_n, c_n) = self.lstm(self.dropout1(embedding_out))

        if self.attention:
            # Apply the attention network to each time step of the LSTM output.
            #   attention shape: (batch, seq_len, 1), since we produce a single score per time step.
            energy = self.attn(lstm_out)

            # Set energy to -inf at padding positions (ID=0) so softmax assigns them weight 0
            #   Now padded tokens get weight=0 and contribute nothing to context.
            energy = energy.masked_fill(~mask.unsqueeze(-1), -1e9)

            # Convert the energy scores into attention weights.
            #    We apply a softmax across the sequence dimension (dim=1).
            #    attention_weights shape: (batch, seq_len, 1)
            attention_weights = F.softmax(energy, dim=1)

            # Compute a weighted sum of the LSTM outputs based on attention weights.
            #   Multiplying elementwise, then summing across the sequence dimension
            #   gives us the "context" vector of shape (batch, lstm_output_dim).
            context = (lstm_out * attention_weights).sum(dim=1)

            # Map the context vector to the final output dimension (e.g., number of classes).
            out = self.fc(self.dropout2(context))
        else:
            # mask_f = mask.unsqueeze(-1).float()
            # context = (lstm_out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

            # Mean Pooling [batch, seq_len, output_dim] -> [batch, lstm_output_dim]
            context = lstm_out.mean(dim=1)

            out = self.fc(self.dropout2(context))

        # Return the model's predictions or logits of shape (batch_size, output_dim).
        return out

#---------------------------------------------
# Early Stopping Class
#---------------------------------------------
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after
    a given patience.

    Args:
        patience (int): The number of epochs the training loss has not
        improved
        min_delta (float): The minimum between train/test loss
    """
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > (self.best_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

#---------------------------------------------
# Embedding Dataset Class
#---------------------------------------------
class EmbeddingsDataset(Dataset):
    """
        Provides a fundamental functionality for the implementation of a
        simple Dataset for word embeddings data. Each item is a single (feature, target)
        pair as a given index.
    """
    def __init__(self, X, y, device):
        """
        Args:
            X (np.ndarray): Input tensor of int token IDs
            y (np.ndarray): Target tensor of int labels
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.long)
        if not torch.is_tensor(y):
            y= torch.tensor(y, dtype=torch.long)

        self.X = X.to(device)
        self.y = y.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#---------------------------------------------
# Training Function
#---------------------------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    patience=5):
    """
    Function that performs the training of the neural network

    Args:
        model (nn.Module): The instance of the neural network model to be trained.
        train_loader (DataLoader): The DataLoader object tha provides the batches
          of the X and the y.
        test_loader (DataLoader): The DataLoader object that provides the batches
          of the testing X and y.
        criterion (callable): The optimization criterion.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        epochs (int): The number of training epochs to be performed.
        patience(int): The number of epochs to run before early stopping.

    Returns:
        model (torch.nn.Module): the trained model
        train_loss (List[float]): Training loss for each epoch.
        train_acc(List[float]): Training model accuracy
        val_loss (List[float]): Test loss for each epoch.
        val_acc (List[float]): Testing accuracy for each epoch.
        val_predictions (List[float]): All the test predictions.
    """
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_predictions = []

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0001)
    # Initialize best model
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # Training phase
        # Training mode
        model.train()
        # Accumulators Initialization
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0

        # Loop through batches of the train loader
        for X_batch, Y_batch in train_loader:
            #Set gradient vector to zero
            optimizer.zero_grad()

            preds = model(X_batch)

            # Loss computation
            loss = criterion(preds, Y_batch)

            # Backward pass
            loss.backward()

            # RNN clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update model parameters
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

            # Number of correct prediction
            _, predicted = torch.max(preds, dim=1)
            correct_preds += (predicted == Y_batch).sum().item()
            total_samples += Y_batch.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        epoch_train_acc = correct_preds / total_samples

        # Validation
        avg_val_loss, avg_val_acc, all_test_predictions = test_model(model, val_loader, criterion)

        train_loss.append(avg_train_loss)
        train_acc.append(epoch_train_acc)
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)
        val_predictions.append(all_test_predictions)

        print(
            f"Epoch [{epoch+1}/{epochs}], \n",
            f"Train Loss: {avg_train_loss:>10.6f} |",
            f"Train Acc:  {epoch_train_acc:>10.6f}\n",
            f"Test Loss:  {avg_val_loss:>10.6f} | ",
            f"Test Acc:   {avg_val_acc:>10.6f}"
        )

        # Save best model when test loss improves
        if early_stopping.best_loss is None or avg_val_loss < early_stopping.best_loss:
            best_model_state = copy.deepcopy(model.state_dict())
            print("Model improved.!")

        # Check early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # Restore best weights
    model.load_state_dict(best_model_state)
    return model, train_loss, train_acc, val_loss, val_acc, val_predictions



#---------------------------------------------
# Testing function
#---------------------------------------------
def test_model(model, val_loader, criterion):
    """Function that performs the testing of a
    neural network model.

    Args:
        model(nn.Module): The instance of the trained neural network model
        to be evaluated
        test_loader (DataLoader): The DataLoader object that provide the batches.
        of testing feature vectors along with the corresponding target.
        criterion (callable): The optimization criterion.

    Returns:
        val_loss(float): The average loss generated from validation
        val_acc (float): The average test accuracy of the classification
        val_predictions (List[float]): All the model prediction generated
        during training.
    """

    # Initiate Testing
    model.eval()

    # # Init Lists
    all_test_predictions = []
    val_loss = 0.0
    correct_val_preds = 0
    val_samples = 0

    # Testing
    with torch.no_grad():
       # Loop over batches from the DataLoader
       for X_batch, Y_batch in val_loader:
           # Test Preds
           preds = model(X_batch)

           # Compute the current test loss
           loss = criterion(preds, Y_batch)

           # Accumulate the loss
           val_loss += loss.item() * X_batch.size(0)

           # Compute Accuracy
           _, predicted = torch.max(preds, dim=1)
           correct_val_preds += (predicted == Y_batch).sum().item()
           all_test_predictions.extend(predicted.cpu().tolist())
           val_samples += Y_batch.size(0)


    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct_val_preds / val_samples

    return val_loss, val_acc, all_test_predictions



#---------------------------------------------
# Get predictions
#---------------------------------------------
def get_predictions(model, data_loader):
    """Function that returns all model predictions and
    respective target values.

    Args:
        model(nn.Module): The trained PyTorch model.
        test_loader (DataLoader): The DataLoader object that provides the batches.
                                  dataset split.

    Returns:
        all_preds (np.ndarray): All the model prediction generated during training.
        all_targets(np.ndarray): Numpy array storing the ground-truth targets.
    """

    # Initiate Testing
    model.eval()

    # # Init Lists
    all_preds = []
    all_targets = []

    # Testing
    with torch.no_grad():
       # Loop over batches from the DataLoader
       for X_batch, Y_batch in data_loader:
           # Test Preds
           preds = model(X_batch)

           # Compute Accuracy
           _, predicted = torch.max(preds, dim=1)
           all_preds.extend(predicted.cpu().numpy())
           all_targets.extend(Y_batch.cpu().numpy())

    return all_preds, all_targets

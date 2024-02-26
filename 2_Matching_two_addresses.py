import streamlit as st
import pandas as pd
import torch
import pickle
from tqdm import tqdm
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

########################################################################################################################################################
max_len = 75
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tags = ['society_name', 'flat_apartment_number', 'landmark', 'street',
        'area_locality_name', 'city_town', 'pincode', 'sub_locality', 'unknown']
num_tags = len(tags)
with open('svenk_labeled_data.p', 'rb') as f:
    train_data = pickle.load(f)
sentences_train, labels_train = zip(*train_data)
words = list(
    set([word for address in sentences_train for word in address.split()]))
words.append("ENDPAD")
word2idx = {w: i+1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
########################################################################################################################################################


def predict_tags(model, input_address):
    # Turn off gradient calculations
    with torch.no_grad():
        # Tokenize the address
        tokens = input_address.split()

        # Convert tokens to indices
        indices = [word2idx.get(token, 0) for token in tokens]

        # Pad the sequence
        padded_indices = indices + [0] * (max_len - len(indices))

        # Convert to tensor and add batch dimension
        tensor = torch.tensor(
            padded_indices, dtype=torch.long).unsqueeze(0).to(device)

        # Get predicted tag indices
        predictions = model(tensor, [len(indices)])

        # Find the tag with the highest probability for each word
        _, predicted_indices = torch.max(predictions, dim=2)

        # Remove batch dimension and convert to list
        predicted_indices = predicted_indices.squeeze(0).tolist()

        # Convert tag indices to tags
        predicted_tags = [tags[index]
                          for index in predicted_indices[:len(indices)]]

        return predicted_tags

########################################################################################################################################################


class RecurrentNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers):
        super(RecurrentNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_network = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fully_connected_network = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(
            embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm_network(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fully_connected_network(output)
        return output


########################################################################################################################################################
# ... previous code ...

# Create an instance of the BiLSTM model
model = RecurrentNet(len(words) + 1, 300, 300, num_tags, num_layers=2)

# Load the state dictionary of the saved model, map it to CPU device
model.load_state_dict(torch.load('trained_model.pt',
                      map_location=torch.device('cpu')))

# Move the model to the appropriate device
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# ... continue with your code ...
########################################################################################################################################################
input_address = "42 akshaya apartment maharshi karve road mumbai maharashtra 400021"
predicted_tags = predict_tags(model, input_address)

print(predicted_tags)
########################################################################################################################################################
st.title("Address Matcher and Schema Normalisation")

# Input Fields and Submit Button
input_address_1 = st.text_input("Input 1")
input_address_2 = st.text_input("Input 2")

submitted = st.button("Submit")

if submitted:
    # Display Table
    table_data = {'Input 1': [input_address_1], 'Input 2': [input_address_2]}
    df = pd.DataFrame(table_data)
    st.table(df)

    # Display Output Fields
    st.subheader("Matching Results")
    st.write("Output 1 value")

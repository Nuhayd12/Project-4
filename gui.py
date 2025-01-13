import tkinter as tk
from tkinter import messagebox
import numpy as np
from keras.models import load_model
import datetime
import re

# Load the trained model and related data
encoder_model = load_model("Data/encoder_model.keras")
decoder_model = load_model("Data/decoder_model.keras")

input_features_dict = np.load("Data/input_features_dict.npy", allow_pickle=True).item()
target_features_dict = np.load("Data/target_features_dict.npy", allow_pickle=True).item()
reverse_target_features_dict = np.load("Data/reverse_target_features_dict.npy", allow_pickle=True).item()
max_lengths = np.load("Data/max_lengths.npy")
max_encoder_seq_length, max_decoder_seq_length = max_lengths

# Utility functions
vowels = "AEIOUaeiou"

def decode_sequence(input_seq):
    # Encode the input sequence to get the initial states
    states_value = encoder_model.predict(input_seq)

    # Generate an empty target sequence with the start token
    target_seq = np.zeros((1, 1, len(target_features_dict)))
    target_seq[0, 0, target_features_dict["<START>"]] = 1.0

    decoded_sentence = ""
    while True:
        # Predict the next token and states
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Get the most probable token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]

        # Stop condition: End of sequence or max length
        if sampled_token == "<END>" or len(decoded_sentence.split()) > max_decoder_seq_length:
            break

        decoded_sentence += " " + sampled_token

        # Update target sequence and states
        target_seq = np.zeros((1, 1, len(target_features_dict)))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]

    return decoded_sentence.strip()

def translate_word():
    word = entry.get().strip()
    if not word:
        messagebox.showerror("Error", "Please enter a word.")
        return

    if word[0] in vowels:
        now = datetime.datetime.now()
        if not (21 <= now.hour < 22):  # Allow translations only between 9 PM and 10 PM
            messagebox.showerror("Error", "Words starting with vowels are allowed only between 9 PM and 10 PM.")
            return

    input_seq = string_to_matrix(word)
    translation = decode_sequence(input_seq)
    output_label["text"] = f"Translation: {translation}"

def string_to_matrix(word):
    tokens = re.findall(r"[\w']+|[^\s\w]", word)
    user_input_matrix = np.zeros((1, max_encoder_seq_length, len(input_features_dict)))
    for timestep, token in enumerate(tokens):
        if token in input_features_dict:
            user_input_matrix[0, timestep, input_features_dict[token]] = 1.0
    return user_input_matrix

# GUI setup
root = tk.Tk()
root.title("English to Hindi Translator")

frame = tk.Frame(root)
frame.pack(pady=20)

label = tk.Label(frame, text="Enter English word:")
label.grid(row=0, column=0, padx=5)

entry = tk.Entry(frame, width=30)
entry.grid(row=0, column=1, padx=5)

translate_button = tk.Button(frame, text="Translate", command=translate_word)
translate_button.grid(row=1, column=0, columnspan=2, pady=10)

output_label = tk.Label(root, text="Translation will appear here", fg="blue")
output_label.pack()

root.mainloop()

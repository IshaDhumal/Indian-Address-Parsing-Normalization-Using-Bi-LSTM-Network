import streamlit as st
import pandas as pd
import re
from difflib import SequenceMatcher

############################################################################################################################################################################
# Function to remove non-ASCII characters from a string


def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

############################################################################################################################################################################


def match_address(test_address, num_matches=1, similarity_threshold=0.9):
    # Reverse the parts of the test address
    test_parts = test_address.split()[::-1]

    if not any(part.isdigit() and len(part) == 6 for part in test_parts):
        # Test address does not contain a 6-digit number part, return None
        return None

    matches = []
    ratios = []

    for _, row in train_data.iterrows():
        # Reverse the parts of the train address
        train_parts = row[address_column].split()[::-1]

        # Compare each part of the addresses
        ratio = 0
        for test_part in test_parts:
            best_ratio = 0
            for train_part in train_parts:
                part_ratio = SequenceMatcher(
                    None, train_part, test_part).ratio()
                best_ratio = max(best_ratio, part_ratio)
            ratio += best_ratio

        ratio /= len(test_parts)  # Calculate the average ratio

        if ratio >= similarity_threshold:
            ratios.append(ratio)
            matches.append(row[address_column])

    # Sort the matches based on the ratio in descending order
    sorted_matches = [x for _, x in sorted(zip(ratios, matches), reverse=True)]

    # Check if pincode or city match exists
    for match in sorted_matches:
        # Reverse the parts of the match address
        match_parts = match.split()[::-1]
        if any(part.isdigit() and len(part) == 6 for part in match_parts):
            # Pincode match found, return the matches
            return sorted_matches[:num_matches]
        elif any(part in match_parts for part in test_parts):
            # City match found, return the matches
            return sorted_matches[:num_matches]

    # No pincode or city match found, return None
    return None
############################################################################################################################################################################


st.header('Addresses Database Upload')
uploaded_file = st.file_uploader(
    'Upload your address data in ".csv" format. The address file you upload, should have only one column named "address"', type='csv')


if uploaded_file is not None:
    train_data = pd.read_csv(uploaded_file)
    address_column = 'address'
    train_data[address_column] = train_data[address_column].apply(
        remove_non_ascii)
    st.write(train_data.head())
    st.write('Your data has been uploaded successfully')
    test = st.text_input('Enter address to match:')
    test_data = [test]


Match = st.button('Match')
if Match:
    print('hello')
    for test_address in test_data:
        closest_matches = match_address(
            test_address, num_matches=1, similarity_threshold=0.9)
        st.write(f"Test Address: {test_address}")
        if closest_matches:
            st.write(f"Closest Matches:")
            for match in closest_matches:
                st.write(match)
        else:
            st.write("Closest Matches: None")
        st.write()
else:
    st.write('Please click on the "Match" button to get the matches')

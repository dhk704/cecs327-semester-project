import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the pattern for the 'acc_norm,none' value in the JSON file
pattern = r'acc_norm,none'

def extract_and_plot():
    # Specify the paths to the folders containing the .json files
    folder_0shot = './arc-0-shot/'
    folder_5shot = './arc-5-shot/'
    folder_10shot = './arc-10-shot/'

    # Initialize lists to store the extracted values and their corresponding labels
    shots = ['0-shot', '5-shot', '10-shot']
    values_0shot = []
    labels_0shot = []
    values_5shot = []
    labels_5shot = []
    values_10shot = []
    labels_10shot = []

    # Iterate through each specified JSON file for the 0-shot category
    for filename in os.listdir(folder_0shot):
        if filename.endswith('.json'):
            # Open each JSON file and load its content into a dictionary
            with open(os.path.join(folder_0shot, filename), 'r') as file:
                data = json.load(file)

                # Extract the 'acc_norm,none' value from the 'results' category
                for category in data['results'].values():
                    if pattern in category:
                        values_0shot.append(float(category[pattern]))
                        labels_0shot.append(filename)

    # Iterate through each specified JSON file for the 5-shot category
    for filename in os.listdir(folder_5shot):
        if filename.endswith('.json'):
            # Open each JSON file and load its content into a dictionary
            with open(os.path.join(folder_5shot, filename), 'r') as file:
                data = json.load(file)

                # Extract the 'acc_norm,none' value from the 'results' category
                for category in data['results'].values():
                    if pattern in category:
                        values_5shot.append(float(category[pattern]))
                        labels_5shot.append(filename)

    # Iterate through each specified JSON file for the 10-shot category
    for filename in os.listdir(folder_10shot):
        if filename.endswith('.json'):
            # Open each JSON file and load its content into a dictionary
            with open(os.path.join(folder_10shot, filename), 'r') as file:
                data = json.load(file)

                # Extract the 'acc_norm,none' value from the 'results' category
                for category in data['results'].values():
                    if pattern in category:
                        values_10shot.append(float(category[pattern]))
                        labels_10shot.append(filename)

    # Create a grouped bar chart using matplotlib
    plt.figure(figsize=(15, 5))
    width = 0.35

    plt.bar(np.arange(len(shots)), [np.mean(values_0shot), np.mean(values_5shot), np.mean(values_10shot)], width, label=shots)
    plt.xticks(np.arange(len(shots)) + width/2, shots, rotation=45)

    plt.legend()
    plt.ylabel('Accuracy')
    plt.title('Shots')

    # Show the plot
    plt.show()

# Call the function to extract and display the results
extract_and_plot()


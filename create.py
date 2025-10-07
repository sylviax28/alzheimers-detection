#!/usr/bin/env python3

import os
import pandas as pd
from sklearn.model_selection import train_test_split


image_data = []

# loop through the folders in resized
for class_name in os.listdir("resized"):
    class_path = os.path.join("resized", class_name)
    
    # loop through each file in their respective folders
    for file_name in os.listdir(class_path):
        if file_name.startswith('.'):
            continue
        image_data.append([os.path.join(class_path, file_name), class_name])

# create a pandas dataframe
df = pd.DataFrame(image_data, columns=["filepath", "label"])

# train_df = df[:int(len(df)*0.7)]
# val_df = df[int(len(df)*0.7):int(len(df)*0.85)]
# test_df = df[int(len(df)*0.85):]
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)


# save CSVs
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

# sanity checks
print("Train labels:\n", train_df['label'].value_counts())
print("Val labels:\n", val_df['label'].value_counts())
print("Test labels:\n", test_df['label'].value_counts())

# ensure no overlaps
train_paths = set(train_df["filepath"])
val_paths = set(val_df["filepath"])
test_paths = set(test_df["filepath"])

print("Train-Val overlap:", train_paths.intersection(val_paths))
print("Train-Test overlap:", train_paths.intersection(test_paths))
print("Val-Test overlap:", val_paths.intersection(test_paths))

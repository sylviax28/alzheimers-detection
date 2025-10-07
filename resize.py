#!/usr/bin/env python3

import os 
from PIL import Image

width = 224
height = 224
impairment = "database/impairment"
noimpairment = "database/noimpairment"
resized_impairment = "resized/impairment"
resized_noimpairment = "resized/noimpairment"

def resize_images(source, dest):
    for filename in os.listdir(source):
        file_path = os.path.join(source, filename)
        
        if filename.lower().endswith('.jpg'):
            try:
                with Image.open(file_path) as img:
                    print(f"Found file: {filename}")

                    img_resized = img.resize((width, height))
                    
                    output_path = os.path.join(dest, filename)
                    img_resized.save(output_path)
                    
                    print(f"Successfully resized {filename}")
                    
            except Exception as event:
                print(f"Error resizing {filename}; {event}")
                
print("Processing impairment images")
resize_images(impairment, resized_impairment)

print("Processing no-impairment images")
resize_images(noimpairment, resized_noimpairment)
    
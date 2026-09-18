path = 'outputs'
import json
import os


import glob

extensions = ['json']
# extensions = ["jpg", "jpeg", "png", "webp"]

files = []

for ext in extensions:
    files.extend(
        glob.glob(
            os.path.join(path, "**", f"*.{ext}"),
            recursive=True
        )
    )
    
    

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        # json_data = f.read()
        # 
        # json_data['image']
        json_data = json.load(f)
        
        images = json_data['images']
        for img_meta in images:
            img_file = img_meta['file']
            img_prompt = img_meta['prompt']
            print(img_file, img_prompt)
            dirname = os.path.dirname(img_file)
            fname = os.path.splitext(os.path.basename(img_file))[0]
            # print(dirname, fname)
            prompt_file = os.path.join(dirname, f"{fname}.txt")
            
            with open(prompt_file, "w", encoding="utf-8") as pf:
                pf.write(img_prompt)
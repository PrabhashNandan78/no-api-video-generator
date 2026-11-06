import json

up = r"C:/Users/prabh/.gemini/antigravity/brain/44167fd8-c10f-4e90-8b59-cc564dc6921c/.user_uploaded/media_1793770001549_19bbf8f0.ipynb"
with open(up, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    text = "".join(c.get("source", []))[:50].replace("\n", " ")
    print(f"Cell {i:02d} [{c['cell_type']}]: {text}")


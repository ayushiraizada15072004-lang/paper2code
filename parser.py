import fitz

doc = fitz.open("paper.pdf")

text = ""

for page in doc:
    text += page.get_text()

with open("paper_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Text extracted and saved to paper_text.txt")
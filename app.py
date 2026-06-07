import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

st.title("📄 AI Research Paper Summarizer")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with open("paper.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF Uploaded Successfully")

    if st.button("Generate Summary & Code"):
        os.system("python main.py")
        if os.path.exists("summary.txt"):
            with open("summary.txt", "r", encoding="utf-8") as f:
                summary = f.read()
            st.subheader("📌 Summary")
            st.write(summary)
            st.download_button("Download Summary", summary, file_name="summary.txt")
            if not os.path.exists("history"):
                os.makedirs("history")
            count = len(os.listdir("history"))
            with open(f"history/summary_{count+1}.txt", "w", encoding="utf-8") as h:
                h.write(summary)
        if os.path.exists("generated_code.txt"):
            with open("generated_code.txt", "r", encoding="utf-8") as f:
                code = f.read()
            st.subheader("💻 Generated Code")
            st.code(code, language="python")
            st.download_button("Download Code", code, file_name="generated_code.py")

st.divider()
st.header("🤖 Chat With PDF")
question = st.text_input("Ask anything about your PDF")

if st.button("Ask Question"):
    if os.path.exists("paper_text.txt"):
        with open("paper_text.txt", "r", encoding="utf-8") as f:
            pdf_text = f.read()
        prompt = f"""
        Answer the question using this PDF.
        PDF Content:
        {pdf_text[:12000]}
        Question:
        {question}
        """
        response = model.generate_content(prompt)
        st.subheader("Answer")
        st.write(response.text)
    else:
        st.error("Generate summary first.")

st.divider()
st.header("📚 Summary History")
if os.path.exists("history"):
    files = os.listdir("history")
    for file in files:
        st.write(file)

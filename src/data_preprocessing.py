import re
from pathlib import Path

import pandas as pd
import spacy

from .utils.constants import PROCESSED_DATA_DIR, RAW_DATA_DIR, TEXT_FILE_PATTERN


class DataPreprocessor:
    def __init__(self, data_dir):
        """Initialize the preprocessor with data directory path."""
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.nlp = spacy.load("en_core_web_sm")

    def load_text_files(self):
        """Load and organize raw text files from data directory."""
        documents = []
        for file_path in self.data_dir.glob(TEXT_FILE_PATTERN):
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(
                    {"file_name": file_path.name, "text": f.read(), "processed": False}
                )
        return documents

    def clean_text(self, text):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        return text.strip().lower()

    def extract_sentences(self, text):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def process_documents(self, documents):
        processed_data = []

        for doc in documents:
            clean_text = self.clean_text(doc["text"])
            sentences = self.extract_sentences(clean_text)

            for sent in sentences:
                processed_data.append(
                    {"file_name": doc["file_name"], "sentence": sent, "processed": True}
                )

        return pd.DataFrame(processed_data)

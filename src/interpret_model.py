# src/interpret_model.py
import logging
import os
from typing import List, Dict, Tuple  # Added Dict import
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import shap
import lime
from lime.lime_text import LimeTextExplainer
import numpy as np
import regex as re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='interpretability.log'
)

class ModelInterpreter:
    def __init__(self,
                 model_path: str = 'models/ner_model/ner/model',  # Updated path
                 val_path: str = 'data/labeled/val.conll',
                 shap_output: str = 'data/interpretability/shap_report.txt',
                 lime_output: str = 'data/interpretability/lime_report.txt',
                 difficult_cases_output: str = 'data/interpretability/difficult_cases.txt'):
        """Initialize with model and output paths."""
        self.model_path = model_path
        self.val_path = val_path
        self.shap_output = shap_output
        self.lime_output = lime_output
        self.difficult_cases_output = difficult_cases_output
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
            logging.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
        self.labels = ['O', 'B-PRODUCT', 'I-PRODUCT', 'B-PRICE', 'I-PRICE', 'B-LOC', 'I-LOC']
        self.class_names = ['PRODUCT', 'PRICE', 'LOC', 'O']  # For LIME

    def load_conll(self) -> List[Tuple[str, List[Tuple[str, str]]]]:
        """Load CoNLL data into sentences and labels."""
        try:
            sentences = []
            current_tokens = []
            current_labels = []
            with open(self.val_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('-DOCSTART-'):
                        continue
                    elif line == '':
                        if current_tokens:
                            text = ' '.join(current_tokens)
                            sentences.append((text, current_labels))
                            current_tokens = []
                            current_labels = []
                    else:
                        try:
                            token, label = line.split('\t')
                            current_tokens.append(token)
                            current_labels.append(label)
                        except ValueError:
                            logging.warning(f"Skipping malformed line: {line}")
                            continue
            if current_tokens:
                text = ' '.join(current_tokens)
                sentences.append((text, current_labels))
            logging.info(f"Loaded {len(sentences)} sentences from {self.val_path}")
            return sentences[:10]  # Limit to 10 for efficiency
        except Exception as e:
            logging.error(f"Error loading CoNLL: {str(e)}")
            raise

    def predict_ner(self, text: str) -> List[Dict]:
        """Run NER pipeline on text."""
        try:
            return self.pipeline(text)
        except Exception as e:
            logging.error(f"Error in NER prediction for text '{text}': {str(e)}")
            return []

    def shap_explain(self, texts: List[str]) -> List[str]:
        """Generate SHAP explanations for NER predictions."""
        try:
            explainer = shap.Explainer(self.predict_ner, masker=shap.maskers.Text(self.tokenizer))
            shap_values = explainer(texts)
            explanations = []
            for i, text in enumerate(texts):
                explanations.append(f"Text: {text}")
                for label in self.class_names:
                    if label in shap_values[i].output_names:
                        values = shap_values[i].values[:, shap_values[i].output_names.index(label)]
                        tokens = shap_values[i].data
                        top_indices = np.argsort(-values)[:5]  # Top 5 contributing tokens
                        explanations.append(f"  Label: {label}")
                        for idx in top_indices:
                            if idx < len(tokens):
                                explanations.append(f"    Token: {tokens[idx]}, Contribution: {values[idx]:.4f}")
            logging.info("Generated SHAP explanations")
            return explanations
        except Exception as e:
            logging.error(f"Error in SHAP explanation: {str(e)}")
            return []

    def lime_explain(self, texts: List[str]) -> List[str]:
        """Generate LIME explanations for NER predictions."""
        try:
            explainer = LimeTextExplainer(class_names=self.class_names)
            explanations = []
            for text in texts:
                exp = explainer.explain_instance(
                    text,
                    lambda x: self._lime_predict_proba(x),
                    num_features=5,
                    labels=self.class_names
                )
                explanations.append(f"Text: {text}")
                for label in self.class_names:
                    label_exp = exp.as_list(label=label)
                    explanations.append(f"  Label: {label}")
                    for feature, weight in label_exp:
                        explanations.append(f"    Feature: {feature}, Weight: {weight:.4f}")
            logging.info("Generated LIME explanations")
            return explanations
        except Exception as e:
            logging.error(f"Error in LIME explanation: {str(e)}")
            return []

    def _lime_predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities for LIME."""
        probs = []
        for text in texts:
            predictions = self.predict_ner(text)
            token_probs = np.zeros((len(self.class_names),))
            for pred in predictions:
                entity = pred['entity'].split('-')[-1] if '-' in pred['entity'] else pred['entity']
                if entity in self.class_names:
                    token_probs[self.class_names.index(entity)] += pred['score']
            token_probs /= max(1, sum(token_probs))  # Normalize
            probs.append(token_probs)
        return np.array(probs)

    def analyze_difficult_cases(self, sentences: List[Tuple[str, List[Tuple[str, str]]]]) -> List[str]:
        """Identify and analyze difficult cases."""
        try:
            difficult_cases = []
            price_pattern = r'^\d+[,\d]*ብር?$|^ዋጋ$|^price$|^birr$'
            location_indicators = {'አድራሻ', 'ለቡ', 'መገናኛ', 'ሞል'}
            product_keywords = {'spatulas', 'slicer', 'bottle', 'bag'}

            for text, true_labels in sentences:
                predictions = self.predict_ner(text)
                pred_labels = ['O'] * len(text.split())
                for pred in predictions:
                    entity = pred['entity'].split('-')[-1] if '-' in pred['entity'] else pred['entity']
                    start, end = pred['start'], pred['end']
                    token_idx = len(text[:start].split()) - 1
                    if token_idx < len(pred_labels):
                        pred_labels[token_idx] = f"B-{entity}" if pred_labels[token_idx] == 'O' else f"I-{entity}"

                for idx, (token, true_label, pred_label) in enumerate(zip(text.split(), true_labels, pred_labels)):
                    if true_label != pred_label:
                        context = ' '.join(text.split()[max(0, idx-2):idx+3])
                        difficult_cases.append(f"Text: {context}")
                        difficult_cases.append(f"  Token: {token}")
                        difficult_cases.append(f"  True Label: {true_label}")
                        difficult_cases.append(f"  Predicted Label: {pred_label}")
                        # Analyze why it might be difficult
                        reason = "Unknown"
                        if re.match(price_pattern, token):
                            reason = "Ambiguous price format"
                        elif token in location_indicators:
                            reason = "Location term confusion"
                        elif token.lower() in product_keywords:
                            reason = "Product name overlap"
                        difficult_cases.append(f"  Possible Reason: {reason}")
                        difficult_cases.append("")
            logging.info(f"Identified {len(difficult_cases)//5} difficult cases")
            return difficult_cases
        except Exception as e:
            logging.error(f"Error analyzing difficult cases: {str(e)}")
            return []

    def save_reports(self, shap_explanations: List[str], lime_explanations: List[str], difficult_cases: List[str]):
        """Save interpretability reports."""
        try:
            os.makedirs(os.path.dirname(self.shap_output), exist_ok=True)
            with open(self.shap_output, 'w', encoding='utf-8') as f:
                f.write('\n'.join(shap_explanations))
            logging.info(f"Saved SHAP report to {self.shap_output}")

            with open(self.lime_output, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lime_explanations))
            logging.info(f"Saved LIME report to {self.lime_output}")

            with open(self.difficult_cases_output, 'w', encoding='utf-8') as f:
                f.write('\n'.join(difficult_cases))
            logging.info(f"Saved difficult cases to {self.difficult_cases_output}")
        except Exception as e:
            logging.error(f"Error saving reports: {str(e)}")
            raise

    def process(self):
        """Run interpretability analysis."""
        try:
            sentences = self.load_conll()
            texts = [text for text, _ in sentences]
            shap_explanations = self.shap_explain(texts)
            lime_explanations = self.lime_explain(texts)
            difficult_cases = self.analyze_difficult_cases(sentences)
            self.save_reports(shap_explanations, lime_explanations, difficult_cases)
            logging.info("Completed interpretability analysis")
        except Exception as e:
            logging.error(f"Error in process: {str(e)}")
            raise

def main():
    interpreter = ModelInterpreter()
    interpreter.process()

if __name__ == "__main__":
    main()
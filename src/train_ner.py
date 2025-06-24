# src/train_ner.py
import spacy
from spacy.training import Example
import random
import logging
from typing import List, Tuple
import os
from sklearn.metrics import precision_recall_fscore_support

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='training.log'
)

class NERTrainer:
    def __init__(self, 
                 train_path: str = 'data/labeled/train.conll',
                 val_path: str = 'data/labeled/val.conll',
                 test_path: str = 'data/labeled/test.conll',
                 output_model: str = 'models/ner_model',
                 eval_output: str = 'data/evaluation/ner_evaluation.txt'):
        """Initialize with input and output paths."""
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.output_model = output_model
        self.eval_output = eval_output
        self.nlp = spacy.blank('am')  # Amharic blank model
        self.labels = ['PRODUCT', 'PRICE', 'LOC']

    def load_conll(self, file_path: str) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
        """Load CoNLL data into spaCy format."""
        try:
            data = []
            current_tokens = []
            current_entities = []
            start_idx = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('-DOCSTART-'):
                        continue
                    elif line == '':
                        if current_tokens:
                            text = ' '.join(current_tokens)
                            data.append((text, {'entities': current_entities}))
                            current_tokens = []
                            current_entities = []
                            start_idx = 0
                    else:
                        token, label = line.split('\t')
                        current_tokens.append(token)
                        if label.startswith('B-'):
                            entity_type = label[2:]
                            current_entities.append((start_idx, start_idx + len(token), entity_type))
                        elif label.startswith('I-'):
                            if current_entities and current_entities[-1][2] == label[2:]:
                                current_entities[-1] = (current_entities[-1][0], start_idx + len(token), label[2:])
                        start_idx += len(token) + 1
            logging.info(f"Loaded {len(data)} sentences from {file_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading CoNLL: {str(e)}")
            raise

    def train_model(self, n_iter: int = 10):
        """Train NER model on training and validation data."""
        try:
            train_data = self.load_conll(self.train_path)
            val_data = self.load_conll(self.val_path)

            if 'ner' not in self.nlp.pipe_names:
                ner = self.nlp.add_pipe('ner')
            else:
                ner = self.nlp.get_pipe('ner')

            for label in self.labels:
                ner.add_label(label)

            other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
            with self.nlp.disable_pipes(*other_pipes):
                optimizer = self.nlp.begin_training()
                for itn in range(n_iter):
                    random.shuffle(train_data)
                    losses = {}
                    for text, annotations in train_data:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        self.nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
                    logging.info(f"Iteration {itn}, Losses: {losses}")

            os.makedirs(self.output_model, exist_ok=True)
            self.nlp.to_disk(self.output_model)
            logging.info(f"Saved model to {self.output_model}")
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise

    def evaluate_model(self):
        """Evaluate model on test data."""
        try:
            test_data = self.load_conll(self.test_path)
            true_labels = []
            pred_labels = []

            for text, annotations in test_data:
                doc = self.nlp(text)
                true_ents = [(ent[0], ent[1], ent[2]) for ent in annotations['entities']]
                pred_ents = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

                # Align true and predicted labels
                tokens = text.split()
                start_idx = 0
                for token in tokens:
                    token_span = (start_idx, start_idx + len(token))
                    true_label = 'O'
                    pred_label = 'O'

                    for start, end, label in true_ents:
                        if token_span[0] >= start and token_span[1] <= end:
                            true_label = label
                            break

                    for start, end, label in pred_ents:
                        if token_span[0] >= start and token_span[1] <= end:
                            pred_label = label
                            break

                    true_labels.append(true_label)
                    pred_labels.append(pred_label)
                    start_idx += len(token) + 1

            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, labels=self.labels, average='weighted')
            report = [
                f"NER Model Evaluation Report",
                f"Test Sentences: {len(test_data)}",
                f"Precision: {precision:.2%}",
                f"Recall: {recall:.2%}",
                f"F1-Score: {f1:.2%}"
            ]
            os.makedirs(os.path.dirname(self.eval_output), exist_ok=True)
            with open(self.eval_output, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            logging.info(f"Saved evaluation to {self.eval_output}")
            return precision, recall, f1
        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")
            raise

    def process(self):
        """Run training and evaluation."""
        try:
            self.train_model()
            precision, recall, f1 = self.evaluate_model()
            logging.info(f"Final Metrics - Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")
        except Exception as e:
            logging.error(f"Error in process: {str(e)}")
            raise

def main():
    trainer = NERTrainer()
    trainer.process()

if __name__ == "__main__":
    main()
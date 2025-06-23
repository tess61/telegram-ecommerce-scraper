import spacy
from spacy.training import Example
import random
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='training.log'
)

def load_conll(file_path: str) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    # Load CoNLL data and convert to spaCy format
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
    return data

def train_ner(train_path: str = 'data/labeled/train.conll', 
              val_path: str = 'data/labeled/val.conll',
              output_model: str = 'models/ner_model'):
    try:
        # Load data
        train_data = load_conll(train_path)
        val_data = load_conll(val_path)
        
        # Initialize spaCy model
        nlp = spacy.blank('am')  # Amharic model (or 'en' for mixed language)
        if 'ner' not in nlp.pipe_names:
            ner = nlp.add_pipe('ner')
        else:
            ner = nlp.get_pipe('ner')
        
        # Add labels
        for _, annotations in train_data:
            for ent in annotations['entities']:
                ner.add_label(ent[2])
        
        # Disable other pipes
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.begin_training()
            for itn in range(10):  # Adjust iterations
                random.shuffle(train_data)
                losses = {}
                for text, annotations in train_data:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
                logging.info(f"Iteration {itn}, Losses: {losses}")
        
        # Save model
        nlp.to_disk(output_model)
        logging.info(f"Saved model to {output_model}")
        
        # Evaluate on validation data
        # Add evaluation logic here if needed
        
    except Exception as e:
        logging.error(f"Error training NER model: {str(e)}")
        raise

def main():
    train_ner()

if __name__ == '__main__':
    main()
import os
import spacy
import re
from tqdm import tqdm
import pandas as pd
from codecarbon import EmissionsTracker

def count_pos(doc):
    """Counts parts of speech in the document."""
    pos_counts = {
        'NOUN': 0,
        'VERB': 0,
        'ADJ': 0,
        'ADV': 0
    }
    
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
    
    return pos_counts

def calculate_rel_freq(counter, doc_length):
    """Calculates relative frequency of a part of speech."""
    return round((counter / doc_length) * 10000, 2)

def count_entities(doc):
    """Counts named entities in the document."""
    entity_counts = {
        'PERSON': 0,
        'LOC': 0,
        'ORG': 0
    }
    
    for ent in doc.ents:
        if ent.label_ in entity_counts:
            entity_counts[ent.label_] += 1
    
    return entity_counts

def process_sub_dir(sub_dir_path, nlp):
    framedata = []

    for txt in tqdm(sorted(os.listdir(sub_dir_path))):
        with open(os.path.join(sub_dir_path, txt), "r", encoding="iso8859_10") as f:
            document = f.read()
        
        cleaned_text = re.sub(r'<[^>]+>', '', document)
        doc = nlp(cleaned_text)
        
        pos_counts = count_pos(doc)
        noun_rel_freq = calculate_rel_freq(pos_counts['NOUN'], len(doc))
        verb_rel_freq = calculate_rel_freq(pos_counts['VERB'], len(doc))
        adj_rel_freq = calculate_rel_freq(pos_counts['ADJ'], len(doc))
        adv_rel_freq = calculate_rel_freq(pos_counts['ADV'], len(doc))
        
        entity_counts = count_entities(doc)
        
        framedata.append((txt, 
                          noun_rel_freq, 
                          verb_rel_freq, 
                          adj_rel_freq,
                          adv_rel_freq, 
                          entity_counts['PERSON'],
                          entity_counts['LOC'], 
                          entity_counts['ORG']))

    return framedata

def save_to_csv(data, output_path):
    """Saves the data to a CSV file."""
    df = pd.DataFrame(data, columns=['Filename', 'RelFreq NOUN', 'RelFreq VERB', 'RelFreq ADJ', 'RelFreq ADV', 'Unique PER', 'Unique LOC', 'Unique ORG'])
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def main():
    input_path = "in"
    output_path = "out"

    tracker = EmissionsTracker(project_name="feature extraction",
                            experiment_id="feature_extraction",
                            output_dir=output_path,
                            output_file="feature_extraction_emissions.csv",
                            log_level="error")

    task_name = "load model"
    print(f"Starting task: {task_name}")
    tracker.start_task(task_name)
    nlp = spacy.load("en_core_web_md")
    print(f"Stopping task: {task_name}")
    tracker.stop_task(task_name)

    task_name = "process directories"
    print(f"Starting task: {task_name}")
    tracker.start_task(task_name)
    input_dirs = sorted(os.listdir(input_path))
    for sub_dir in input_dirs:
        sub_dir_path = os.path.join(input_path, sub_dir)
    
        print(f"Processing directory: {sub_dir}")
        framedata = process_sub_dir(sub_dir_path, nlp)
        out_path = os.path.join(output_path, f'annotations_{sub_dir}.csv')
        save_to_csv(framedata, out_path)

    print(f"Stopping task: {task_name}")
    tracker.stop_task(task_name)
    tracker.stop()

if __name__ == "__main__":
    main()

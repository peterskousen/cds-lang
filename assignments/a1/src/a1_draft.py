import os
import spacy
import re
import pandas as pd
nlp = spacy.load("en_core_web_md")

input_dir_path = os.path.join(
    "..",
    "in"
)

input_dir = os.listdir(input_dir_path)

for sub_dir in input_dir:
    noun_counter = 0
    verb_counter = 0
    adj_counter = 0
    adv_counter = 0
    per_counter = 0
    loc_counter = 0
    org_counter = 0

    if sub_dir == input_dir[:2]:
        break
    sub_dir_path = os.path.join(input_dir_path, sub_dir)

    for txt in os.listdir(sub_dir_path):
        txtpath = os.path.join(sub_dir_path, txt)
        with open(txtpath, "r", encoding="iso8859_10") as f:
            document = f.read()
        cleaned_text = re.sub(r'<[^>]+>', '', document)
        doc = nlp(cleaned_text)

    def pos_counter():      
        for token in doc:
            if token.pos_ == "NOUN":
                noun_counter += 1
            elif token.pos_ == "VERB":
                verb_counter += 1
            elif token.pos_ == "ADJ":
                adj_counter += 1
            elif token.pos_ == "ADV":
                adv_counter += 1
        return noun_counter, verb_counter, adj_counter, adv_counter
    pos_counter()

    def calculate_rel_freq():
        noun_rel_freq = round((noun_counter/len(doc) * 10000), 2)
        verb_rel_freq = round((verb_counter/len(doc) * 10000), 2)
        adj_rel_freq = round((adj_counter/len(doc) * 10000), 2)
        adv_rel_freq = round((adv_counter/len(doc) * 10000), 2)
        return noun_rel_freq, verb_rel_freq, adj_rel_freq, adv_rel_freq
    calculate_rel_freq()

    def ent_counter():
        for ent in doc.ents: 
            if ent.label_ == "PERSON":
                per_counter += 1
            elif ent.label_ == "LOC":
                loc_counter += 1
            elif ent.label_ == "ORG":
                org_counter += 1
            else:
                pass
        return per_counter, loc_counter, org_counter
        print(txt)
        print(noun_rel_freq)
        print(verb_rel_freq)
        print(adj_rel_freq)
        print(adv_rel_freq)
    ent_counter()

    print(f'nouns: {noun_counter}, verbs: {verb_counter}, adjectives: {adj_counter}, adverbs: {adv_counter}')
    print(f'rel nouns: {noun_rel_freq}, rel verbs: {verb_rel_freq}, rel adjectives: {adj_rel_freq}, rel adverbs: {adv_rel_freq}')
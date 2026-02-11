#!/usr/bin/env python3
"""json_to_bio.py - Convert NER corpus to BIO format

Extracted from the provided Jupyter notebook (BIO.ipynb).
Requires: estnltk, regex
"""

import os
import json
import argparse
import regex as re
from estnltk import Text
from estnltk.taggers.standard.text_segmentation.compound_token_tagger import CompoundTokenTagger, ALL_1ST_LEVEL_PATTERNS
from estnltk.taggers.standard.text_segmentation.patterns import MACROS


# ============================================================================
# TOKENIZATION CORRECTION
# ============================================================================

def make_adapted_cp_tagger(**kwargs):
    """Creates an adapted CompoundTokenTagger that:
       1) excludes roman numerals from names with initials;
       2) does not join date-like token sequences as numbers;
    """
    # Pattern 1: Names with 2 initials (exclude titles and roman numerals I, V, X)
    redefined_pat_1 = {
        'comment': '*) Names starting with 2 initials (exclude titles and roman numerals I, V, X);',
        'pattern_type': 'name_with_initial',
        'example': 'A. H. Tammsaare',
        '_regex_pattern_': re.compile(r'''
            (?!(Dr\.|Lb\.|Lh\.|Lm\.|Ln\.|Lv\.|Lw\.|Pr\.))     # exclude titles
            ([ABCDEFGHJKLMNOPQRSTUWYZŠŽÕÄÖÜ][{LOWERCASE}]?)   # first initial
            \s?\.\s?-?                                        # period (and hyphen potentially)
            ([ABCDEFGHJKLMNOPQRSTUWYZŠŽÕÄÖÜ][{LOWERCASE}]?)   # second initial
            \s?\.\s?                                          # period
            ((\.[{UPPERCASE}]\.)?[{UPPERCASE}][{LOWERCASE}]+) # last name
        '''.format(**MACROS), re.X),
        '_group_': 0,
        '_priority_': (4, 1),
        'normalized': lambda m: re.sub('\1.\2. \3', '', m.group(0)),
    }

    # Pattern 2: Names with 1 initial (exclude roman numerals I, V, X)
    redefined_pat_2 = {
        'comment': '*) Names starting with one initial (exclude roman numerals I, V, X);',
        'pattern_type': 'name_with_initial',
        'example': 'A. Hein',
        '_regex_pattern_': re.compile(r'''
            ([ABCDEFGHJKLMNOPQRSTUWYZŠŽÕÄÖÜ])   # first initial
            \s?\.\s?                            # period
            ([{UPPERCASE}][{LOWERCASE}]+)       # last name
        '''.format(**MACROS), re.X),
        '_group_': 0,
        '_priority_': (4, 2),
        'normalized': lambda m: re.sub('\1. \2', '', m.group(0)),
    }

    # Pattern 3: Long numbers (1 group, corrected for timex tagger)
    redefined_number_pat_1 = {
        'comment': '*) A generic pattern for detecting long numbers (1 group) (corrected for timex tagger).',
        'example': '12,456',
        'pattern_type': 'numeric',
        '_group_': 0,
        '_priority_': (2, 1, 5),
        '_regex_pattern_': re.compile(r'''                             
            \d+           # 1 group of numbers
            (,\d+|\ *\.)  # + comma-separated numbers or period-ending
        ''', re.X),
        'normalized': r"lambda m: re.sub(r'[\s]' ,'' , m.group(0))"
    }

    # Pattern 4: Long numbers (2 groups, point-separated, followed by comma-separated)
    redefined_number_pat_2 = {
        'comment': '*) A generic pattern for detecting long numbers (2 groups, point-separated, followed by comma-separated numbers) (corrected for timex tagger).',
        'example': '67.123,456',
        'pattern_type': 'numeric',
        '_group_': 0,
        '_priority_': (2, 1, 3, 1),
        '_regex_pattern_': re.compile(r'''
            \d+\.+\d+   # 2 groups of numbers
            (,\d+)      # + comma-separated numbers
        ''', re.X),
        'normalized': r"lambda m: re.sub(r'[\s\.]' ,'' , m.group(0))"
    }

    # Build new pattern list
    new_1st_level_patterns = []
    for pat in ALL_1ST_LEVEL_PATTERNS:
        # Skip these patterns
        if pat['comment'] in [
            '*) Abbreviations of type <uppercase letter> + <numbers>;',
            '*) Date patterns that contain month as a Roman numeral: "dd. roman_mm yyyy";',
            '*) Date patterns in the commonly used form "dd/mm/yy";'
        ]:
            continue
        
        # Replace these patterns
        if pat['comment'] == '*) Names starting with 2 initials;':
            new_1st_level_patterns.append(redefined_pat_1)
        elif pat['comment'] == '*) Names starting with one initial;':
            new_1st_level_patterns.append(redefined_pat_2)
        elif pat['comment'] == '*) A generic pattern for detecting long numbers (1 group).':
            new_1st_level_patterns.append(redefined_number_pat_1)
        elif pat['comment'] == '*) A generic pattern for detecting long numbers (2 groups, point-separated, followed by comma-separated numbers).':
            new_1st_level_patterns.append(redefined_number_pat_2)
        else:
            new_1st_level_patterns.append(pat)
    
    assert len(new_1st_level_patterns) + 3 == len(ALL_1ST_LEVEL_PATTERNS)
    
    if kwargs and 'patterns_1' in kwargs:
        raise ValueError("Cannot overwrite 'patterns_1' in adapted CompoundTokenTagger.")
    
    return CompoundTokenTagger(
        patterns_1=new_1st_level_patterns,
        do_not_join_on_strings=('\n\n', '\n'),
        **kwargs
    )


# Initialize the adapted tagger
adapted_cp_tokens_tagger = make_adapted_cp_tagger(
    input_tokens_layer='tokens',
    output_layer='compound_tokens'
)


def preprocess_words(input_text):
    """Pre-processes Text object: adds word segmentation."""
    input_text.tag_layer('tokens')
    adapted_cp_tokens_tagger.tag(input_text)
    input_text.tag_layer('words')
    
    return input_text


# ============================================================================
# BIO CONVERSION
# ============================================================================

# Category mapping for unification
CATEGORY_MAPPING = {
    'LOC_ADDRESS': 'LOC',
    'ORG_GPE': 'ORG',
    'ORG_POL': 'ORG',
}


def unify_category(category):
    """Unify entity categories according to mapping."""
    return CATEGORY_MAPPING.get(category, category)


def convert_file_to_bio(file_path):
    """
    Convert a single JSON file to BIO format.
    Returns list of sentences, where each sentence is a list of (token, bio_tag) tuples.
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    text = data.get("text", "")
    txt = Text(text)
    txt = preprocess_words(txt)

    # Tag sentences
    txt.tag_layer('sentences')

    # Build word token list with sentence info
    words = []
    for sent_id, sent_span in enumerate(txt['sentences']):
        for w in txt['words']:
            # Check if word is within this sentence
            if w.start >= sent_span.start and w.end <= sent_span.end:
                norm = w.annotations[0].get('normalized_form') if w.annotations else None
                norm = norm or w.text                
                words.append({
                    "text": w.text,
                    "norm": norm,
                    "start": w.start,
                    "end": w.end,
                    "sent_id": sent_id
                })

    # Find NER layer
    ne_layer = None
    for candidate in ("ne_gold_a", "ne_gold_b"):
        ne_layer = next((l for l in data.get("layers", []) if l["name"] == candidate), None)
        if ne_layer:
            break

    if not ne_layer:
        return []

    spans = ne_layer.get("spans", [])

    # Create BIO tags for each word
    for word in words:
        word['bio_tag'] = 'O'  # Default: outside any entity

    # Process each NER span
    for span in spans:
        start, end = span["base_span"]
        etype = span["annotations"][0].get("tag", "?")
        etype = unify_category(etype)  # Unify categories

        # Find overlapping words
        overlapping = [w for w in words if w["start"] < end and w["end"] > start]

        if overlapping:
            # First token gets B- (Beginning)
            overlapping[0]['bio_tag'] = f'B-{etype}'
            # Rest get I- (Inside)
            for w in overlapping[1:]:
                w['bio_tag'] = f'I-{etype}'

    # Group words by sentence
    sentences = []
    current_sent_id = None
    current_sent = []

    for word in words:
        if word['sent_id'] != current_sent_id:
            if current_sent:
                sentences.append(current_sent)
            current_sent = []
            current_sent_id = word['sent_id']
        current_sent.append((word['norm'], word['bio_tag']))

    if current_sent:
        sentences.append(current_sent)

    return sentences


def convert_corpus_to_bio(input_dir, output_file):
    """
    Convert all JSON files to BIO format and save as TSV.
    Format: token<TAB>bio_tag, with double newline between sentences.
    """
    all_sentences = []

    print("Converting corpus to BIO format...")

    for fname in sorted(f for f in os.listdir(input_dir) if f.endswith(".json")):
        file_path = os.path.join(input_dir, fname)
        sentences = convert_file_to_bio(file_path)
        all_sentences.extend(sentences)
        print(f"  ✓ {fname}: {len(sentences)} sentences")

    # Write to TSV
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in all_sentences:
            for token, bio_tag in sentence:
                f.write(f"{token}\t{bio_tag}\n")
            f.write("\n")  # Double newline between sentences

    print(f"\n✓ Saved BIO corpus to: {output_file}")
    print(f"  Total sentences: {len(all_sentences)}")
    print(f"  Total tokens: {sum(len(s) for s in all_sentences)}")

    # Count entities by type
    entity_counts = {}
    for sentence in all_sentences:
        for token, bio_tag in sentence:
            if bio_tag.startswith('B-'):
                etype = bio_tag[2:]
                entity_counts[etype] = entity_counts.get(etype, 0) + 1

    print(f"\n  Entity counts:")
    for etype, count in sorted(entity_counts.items()):
        print(f"    {etype}: {count}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert NER corpus to BIO format")
    parser.add_argument("--filtered_dir", default="json_ne_gold_a",
                        help="Input directory with filtered JSON files")
    parser.add_argument("--bio_output_file", default="corpus_bio.tsv",
                        help="Output BIO format file")
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 1.3: Converting corpus to BIO format")
    print("=" * 80)

    convert_corpus_to_bio(args.filtered_dir, args.bio_output_file)

    print("\n" + "=" * 80)
    print("✓ BIO conversion complete!")
    print("=" * 80)
    print(f"Output file: {args.bio_output_file}")
    print("\nNext: Run split_train_dev_test.py")


if __name__ == "__main__":
    main()

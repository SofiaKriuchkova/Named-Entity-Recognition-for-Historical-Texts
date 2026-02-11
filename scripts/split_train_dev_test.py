#!/usr/bin/env python3
"""split_train_dev_test.py - Split NER corpus into train/dev/test sets

Extracted from the provided Jupyter notebook (BIO.ipynb).
Requires: estnltk, regex
"""

import os
import json
import random
import argparse
import sys
import regex as re
from estnltk import Text
from estnltk.taggers.standard.text_segmentation.compound_token_tagger import CompoundTokenTagger, ALL_1ST_LEVEL_PATTERNS
from estnltk.taggers.standard.text_segmentation.patterns import MACROS
from collections import defaultdict


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
    """Pre-processes Text object: adds word segmentation"""
    input_text.tag_layer('tokens')
    adapted_cp_tokens_tagger.tag(input_text)
    input_text.tag_layer('words')
    return input_text


# ============================================================================
# TRAIN/DEV/TEST SPLIT
# ============================================================================

# Exclude these entity types everywhere
EXCLUDED_ENTITY_TYPES = {"EVENT", "UNK"}

# Category mapping for unification
CATEGORY_MAPPING = {
    'LOC_ADDRESS': 'LOC',
    'ORG_GPE': 'ORG',
    'ORG_POL': 'ORG',
}


def unify_category(category: str) -> str:
    """Unify entity categories according to mapping."""
    return CATEGORY_MAPPING.get(category, category)


def get_ne_layer(data: dict):
    """Get the first available NER gold layer."""
    for candidate in ("ne_gold_a", "ne_gold_b"):
        layer = next((l for l in data.get("layers", []) if l.get("name") == candidate), None)
        if layer:
            return layer
    return None


def get_file_statistics(file_path: str) -> dict:
    """
    Get statistics for a single file: tokens, sentences, entities (EXCLUDING EVENT/UNK).
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    text = data.get("text", "")
    txt = Text(text)
    txt = preprocess_words(txt)
    txt.tag_layer('sentences')

    num_tokens = len(txt['words'])
    num_sentences = len(txt['sentences'])

    # Count entities excluding EVENT/UNK
    ne_layer = get_ne_layer(data)
    num_entities = 0
    if ne_layer:
        for span in ne_layer.get("spans", []):
            tag = span.get("annotations", [{}])[0].get("tag", "?")
            etype = unify_category(tag)
            if etype in EXCLUDED_ENTITY_TYPES:
                continue
            num_entities += 1

    return {
        'tokens': num_tokens,
        'sentences': num_sentences,
        'entities': num_entities
    }


def convert_file_to_bio_sentences(file_path: str):
    """Convert a file to BIO format sentences (EXCLUDING EVENT/UNK spans)."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    text = data.get("text", "")
    txt = Text(text)
    txt = preprocess_words(txt)
    txt.tag_layer('sentences')

    # Build word list
    words = []
    for sent_id, sent_span in enumerate(txt['sentences']):
        for w in txt['words']:
            if w.start >= sent_span.start and w.end <= sent_span.end:
                norm = w.annotations[0].get('normalized_form') if w.annotations else None
                norm = norm or w.text
                words.append({
                    "norm": norm,
                    "start": w.start,
                    "end": w.end,
                    "sent_id": sent_id,
                    "bio_tag": 'O'
                })

    # Apply NER spans
    ne_layer = get_ne_layer(data)
    if ne_layer:
        for span in ne_layer.get("spans", []):
            start, end = span["base_span"]
            tag = span.get("annotations", [{}])[0].get("tag", "?")
            etype = unify_category(tag)

            # Exclude these types completely
            if etype in EXCLUDED_ENTITY_TYPES:
                continue

            overlapping = [w for w in words if w["start"] < end and w["end"] > start]
            if overlapping:
                overlapping[0]['bio_tag'] = f'B-{etype}'
                for w in overlapping[1:]:
                    w['bio_tag'] = f'I-{etype}'

    # Group by sentence
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


def split_corpus(input_dir: str, test_size=0.10, dev_size=0.09, random_seed=42):
    """
    Split corpus into train/dev/test by files (protocols).
    Returns: dict with keys train/dev/test including files, data, stats, entities.
    """
    random.seed(random_seed)

    # Get all files and their statistics
    files_stats = {}
    for fname in sorted(f for f in os.listdir(input_dir) if f.endswith(".json")):
        file_path = os.path.join(input_dir, fname)
        stats = get_file_statistics(file_path)
        files_stats[fname] = stats

    total_tokens = sum(s['tokens'] for s in files_stats.values())
    target_test_tokens = int(total_tokens * test_size)
    target_dev_tokens = int(total_tokens * dev_size)

    # Shuffle files
    all_files = list(files_stats.keys())
    random.shuffle(all_files)

    # Greedy allocation to get close to target sizes
    test_files, dev_files, train_files = [], [], []
    test_tokens = 0
    dev_tokens = 0

    for fname in all_files:
        tokens = files_stats[fname]['tokens']
        if test_tokens < target_test_tokens:
            test_files.append(fname)
            test_tokens += tokens
        elif dev_tokens < target_dev_tokens:
            dev_files.append(fname)
            dev_tokens += tokens
        else:
            train_files.append(fname)

    # Convert files to sentences and collect stats
    def load_files_data(file_list):
        all_sentences = []
        stats = defaultdict(int)
        entity_counts = defaultdict(int)

        for fname in sorted(file_list):
            file_path = os.path.join(input_dir, fname)
            sentences = convert_file_to_bio_sentences(file_path)
            all_sentences.extend(sentences)

            file_stats = files_stats[fname]
            stats['protocols'] += 1
            stats['sentences'] += file_stats['sentences']
            stats['tokens'] += file_stats['tokens']
            stats['entities'] += file_stats['entities']  # already excludes EVENT/UNK

            # Count entities by type from BIO (B- tags only)
            for sentence in sentences:
                for _, bio_tag in sentence:
                    if bio_tag.startswith('B-'):
                        etype = bio_tag[2:]
                        entity_counts[etype] += 1

        return all_sentences, dict(stats), dict(entity_counts)

    train_data, train_stats, train_entities = load_files_data(train_files)
    dev_data, dev_stats, dev_entities = load_files_data(dev_files)
    test_data, test_stats, test_entities = load_files_data(test_files)

    return {
        'train': {'files': train_files, 'data': train_data, 'stats': train_stats, 'entities': train_entities},
        'dev': {'files': dev_files, 'data': dev_data, 'stats': dev_stats, 'entities': dev_entities},
        'test': {'files': test_files, 'data': test_data, 'stats': test_stats, 'entities': test_entities},
    }


def write_bio_file(sentences, output_file: str):
    """Write sentences to BIO format TSV file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for token, bio_tag in sentence:
                f.write(f"{token}\t{bio_tag}\n")
            f.write("\n")


def print_statistics_table(splits_data: dict):
    """Print a nice table with corpus statistics."""
    print("\n" + "=" * 80)
    print("CORPUS STATISTICS")
    print("=" * 80)

    print(f"{'Split':<10} {'Protocols':<12} {'Sentences':<12} {'Tokens':<12} {'Entities':<12}")
    print("-" * 80)

    for split_name in ['train', 'dev', 'test']:
        stats = splits_data[split_name]['stats']
        print(f"{split_name.capitalize():<10} "
              f"{stats.get('protocols', 0):<12} "
              f"{stats.get('sentences', 0):<12} "
              f"{stats.get('tokens', 0):<12} "
              f"{stats.get('entities', 0):<12}")

    print("-" * 80)
    total_protocols = sum(s['stats'].get('protocols', 0) for s in splits_data.values())
    total_sentences = sum(s['stats'].get('sentences', 0) for s in splits_data.values())
    total_tokens = sum(s['stats'].get('tokens', 0) for s in splits_data.values())
    total_entities = sum(s['stats'].get('entities', 0) for s in splits_data.values())

    print(f"{'TOTAL':<10} "
          f"{total_protocols:<12} "
          f"{total_sentences:<12} "
          f"{total_tokens:<12} "
          f"{total_entities:<12}")

    print("\n" + "=" * 80)
    print("PERCENTAGES (by tokens)")
    print("=" * 80)
    for split_name in ['train', 'dev', 'test']:
        tokens = splits_data[split_name]['stats'].get('tokens', 0)
        pct = 100 * tokens / total_tokens if total_tokens else 0.0
        print(f"{split_name.capitalize():<10} {pct:>6.2f}%")

    print("\n" + "=" * 80)
    print("ENTITY COUNTS BY TYPE (EVENT/UNK excluded)")
    print("=" * 80)

    all_types = set()
    for split_data in splits_data.values():
        all_types.update(split_data.get('entities', {}).keys())

    print(f"{'Type':<15} {'Train':<10} {'Dev':<10} {'Test':<10} {'Total':<10}")
    print("-" * 80)

    for etype in sorted(all_types):
        train_count = splits_data['train']['entities'].get(etype, 0)
        dev_count = splits_data['dev']['entities'].get(etype, 0)
        test_count = splits_data['test']['entities'].get(etype, 0)
        total_count = train_count + dev_count + test_count
        print(f"{etype:<15} {train_count:<10} {dev_count:<10} {test_count:<10} {total_count:<10}")

    print("=" * 80)


def save_file_distribution(splits_data: dict, output_file='file_distribution.txt'):
    """Save information about which files are in which split (entities exclude EVENT/UNK)."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FILE DISTRIBUTION ACROSS SPLITS\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nNOTE: Total entities EXCLUDE types: {', '.join(sorted(EXCLUDED_ENTITY_TYPES))}\n\n")

        for split_name in ['train', 'dev', 'test']:
            files = splits_data[split_name]['files']
            stats = splits_data[split_name]['stats']

            f.write(f"{split_name.upper()} SET ({len(files)} files)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total tokens: {stats.get('tokens', 0)}\n")
            f.write(f"Total sentences: {stats.get('sentences', 0)}\n")
            f.write(f"Total entities: {stats.get('entities', 0)}\n\n")

            f.write("Files:\n")
            for fname in sorted(files):
                f.write(f"  - {fname}\n")
            f.write("\n\n")

        f.write("=" * 80 + "\n")

    print(f"File distribution saved to {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Split NER corpus into train/dev/test sets")
    parser.add_argument("--filtered_dir", default="json_ne_gold_a",
                        help="Input directory with filtered JSON files")
    parser.add_argument("--train_file", default="train.tsv",
                        help="Output training set file")
    parser.add_argument("--dev_file", default="dev.tsv",
                        help="Output development set file")
    parser.add_argument("--test_file", default="test.tsv",
                        help="Output test set file")
    parser.add_argument("--test_size", type=float, default=0.10,
                        help="Test set size as fraction (default: 0.10)")
    parser.add_argument("--dev_size", type=float, default=0.09,
                        help="Dev set size as fraction (default: 0.09)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 1.4: Splitting corpus into train/dev/test (EVENT/UNK excluded)")
    print("=" * 80)
    print(f"Random seed: {args.random_seed}")
    print(f"Excluded entity types: {', '.join(sorted(EXCLUDED_ENTITY_TYPES))}")
    print(f"Target split: Train ~{100*(1-args.test_size-args.dev_size):.0f}%, "
          f"Dev ~{100*args.dev_size:.0f}%, Test ~{100*args.test_size:.0f}%")

    splits_data = split_corpus(args.filtered_dir, args.test_size, args.dev_size, args.random_seed)

    print("\nWriting split files...")
    write_bio_file(splits_data['train']['data'], args.train_file)
    print(f"{args.train_file}")
    write_bio_file(splits_data['dev']['data'], args.dev_file)
    print(f"{args.dev_file}")
    write_bio_file(splits_data['test']['data'], args.test_file)
    print(f"{args.test_file}")

    print_statistics_table(splits_data)

    with open('corpus_statistics.txt', 'w', encoding='utf-8') as f:
        old_stdout = sys.stdout
        sys.stdout = f
        print_statistics_table(splits_data)
        sys.stdout = old_stdout
    print(f"Statistics saved to: corpus_statistics.txt")

    save_file_distribution(splits_data, 'file_distribution.txt')

    print("\nTrain/dev/test split complete!")
    print("\nFiles created:")
    print(f"  - {args.train_file}")
    print(f"  - {args.dev_file}")
    print(f"  - {args.test_file}")
    print(f"  - corpus_statistics.txt")
    print(f"  - file_distribution.txt")


if __name__ == "__main__":
    main()

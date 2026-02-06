    #!/usr/bin/env python3
    """split_train_dev_test

    Extracted from the provided Jupyter notebook (BIO.ipynb).
    """

    import os
import json
import random
import string
import sys
import regex as re
from estnltk import Text, Layer
from estnltk.taggers.standard.text_segmentation.compound_token_tagger import CompoundTokenTagger, ALL_1ST_LEVEL_PATTERNS
from estnltk.taggers.standard.text_segmentation.patterns import MACROS
from collections import Counter, defaultdict

    def main():
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--data_dir", default=None)
        p.add_argument("--filtered_dir", default=None)
        p.add_argument("--bio_output_file", default=None)
        p.add_argument("--train_file", default=None)
        p.add_argument("--dev_file", default=None)
        p.add_argument("--test_file", default=None)
        args = p.parse_args()

    """
    Step 1.4: Split corpus into train/dev/test sets
    - Test: 10% of tokens
    - Dev: 8-10% of tokens
    - Train: ~80% of tokens
    Splits by complete protocols (files), not by individual sentences

    This version EXCLUDES entity types: EVENT and UNK (everywhere):
    - They are not converted to BIO tags
    - They are not counted in per-file entity totals
    - They do not appear in "ENTITY COUNTS BY TYPE"
    - File distribution "Total entities" reflects the excluded setting
    """
    # ====== CONFIG ======
    FILTERED_DIR = "json_ne_gold_a"  # Input: filtered JSONs
    TRAIN_FILE = "train.tsv"
    DEV_FILE = "dev.tsv"
    TEST_FILE = "test.tsv"

    if args.filtered_dir is not None: FILTERED_DIR = args.filtered_dir
    if args.train_file is not None: TRAIN_FILE = args.train_file
    if args.dev_file is not None: DEV_FILE = args.dev_file
    if args.test_file is not None: TEST_FILE = args.test_file


    RANDOM_SEED = 42
    TEST_SIZE = 0.10   # 10% for test
    DEV_SIZE = 0.09    # 9% for dev
    # Train will be remaining (~81%)

    # Exclude these entity types everywhere
    EXCLUDED_ENTITY_TYPES = {"EVENT", "UNK"}

    # NOTE: These come from your environment. Keep as-is.
    # - FILTERED_DIR
    # - Text
    # - preprocess_words


    def unify_category(category: str) -> str:
        """Unify entity categories according to mapping."""
        CATEGORY_MAPPING = {
            'LOC_ADDRESS': 'LOC',
            'ORG_GPE': 'ORG',
            'ORG_POL': 'ORG',
        }
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
        Returns dict with counts.
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
                    norm = (norm or w.text).replace("w", "v").replace("W", "V")
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

        print(f" File distribution saved to {output_file}")


    if __name__ == "__main__":
        print("=" * 80)
        print("STEP 1.4: Splitting corpus into train/dev/test (EVENT/UNK excluded)")
        print("=" * 80)
        print(f"Random seed: {RANDOM_SEED}")
        print(f"Excluded entity types: {', '.join(sorted(EXCLUDED_ENTITY_TYPES))}")
        print(f"Target split: Train ~{100*(1-TEST_SIZE-DEV_SIZE):.0f}%, Dev ~{100*DEV_SIZE:.0f}%, Test ~{100*TEST_SIZE:.0f}%")

        splits_data = split_corpus(FILTERED_DIR, TEST_SIZE, DEV_SIZE, RANDOM_SEED)

        print("Writing split files...")
        write_bio_file(splits_data['train']['data'], TRAIN_FILE)
        print(f"   {TRAIN_FILE}")
        write_bio_file(splits_data['dev']['data'], DEV_FILE)
        print(f"   {DEV_FILE}")
        write_bio_file(splits_data['test']['data'], TEST_FILE)
        print(f"   {TEST_FILE}")

        print_statistics_table(splits_data)

        with open('corpus_statistics.txt', 'w', encoding='utf-8') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            print_statistics_table(splits_data)
            sys.stdout = old_stdout
        print(f" Statistics saved to: corpus_statistics.txt")

        save_file_distribution(splits_data, 'file_distribution.txt')

        print("\n Train/dev/test split complete!")
        print("\nFiles created:")
        print(f"  - {TRAIN_FILE}")
        print(f"  - {DEV_FILE}")
        print(f"  - {TEST_FILE}")
        print(f"  - corpus_statistics.txt")
        print(f"  - file_distribution.txt")

if __name__ == "__main__":
    main()

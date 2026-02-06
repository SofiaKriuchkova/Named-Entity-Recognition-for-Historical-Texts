    #!/usr/bin/env python3
    """json_to_bio

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
    Step 1.3: Convert corpus to BIO format
    Unifies entity categories and converts to BIO tagging format
    """
    # ============================================================================
    # CONFIGURATION
    # ============================================================================

    FILTERED_DIR = "json_ne_gold_a"           # Input: filtered JSONs
    BIO_OUTPUT_FILE = "corpus_bio.tsv"        # Output: BIO format corpus

    if args.filtered_dir is not None: FILTERED_DIR = args.filtered_dir
    if args.bio_output_file is not None: BIO_OUTPUT_FILE = args.bio_output_file


    # Category mapping for unification
    CATEGORY_MAPPING = {
        'LOC_ADDRESS': 'LOC',
        'ORG_GPE': 'ORG',
        'ORG_POL': 'ORG',
        # Keep others as-is: PER, LOC, ORG, POSITION, etc.
    }

    # ============================================================================
    # BIO CONVERSION FUNCTIONS
    # ============================================================================

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
            sent_words = []
            for w in txt['words']:
                # Check if word is within this sentence
                if w.start >= sent_span.start and w.end <= sent_span.end:
                    norm = w.annotations[0].get('normalized_form') if w.annotations else None
                    norm = (norm or w.text).replace("w", "v").replace("W", "V")
                    sent_words.append({
                        "text": w.text,
                        "norm": norm,
                        "start": w.start,
                        "end": w.end,
                        "sent_id": sent_id
                    })
            if sent_words:  # Only add non-empty sentences
                words.extend(sent_words)

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
        Returns: list of all sentences for further processing.
        """
        all_sentences = []
        file_sentences = {}  # Track which file each sentence came from

        print("Converting corpus to BIO format...")

        for fname in sorted(f for f in os.listdir(input_dir) if f.endswith(".json")):
            file_path = os.path.join(input_dir, fname)
            sentences = convert_file_to_bio(file_path)
            file_sentences[fname] = sentences
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

        return all_sentences, file_sentences


    # ============================================================================
    # MAIN EXECUTION
    # ============================================================================

    if __name__ == "__main__":
        print("=" * 80)
        print("STEP 1.3: Converting corpus to BIO format")
        print("=" * 80)

        all_sentences, file_sentences = convert_corpus_to_bio(FILTERED_DIR, BIO_OUTPUT_FILE)

        print("\n" + "=" * 80)
        print("✓ BIO conversion complete!")
        print("=" * 80)
        print(f"Output file: {BIO_OUTPUT_FILE}")
        print("\nNext: Run Step 1.4 (train/dev/test split)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""preprocess_filter_json.py - Filter and diagnose NER corpus

NER Preprocessing Pipeline for Tartu City Council Protocols
Handles filtering, tokenization correction, and diagnostics

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
# TOKENIZATION CORRECTION (from words_tokenization.py)
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
    """Pre-processes Text object: adds word segmentation and normalizes w->v."""
    input_text.tag_layer('tokens')
    adapted_cp_tokens_tagger.tag(input_text)
    input_text.tag_layer('words')
    
    # Normalize w -> v
    for word_span in input_text['words']:
        word_text = word_span.text
        if 'w' in word_text.lower():
            word_span.clear_annotations()
            word_norm = word_text.replace('w', 'v').replace('W', 'V')
            word_span.add_annotation(normalized_form=word_norm)
    
    return input_text


# ============================================================================
# STEP 1: FILTER JSON FILES (keep only ne_gold_a or ne_gold_b)
# ============================================================================

def filter_json_to_gold_a(input_dir, output_dir):
    """
    Filter JSON files to keep only ne_gold_a layer (or ne_gold_b if ne_gold_a missing).
    Prioritizes first annotator (ne_gold_a) over second (ne_gold_b).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".json"):
            continue
        
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)
        
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)
        
        layers = data.get("layers", [])
        
        # Prioritize ne_gold_a over ne_gold_b
        has_a = any(layer['name'] == 'ne_gold_a' for layer in layers)
        if has_a:
            filtered_layers = [layer for layer in layers if layer["name"] == "ne_gold_a"]
        else:
            filtered_layers = [layer for layer in layers if layer["name"] == "ne_gold_b"]
        
        data["layers"] = filtered_layers
        
        if not filtered_layers:
            print(f"⚠ No NER layer in: {fname}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Filtered {len([f for f in os.listdir(input_dir) if f.endswith('.json')])} files to {output_dir}")


# ============================================================================
# STEP 2: DIAGNOSTIC - CHECK TOKENIZATION ALIGNMENT
# ============================================================================

def diagnose_tokenization(file_path, max_spans=None, verbose=True, all_spans_log=None, error_log=None):
    """
    Check alignment between NER spans and word tokens.
    Args:
        file_path: Path to JSON file
        max_spans: Maximum spans to check (None = all spans)
        verbose: Print detailed output
        all_spans_log: File handle to write ALL spans to (optional)
        error_log: File handle to write errors to (optional)
    Returns: (total_spans, missing_alignments)
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    
    text = data.get("text", "")
    txt = Text(text)
    txt = preprocess_words(txt)
    
    # Build word token list
    words = []
    for w in txt['words']:
        norm = w.annotations[0].get('normalized_form') if w.annotations else None
        norm = (norm or w.text).replace("w", "v").replace("W", "V")
        words.append({
            "text": w.text,
            "start": w.start,
            "end": w.end,
            "norm": norm
        })
    
    # Find NER layer (ne_gold_a or ne_gold_b)
    ne_layer = None
    for candidate in ("ne_gold_a", "ne_gold_b"):
        ne_layer = next((l for l in data.get("layers", []) if l["name"] == candidate), None)
        if ne_layer:
            break
    
    if not ne_layer:
        if verbose:
            print(f"⚠ No NER layer in: {file_path}")
        return 0, 0
    
    spans = ne_layer.get("spans", [])
    
    def tokens_overlapping(start, end):
        return [w for w in words if w["start"] < end and w["end"] > start]
    
    missing = 0
    
    if verbose:
        print(f"\n{file_path}")
        print("=" * 80)
    
    spans_to_check = spans if max_spans is None else spans[:max_spans]
    
    for span in spans_to_check:
        start, end = span["base_span"]
        etype = span["annotations"][0].get("tag", "?")
        etext = text[start:end]
        covered = tokens_overlapping(start, end)
        
        # Get token strings
        token_strings = [w["norm"] for w in covered] if covered else []
        tokens_str = "|".join(token_strings)
        has_error = "YES" if not covered else "NO"
        
        # Write to all_spans_log if provided
        if all_spans_log:
            all_spans_log.write(f"{os.path.basename(file_path)}\t{repr(etext)}\t{etype}\t({start},{end})\t{tokens_str}\t{has_error}\n")
        
        # Track and log errors separately
        if not covered:
            missing += 1
            
            if error_log:
                error_log.write(f"{os.path.basename(file_path)}\t{repr(etext)}\t{etype}\t({start},{end})\n")
            
            if verbose:
                print(f"⚠ MISSING: {repr(etext)} → {etype} | span=({start},{end})")
        elif verbose:
            print(f"✓ {repr(etext)} → {etype} | tokens={token_strings}")
    
    if verbose:
        checked_count = len(spans_to_check)
        print(f"\n📊 Checked {checked_count}/{len(spans)} spans, missing alignments: {missing}")
        print("-" * 80)
    
    return len(spans), missing


def diagnose_all_files(input_dir, max_spans=None, verbose=True, all_spans_file="all_spans.tsv", error_file="tokenization_errors.tsv"):
    """
    Run diagnostics on all JSON files in directory.
    Args:
        input_dir: Directory with JSON files
        max_spans: Maximum spans per file (None = all spans)
        verbose: Print detailed output
        all_spans_file: File to write ALL spans to (None = don't write)
        error_file: File to write errors only (None = don't write)
    """
    total_spans = 0
    total_missing = 0
    
    all_spans_log = None
    error_log = None
    
    if all_spans_file:
        all_spans_log = open(all_spans_file, "w", encoding="utf-8")
        all_spans_log.write("file\tentity_text\tentity_type\tspan_positions\ttokens\thas_error\n")
    
    if error_file:
        error_log = open(error_file, "w", encoding="utf-8")
        error_log.write("file\tentity_text\tentity_type\tspan_positions\n")
    
    try:
        for fname in sorted(f for f in os.listdir(input_dir) if f.endswith(".json")):
            file_path = os.path.join(input_dir, fname)
            spans, missing = diagnose_tokenization(file_path, max_spans, verbose, all_spans_log, error_log)
            total_spans += spans
            total_missing += missing
        
        print(f"\n{'=' * 80}")
        print(f"TOTAL: {total_spans} spans, {total_missing} missing alignments ({100*total_missing/total_spans:.1f}%)")
        print(f"{'=' * 80}")
        
        if all_spans_file:
            print(f"✓ All spans written to: {all_spans_file}")
        if error_file and total_missing > 0:
            print(f"⚠ Errors written to: {error_file}")
    
    finally:
        if all_spans_log:
            all_spans_log.close()
        if error_log:
            error_log.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Filter and diagnose NER corpus")
    parser.add_argument("--data_dir", default="margendatud_json_2025-06-19",
                        help="Input directory with original annotated JSON files")
    parser.add_argument("--filtered_dir", default="json_ne_gold_a",
                        help="Output directory for filtered JSON files")
    parser.add_argument("--all_spans_file", default="all_spans.tsv",
                        help="Output file for all spans diagnostics")
    parser.add_argument("--error_file", default="tokenization_errors.tsv",
                        help="Output file for tokenization errors")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed diagnostic output")
    args = parser.parse_args()

    # Step 1: Filter JSON files (keep only ne_gold_a)
    print("Step 1: Filtering JSON files...")
    filter_json_to_gold_a(args.data_dir, args.filtered_dir)
    
    # Step 2: Run diagnostics on ALL spans and save to files
    print("\nStep 2: Running tokenization diagnostics on ALL spans...")
    diagnose_all_files(
        args.filtered_dir, 
        max_spans=None,  # Check ALL spans
        verbose=args.verbose,
        all_spans_file=args.all_spans_file,
        error_file=args.error_file
    )
    
    print("\n✓ Done!")
    print(f"  - {args.all_spans_file}: Contains ALL spans with their tokenization")
    print(f"  - {args.error_file}: Contains only problematic spans")


if __name__ == "__main__":
    main()

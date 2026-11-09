import argparse
import csv
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pdfplumber

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("parse_nas")

# Define the 15 class/subject slots in required order
SLOTS = [
    ("Class 3", "Language"),
    ("Class 3", "Maths"),
    ("Class 3", "EVS"),
    ("Class 5", "Language"),
    ("Class 5", "Maths"),
    ("Class 5", "EVS"),
    ("Class 8", "Language"),
    ("Class 8", "Maths"),
    ("Class 8", "Science"),
    ("Class 8", "Social Science"),
    ("Class 10", "Language"),
    ("Class 10", "Maths"),
    ("Class 10", "Science"),
    ("Class 10", "Social Science"),
    ("Class 10", "English"),
]

# Standard page indices (0-based) for Class 3, 5, 8, 10
CLASS_PAGE_MAP = [
    (7, 3),   # Page 8: Class 3 (3 subjects: Language, Maths, EVS -> 3 blocks)
    (12, 3),  # Page 13: Class 5 (3 subjects: Language, Maths, EVS -> 3 blocks)
    (18, 4),  # Page 19: Class 8 (4 subjects: Language, Maths, Science, Social Science -> 4 blocks)
    (24, 5),  # Page 25: Class 10 (5 subjects: Language, Maths, Science, Social Science, English -> 5 blocks)
]

NUMBER_PAREN_REGEX = re.compile(r'(\d+)[\s\u200b]*\([\s\u200b]*(\d+(?:\.\d+)?)[\s\u200b]*\)')


def extract_text_from_page(page) -> str:
    """Extract character stream or text from a pdfplumber page."""
    # Characters in character stream preserve reading order of elements
    chars_text = "".join(c.get("text", "") for c in page.chars)
    if not chars_text.strip():
        chars_text = page.extract_text() or ""
    
    # Replace zero-width spaces with regular spaces
    clean_text = chars_text.replace("\u200b", " ")
    
    # Handle a number split across two lines (e.g. 3 then 8 -> 38)
    clean_text = re.sub(r'(\d+)[\t ]*[\r\n]+[\t ]*(\d+)', r'\1\2', clean_text)
    clean_text = re.sub(r'(\d+)\.[\t ]*[\r\n]+[\t ]*(\d+)', r'\1.\2', clean_text)
    
    return clean_text


def extract_values_from_text(text: str):
    """Find all (number, parenthesised decimal) pairs and collapse duplicates."""
    matches = NUMBER_PAREN_REGEX.findall(text)
    if not matches:
        return []
    
    # The text duplicates every value (e.g. 61 (1.4) 61 (1.4)).
    # If all adjacent pairs match, take every second match to collapse repeats cleanly.
    if len(matches) >= 2 and len(matches) % 2 == 0 and all(matches[i] == matches[i + 1] for i in range(0, len(matches), 2)):
        collapsed = [matches[i] for i in range(0, len(matches), 2)]
    else:
        # Fallback to general consecutive deduplication
        collapsed = []
        for m in matches:
            if not collapsed or collapsed[-1] != m:
                collapsed.append(m)
                
    return collapsed


def parse_pdf_file(pdf_path: Path):
    """
    Parse a single NAS PDF into rows.
    Supports 15 blocks (standard) or 9 blocks (Class 8 & 10 only, with Class 3 & 5 empty).
    Returns (rows, failure_reason).
    """
    state = pdf_path.parent.name
    district = pdf_path.stem
    if district.startswith("NAS21_DRC_"):
        district = district[len("NAS21_DRC_"):]
        
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            values = []
            
            # Strategy 1: Check standard pages (pages 8, 13, 19, 25)
            can_use_standard = all(p_idx < num_pages for p_idx, _ in CLASS_PAGE_MAP)
            if can_use_standard:
                std_values = []
                for p_idx, _ in CLASS_PAGE_MAP:
                    p = pdf.pages[p_idx]
                    page_text = extract_text_from_page(p)
                    p_values = extract_values_from_text(page_text)
                    std_values.extend(p_values)
                    
                if len(std_values) == 45:
                    values = std_values
                    
            # Strategy 2: If standard pages didn't yield 45 values, scan all pages
            if len(values) != 45:
                all_values = []
                for p in pdf.pages:
                    page_text = extract_text_from_page(p)
                    p_values = extract_values_from_text(page_text)
                    all_values.extend(p_values)
                values = all_values
                
    except Exception as e:
        return [], f"Failed to open/parse PDF with pdfplumber: {type(e).__name__}: {str(e)}"

    # Group into blocks of 3: (district, state, national)
    num_blocks = len(values) // 3
    if len(values) % 3 != 0 or num_blocks not in (9, 15):
        return [], f"Yielded {len(values)} values ({num_blocks} blocks, expected 15 or 9)"
    
    rows = []
    if num_blocks == 15:
        blocks = [values[i * 3:(i + 1) * 3] for i in range(15)]
        for (cls, subj), block in zip(SLOTS, blocks):
            dist_score, dist_se = block[0]
            st_score, st_se = block[1]
            nat_score, nat_se = block[2]
            
            row = {
                "state": state,
                "district": district,
                "class": cls,
                "subject": subj,
                "district_score": int(dist_score),
                "district_se": float(dist_se),
                "state_score": int(st_score),
                "state_se": float(st_se),
                "national_score": int(nat_score),
                "national_se": float(nat_se),
            }
            rows.append(row)
    elif num_blocks == 9:
        # A PDF yielding exactly 9 blocks is assigned to Class 8 (4) and Class 10 (5) slots only,
        # with Class 3 (3) and Class 5 (3) rows written as empty values.
        blocks = [values[i * 3:(i + 1) * 3] for i in range(9)]
        block_idx = 0
        for cls, subj in SLOTS:
            if cls in ("Class 3", "Class 5"):
                row = {
                    "state": state,
                    "district": district,
                    "class": cls,
                    "subject": subj,
                    "district_score": "",
                    "district_se": "",
                    "state_score": "",
                    "state_se": "",
                    "national_score": "",
                    "national_se": "",
                }
            else:
                block = blocks[block_idx]
                block_idx += 1
                dist_score, dist_se = block[0]
                st_score, st_se = block[1]
                nat_score, nat_se = block[2]
                
                row = {
                    "state": state,
                    "district": district,
                    "class": cls,
                    "subject": subj,
                    "district_score": int(dist_score),
                    "district_se": float(dist_se),
                    "state_score": int(st_score),
                    "state_se": float(st_se),
                    "national_score": int(nat_score),
                    "national_se": float(nat_se),
                }
            rows.append(row)
        
    return rows, None


def _worker_parse_pdf(pdf_path: Path):
    """Wrapper function for worker processes to catch any unexpected error."""
    try:
        rows, reason = parse_pdf_file(pdf_path)
        return pdf_path, rows, reason
    except Exception as e:
        return pdf_path, [], f"Unhandled exception: {type(e).__name__}: {str(e)}"


def validate_national_scores(rows):
    """
    Validate that national_score is identical across all rows for each class/subject.
    Prints failures and returns boolean validation status.
    Empty values (e.g. from 9-block PDFs where Class 3 & 5 were omitted) are skipped.
    """
    by_slot = defaultdict(dict)
    for r in rows:
        key = (r["class"], r["subject"])
        score = r["national_score"]
        # Skip empty values
        if score == "" or score is None:
            continue
        if score not in by_slot[key]:
            by_slot[key][score] = []
        by_slot[key][score].append(f"{r['state']}/{r['district']}")
        
    failures = []
    for (cls, subj), scores in by_slot.items():
        if len(scores) > 1:
            failures.append(((cls, subj), scores))
            
    if failures:
        print("\n" + "=" * 80)
        print("VALIDATION RESULT: FAILED")
        print("=" * 80)
        print("Inconsistent national_score values detected across rows for the same class/subject:")
        for (cls, subj), scores in failures:
            print(f"\n  Slot: {cls} - {subj}")
            for score, locations in scores.items():
                print(f"    - national_score = {score} ({len(locations)} rows, e.g. {locations[:3]})")
        print("=" * 80 + "\n")
        return False
    else:
        print("\n" + "=" * 80)
        print("VALIDATION RESULT: PASSED")
        print("=" * 80)
        print("All rows have identical national_score for each class/subject slot.")
        print(f"Verified {len(by_slot)} class/subject slots across non-empty rows.")
        for (cls, subj), scores in by_slot.items():
            score_val = list(scores.keys())[0]
            count = len(list(scores.values())[0])
            print(f"  - {cls:<8} {subj:<15}: national_score = {score_val} ({count} rows)")
        print("=" * 80 + "\n")
        return True


def print_table(rows, district_filter: str = None):
    """Print long table in GitHub-style markdown format."""
    filtered_rows = rows
    if district_filter:
        filtered_rows = [r for r in rows if r["district"].strip().lower() == district_filter.strip().lower()]
        
    if not filtered_rows:
        print(f"No rows matching district '{district_filter}' to display.")
        return
        
    headers = [
        "state", "district", "class", "subject",
        "district_score", "district_se",
        "state_score", "state_se",
        "national_score", "national_se"
    ]
    
    widths = {h: len(h) for h in headers}
    for r in filtered_rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(r[h])))
            
    header_line = "| " + " | ".join(f"{h:<{widths[h]}}" for h in headers) + " |"
    separator_line = "|-" + "-|-".join("-" * widths[h] for h in headers) + "-|"
    
    print("\n" + header_line)
    print(separator_line)
    for r in filtered_rows:
        row_line = "| " + " | ".join(f"{str(r[h]):<{widths[h]}}" for h in headers) + " |"
        print(row_line)
    print()


def main():
    parser = argparse.ArgumentParser(description="Parse NAS 2021 District Report Card PDFs into a long CSV.")
    parser.add_argument(
        "target",
        nargs="?",
        default="data/raw/NAS_data21",
        help="Path to a PDF file or directory containing PDFs (default: data/raw/NAS_data21)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/clean/nas_district_scores.csv",
        help="Path to output CSV file (default: data/clean/nas_district_scores.csv)"
    )
    parser.add_argument(
        "--failures", "-f",
        default="data/clean/parse_failures.csv",
        help="Path to failures CSV file (default: data/clean/parse_failures.csv)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: os.cpu_count())"
    )
    parser.add_argument(
        "--district", "-d",
        default=None,
        help="Specific district table to print"
    )
    args = parser.parse_args()
    
    target_path = Path(args.target)
    if not target_path.exists():
        logger.error(f"Target path does not exist: {target_path}")
        return
        
    pdf_files = []
    if target_path.is_file() and target_path.suffix.lower() == ".pdf":
        pdf_files.append(target_path)
    elif target_path.is_dir():
        pdf_files = sorted([p for p in target_path.rglob("*.pdf")])
        
    total_pdfs = len(pdf_files)
    logger.info(f"Found {total_pdfs} PDF file(s) in '{target_path}'.")
    if total_pdfs == 0:
        logger.warning("No PDF files found to process.")
        return

    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    logger.info(f"Processing with {workers} worker process(es)...")

    all_rows = []
    failures = []
    parsed_districts_set = set()

    if workers == 1 or total_pdfs == 1:
        for idx, pdf_file in enumerate(pdf_files, 1):
            pdf_path, rows, reason = _worker_parse_pdf(pdf_file)
            if rows:
                all_rows.extend(rows)
                parsed_districts_set.add((rows[0]["state"], rows[0]["district"]))
            else:
                failures.append({"filename": pdf_file.name, "reason": reason})
            if idx % 50 == 0 or idx == total_pdfs:
                logger.info(f"Progress: {idx}/{total_pdfs} PDFs processed ({len(all_rows)} rows, {len(failures)} failures)...")
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_pdf = {executor.submit(_worker_parse_pdf, p): p for p in pdf_files}
            for future in as_completed(future_to_pdf):
                pdf_file = future_to_pdf[future]
                completed += 1
                try:
                    pdf_path, rows, reason = future.result()
                    if rows:
                        all_rows.extend(rows)
                        parsed_districts_set.add((rows[0]["state"], rows[0]["district"]))
                    else:
                        failures.append({"filename": pdf_file.name, "reason": reason})
                except Exception as exc:
                    failures.append({"filename": pdf_file.name, "reason": f"Worker crashed: {type(exc).__name__}: {str(exc)}"})
                
                if completed % 50 == 0 or completed == total_pdfs:
                    logger.info(f"Progress: {completed}/{total_pdfs} PDFs processed ({len(all_rows)} rows, {len(failures)} failures)...")

    # Sort rows by state, district, class, subject to ensure reproducible ordering
    slot_order = {s: i for i, s in enumerate(SLOTS)}
    all_rows.sort(key=lambda r: (r["state"], r["district"], slot_order.get((r["class"], r["subject"]), 99)))

    # Save output CSV
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "state", "district", "class", "subject",
        "district_score", "district_se",
        "state_score", "state_se",
        "national_score", "national_se"
    ]
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info(f"Saved {len(all_rows)} rows to '{output_file}'.")

    # Save failures CSV
    failures_file = Path(args.failures)
    failures_file.parent.mkdir(parents=True, exist_ok=True)
    with open(failures_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "reason"])
        writer.writeheader()
        writer.writerows(failures)
    logger.info(f"Saved {len(failures)} failures to '{failures_file}'.")

    # Print requested district table if specified
    if args.district:
        print_table(all_rows, district_filter=args.district)

    # Print Summary & Validation Result as requested
    total_rows = len(all_rows)
    num_districts_parsed = len(parsed_districts_set)
    num_failures = len(failures)

    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total rows parsed          : {total_rows}")
    print(f"Number of districts parsed : {num_districts_parsed}")
    print(f"Number of failures         : {num_failures}")
    print("=" * 80)

    if failures:
        print("\nFailures encountered:")
        for idx, fail in enumerate(failures[:10], 1):
            print(f"  {idx}. {fail['filename']}: {fail['reason']}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more (see '{failures_file}')")

    # Validation
    if all_rows:
        validate_national_scores(all_rows)
    else:
        logger.warning("No rows extracted, validation skipped.")


if __name__ == "__main__":
    main()

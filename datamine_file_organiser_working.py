from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ActionResult:
    source_file: str
    source_relative_parent: str | None
    source_relative_file: str | None
    preserve_relative_used: bool
    detected_type: str
    suggested_folder: str
    confidence: float
    destination_file: str | None
    action: str
    status: str
    note: str


REQUIRED_COLUMNS = {
    "file",
    "detected_type",
    "suggested_folder",
    "confidence",
}


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"Classification file is missing required columns: {sorted(missing)}"
        )


def sanitize_folder_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        return "99_Unknown"
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        text = text.replace(ch, "_")
    return text


def sanitize_filename(name: str) -> str:
    text = str(name).strip()
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        text = text.replace(ch, "_")
    return text


def unique_destination_path(dest_path: Path) -> Path:
    if not dest_path.exists():
        return dest_path

    stem = dest_path.stem
    suffix = dest_path.suffix
    parent = dest_path.parent
    i = 2
    while True:
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def get_relative_parts(
    src_file: Path,
    preserve_relative: bool = False,
    source_root: Optional[Path] = None,
) -> tuple[str | None, str | None]:
    if not preserve_relative:
        return None, None

    if source_root is None:
        raise ValueError("source_root is required when preserve_relative=True")

    try:
        rel_file = src_file.relative_to(source_root)
        rel_parent = rel_file.parent
    except Exception:
        rel_file = Path(src_file.name)
        rel_parent = Path()

    rel_parent_text = str(rel_parent) if str(rel_parent) not in ("", ".") else ""
    rel_file_text = str(rel_file)
    return rel_parent_text, rel_file_text


def build_destination_path(
    src_file: Path,
    output_root: Path,
    suggested_folder: str,
    preserve_relative: bool = False,
    source_root: Optional[Path] = None,
) -> tuple[Path, str | None, str | None]:
    safe_folder = sanitize_folder_name(suggested_folder)
    base_dir = output_root / safe_folder

    rel_parent_text, rel_file_text = get_relative_parts(
        src_file=src_file,
        preserve_relative=preserve_relative,
        source_root=source_root,
    )

    if preserve_relative:
        rel_parent = Path(rel_parent_text) if rel_parent_text else Path()
        dest_dir = base_dir / rel_parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = unique_destination_path(dest_dir / sanitize_filename(src_file.name))
        return dest_path, rel_parent_text, rel_file_text

    base_dir.mkdir(parents=True, exist_ok=True)
    dest_path = unique_destination_path(base_dir / sanitize_filename(src_file.name))
    return dest_path, rel_parent_text, rel_file_text


def should_process_row(
    row: pd.Series,
    min_confidence: float,
    include_unknown: bool,
) -> tuple[bool, str]:
    detected_type = str(row.get("detected_type", "")).strip()
    suggested_folder = str(row.get("suggested_folder", "")).strip()

    try:
        confidence = float(row.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    if confidence < min_confidence:
        return False, f"confidence {confidence:.3f} < threshold {min_confidence:.3f}"

    if (detected_type.upper() == "UNKNOWN" or suggested_folder == "99_Unknown") and not include_unknown:
        return False, "unknown classification skipped"

    return True, ""


def perform_action(src: Path, dest: Path, action: str) -> None:
    if action == "copy":
        shutil.copy2(src, dest)
    elif action == "move":
        shutil.move(str(src), str(dest))
    else:
        raise ValueError(f"Unsupported action: {action}")


def organise_from_classification(
    classification_csv: Path,
    output_root: Path,
    action: str = "copy",
    dry_run: bool = True,
    min_confidence: float = 0.0,
    include_unknown: bool = False,
    preserve_relative: bool = False,
    source_root: Optional[Path] = None,
) -> pd.DataFrame:
    df = pd.read_csv(classification_csv)
    ensure_required_columns(df)

    results: list[ActionResult] = []

    for _, row in df.iterrows():
        src_text = str(row.get("file", "")).strip()
        detected_type = str(row.get("detected_type", "")).strip()
        suggested_folder = str(row.get("suggested_folder", "")).strip()

        try:
            confidence = float(row.get("confidence", 0.0))
        except Exception:
            confidence = 0.0

        if not src_text:
            results.append(ActionResult(
                source_file="",
                source_relative_parent=None,
                source_relative_file=None,
                preserve_relative_used=preserve_relative,
                detected_type=detected_type,
                suggested_folder=suggested_folder,
                confidence=confidence,
                destination_file=None,
                action=action,
                status="skipped",
                note="blank source file path",
            ))
            continue

        src = Path(src_text)

        process, reason = should_process_row(
            row=row,
            min_confidence=min_confidence,
            include_unknown=include_unknown,
        )
        if not process:
            results.append(ActionResult(
                source_file=str(src),
                source_relative_parent=None,
                source_relative_file=None,
                preserve_relative_used=preserve_relative,
                detected_type=detected_type,
                suggested_folder=suggested_folder,
                confidence=confidence,
                destination_file=None,
                action=action,
                status="skipped",
                note=reason,
            ))
            continue

        if not src.exists():
            results.append(ActionResult(
                source_file=str(src),
                source_relative_parent=None,
                source_relative_file=None,
                preserve_relative_used=preserve_relative,
                detected_type=detected_type,
                suggested_folder=suggested_folder,
                confidence=confidence,
                destination_file=None,
                action=action,
                status="error",
                note="source file not found",
            ))
            continue

        if not src.is_file():
            results.append(ActionResult(
                source_file=str(src),
                source_relative_parent=None,
                source_relative_file=None,
                preserve_relative_used=preserve_relative,
                detected_type=detected_type,
                suggested_folder=suggested_folder,
                confidence=confidence,
                destination_file=None,
                action=action,
                status="error",
                note="source path is not a file",
            ))
            continue

        try:
            dest, rel_parent_text, rel_file_text = build_destination_path(
                src_file=src,
                output_root=output_root,
                suggested_folder=suggested_folder,
                preserve_relative=preserve_relative,
                source_root=source_root,
            )

            if dry_run:
                results.append(ActionResult(
                    source_file=str(src),
                    source_relative_parent=rel_parent_text,
                    source_relative_file=rel_file_text,
                    preserve_relative_used=preserve_relative,
                    detected_type=detected_type,
                    suggested_folder=suggested_folder,
                    confidence=confidence,
                    destination_file=str(dest),
                    action=action,
                    status="dry_run",
                    note="planned only; no file operation executed",
                ))
            else:
                perform_action(src, dest, action=action)
                results.append(ActionResult(
                    source_file=str(src),
                    source_relative_parent=rel_parent_text,
                    source_relative_file=rel_file_text,
                    preserve_relative_used=preserve_relative,
                    detected_type=detected_type,
                    suggested_folder=suggested_folder,
                    confidence=confidence,
                    destination_file=str(dest),
                    action=action,
                    status="ok",
                    note="",
                ))

        except Exception as exc:
            results.append(ActionResult(
                source_file=str(src),
                source_relative_parent=None,
                source_relative_file=None,
                preserve_relative_used=preserve_relative,
                detected_type=detected_type,
                suggested_folder=suggested_folder,
                confidence=confidence,
                destination_file=None,
                action=action,
                status="error",
                note=str(exc),
            ))

    out_df = pd.DataFrame([r.__dict__ for r in results])
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organise Datamine files from the classifier output CSV by copying or moving them into suggested folders."
    )
    parser.add_argument("classification_csv", type=Path, help="Classifier output CSV")
    parser.add_argument("output_root", type=Path, help="Destination root folder")
    parser.add_argument(
        "--action",
        choices=["copy", "move"],
        default="copy",
        help="Whether to copy or move files. Default: copy",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform file operations. Without this flag, the script runs in dry-run mode.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only process files with confidence >= this threshold",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Also process rows classified as Unknown / 99_Unknown",
    )
    parser.add_argument(
        "--preserve-relative",
        action="store_true",
        help="Preserve the source subfolder structure under each suggested folder",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Root folder used to calculate relative paths when --preserve-relative is set",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional output CSV log path. Defaults to <classification stem>_organised_log.csv beside the classification file.",
    )

    args = parser.parse_args()

    if args.preserve_relative and args.source_root is None:
        parser.error("--source-root is required when --preserve-relative is used")

    if args.min_confidence < 0 or args.min_confidence > 1:
        parser.error("--min-confidence must be between 0 and 1")

    result_df = organise_from_classification(
        classification_csv=args.classification_csv,
        output_root=args.output_root,
        action=args.action,
        dry_run=not args.execute,
        min_confidence=args.min_confidence,
        include_unknown=args.include_unknown,
        preserve_relative=args.preserve_relative,
        source_root=args.source_root,
    )

    log_path = args.log or args.classification_csv.with_name(
        args.classification_csv.stem + "_organised_log.csv"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(log_path, index=False)

    print(f"Organisation log written to: {log_path}")
    print()
    display_cols = [
        "source_file",
        "source_relative_parent",
        "source_relative_file",
        "preserve_relative_used",
        "detected_type",
        "suggested_folder",
        "confidence",
        "destination_file",
        "status",
        "note",
    ]
    print(result_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()

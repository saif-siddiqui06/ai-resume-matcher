"""Utility functions: resume text extraction, cleaning, scoring and SQLite storage.

Everything in this module is deterministic and does not call an LLM, which
keeps it easy to unit-test.
"""

from __future__ import annotations

import io
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from models import MatchScore, MatchStatus, SkillMatch

SUPPORTED_EXTENSIONS = (".pdf", ".docx")
MIN_RESUME_CHARS = 50          # Below this, the text is treated as empty/unreadable.
MAX_INPUT_CHARS = 30_000       # Upper bound sent to the LLM to control cost.


class ResumeParsingError(ValueError):
    """Raised when a resume cannot be read or contains no usable text."""


class DatabaseError(RuntimeError):
    """Raised when a SQLite operation fails."""


# --------------------------------------------------------------------------- #
# Text extraction and cleaning
# --------------------------------------------------------------------------- #


def clean_text(text: str) -> str:
    """Normalise whitespace and strip control characters from extracted text."""
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("•", "-").replace("", "-")  # common bullet glyphs
    text = re.sub(r"[ \t ]+", " ", text)                    # collapse spaces/tabs
    text = re.sub(r" *\n *", "\n", text)                          # trim around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)                        # max one blank line
    return text.strip()


def extract_text_from_pdf(data: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    import pymupdf

    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as exc:  # PyMuPDF raises several error types
        raise ResumeParsingError(f"Could not read the PDF file: {exc}") from exc


def extract_text_from_docx(data: bytes) -> str:
    """Extract text from DOCX bytes, including text inside tables."""
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ResumeParsingError(f"Could not read the DOCX file: {exc}") from exc

    lines = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(lines)


def extract_resume_text(data: bytes, filename: str) -> str:
    """Extract and validate resume text from an uploaded PDF or DOCX file.

    Args:
        data: Raw file bytes.
        filename: Original filename, used to detect the format.

    Returns:
        Cleaned resume text.

    Raises:
        ResumeParsingError: If the format is unsupported or the text is empty.
    """
    name = (filename or "").lower()
    if not name.endswith(SUPPORTED_EXTENSIONS):
        raise ResumeParsingError("Unsupported file format. Please upload a PDF or DOCX resume.")
    if not data:
        raise ResumeParsingError("The uploaded file is empty.")

    raw = extract_text_from_pdf(data) if name.endswith(".pdf") else extract_text_from_docx(data)
    return validate_resume_text(raw)


def validate_resume_text(text: str) -> str:
    """Clean resume text and make sure it contains enough content to analyse."""
    cleaned = clean_text(text)
    if len(cleaned) < MIN_RESUME_CHARS:
        raise ResumeParsingError(
            "The resume appears to be empty or unreadable. "
            "If it is a scanned image, please upload a text-based PDF or DOCX."
        )
    return cleaned[:MAX_INPUT_CHARS]


def validate_job_description(text: str) -> str:
    """Clean the job description and reject empty input."""
    cleaned = clean_text(text or "")
    if not cleaned:
        raise ValueError("The job description is empty. Please paste a job description.")
    return cleaned[:MAX_INPUT_CHARS]


# --------------------------------------------------------------------------- #
# Skill matching helpers and scoring
# --------------------------------------------------------------------------- #


def normalize_skill(skill: str) -> str:
    """Normalise a skill name for comparison (case, spacing, punctuation)."""
    return re.sub(r"\s+", " ", (skill or "").strip().lower().rstrip(".,;:"))


def dedupe_skills(skills: list[str]) -> list[str]:
    """Remove duplicate skills (case-insensitive) while keeping the original order."""
    seen: set[str] = set()
    result: list[str] = []
    for skill in skills:
        key = normalize_skill(skill)
        if key and key not in seen:
            seen.add(key)
            result.append(skill.strip())
    return result


def reconcile_matches(job_skills: list[str], llm_matches: list[SkillMatch]) -> list[SkillMatch]:
    """Align LLM match results with the job's skill list.

    Guarantees exactly one result per job skill:
      * skills the LLM forgot are marked MISSING,
      * duplicates and skills the LLM invented are dropped,
      * an EXACT/PARTIAL match without resume evidence is downgraded to MISSING
        (a match must be grounded in the resume).
    """
    by_name = {}
    for match in llm_matches:
        by_name.setdefault(normalize_skill(match.skill), match)

    reconciled: list[SkillMatch] = []
    for skill in dedupe_skills(job_skills):
        match = by_name.get(normalize_skill(skill))
        if match is None:
            reconciled.append(SkillMatch(skill=skill, status=MatchStatus.MISSING,
                                         note="Not found in the resume."))
            continue
        match = match.model_copy(update={"skill": skill})
        if match.status != MatchStatus.MISSING and not (match.evidence or "").strip():
            match = match.model_copy(update={
                "status": MatchStatus.MISSING,
                "note": "Downgraded: no supporting evidence found in the resume.",
            })
        reconciled.append(match)
    return reconciled


def calculate_match_score(required: list[SkillMatch], preferred: Optional[list[SkillMatch]] = None) -> MatchScore:
    """Compute the transparent, deterministic skill match score.

    Formula:
        (Matched Required + 0.5 x Partial Required) / Total Required x 100

    Each required skill is counted once. Preferred skills are reported
    separately and do not affect the score. If there are no required skills,
    ``score`` is ``None`` and ``message`` explains why.
    """
    preferred = preferred or []

    unique_required: dict[str, SkillMatch] = {}
    for match in required:
        unique_required.setdefault(normalize_skill(match.skill), match)
    statuses = [m.status for m in unique_required.values()]

    exact = statuses.count(MatchStatus.EXACT)
    partial = statuses.count(MatchStatus.PARTIAL)
    total = len(statuses)

    result = MatchScore(
        matched_required=exact,
        partial_required=partial,
        missing_required=total - exact - partial,
        total_required=total,
        matched_preferred=sum(m.status == MatchStatus.EXACT for m in preferred),
        partial_preferred=sum(m.status == MatchStatus.PARTIAL for m in preferred),
        total_preferred=len(preferred),
    )

    if total == 0:
        result.message = (
            "No required skills could be identified in the job description, "
            "so a required-skill match score cannot be calculated."
        )
        return result

    result.score = round((exact + 0.5 * partial) / total * 100, 1)
    result.message = (
        f"({exact} matched + 0.5 x {partial} partial) / {total} required = {result.score}%"
    )
    return result


# --------------------------------------------------------------------------- #
# SQLite persistence
# --------------------------------------------------------------------------- #


def get_db_path() -> str:
    """Return the SQLite database path from the environment (default: analyses.db)."""
    return os.getenv("DATABASE_PATH", "analyses.db")


@contextmanager
def _connect(db_path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection, commit on success and wrap errors in DatabaseError."""
    conn = None
    try:
        conn = sqlite3.connect(db_path or get_db_path())
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Database error: {exc}") from exc
    finally:
        if conn is not None:
            conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    """Create the analyses table if it does not exist."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                job_title TEXT,
                candidate_name TEXT,
                match_score REAL,
                final_report TEXT NOT NULL
            )
            """
        )


def save_analysis(
    final_report: str,
    job_title: Optional[str] = None,
    candidate_name: Optional[str] = None,
    match_score: Optional[float] = None,
    db_path: Optional[str] = None,
) -> str:
    """Save a completed analysis and return its generated ID."""
    analysis_id = uuid.uuid4().hex[:12]
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO analyses (id, created_at, job_title, candidate_name, match_score, final_report) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (analysis_id, created_at, job_title, candidate_name, match_score, final_report),
        )
    return analysis_id


def get_history(limit: int = 50, db_path: Optional[str] = None) -> list[dict[str, Any]]:
    """Return recent analyses (without the full report), newest first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, created_at, job_title, candidate_name, match_score "
            "FROM analyses ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_analysis(analysis_id: str, db_path: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Return a single analysis including its full report, or None if not found."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    return dict(row) if row else None


def delete_analysis(analysis_id: str, db_path: Optional[str] = None) -> bool:
    """Delete an analysis. Returns True if a row was deleted."""
    with _connect(db_path) as conn:
        cursor = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        return cursor.rowcount > 0

"""FastAPI backend: runs the LangGraph workflow and manages analysis history.

Run with:  uvicorn api:app --reload
Docs:      http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

import utils
from workflow import FAILED, run_analysis

load_dotenv()

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create the SQLite table on startup."""
    utils.init_db()
    yield


app = FastAPI(
    title="AI Resume & Job Matcher API",
    description="Analyse a resume against a job description with a LangGraph workflow.",
    version="1.0.0",
    lifespan=lifespan,
)


def _db_call(func, *args, **kwargs) -> Any:
    """Run a database function and convert DatabaseError into HTTP 500."""
    try:
        return func(*args, **kwargs)
    except utils.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/analyze")
def analyze(resume: UploadFile = File(...), job_description: str = Form("")) -> dict:
    """Analyse an uploaded resume (PDF/DOCX) against a pasted job description.

    The uploaded file is processed in memory and never written to disk.
    Defined as a sync endpoint so FastAPI runs the (blocking) workflow in a thread pool.
    """
    data = resume.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Resume file is too large (max 5 MB).")

    try:
        resume_text = utils.extract_resume_text(data, resume.filename or "")
        job_text = utils.validate_job_description(job_description)
    except ValueError as exc:  # ResumeParsingError is a ValueError
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    state = run_analysis(resume_text, job_text)

    if state.get("workflow_status") == FAILED:
        message = state.get("error_message") or "The analysis failed."
        code = 502 if "AI service" in message or "Could not start" in message else 422
        raise HTTPException(status_code=code, detail=message)

    warnings = list(state.get("warnings") or [])
    analysis_id = None
    try:
        analysis_id = utils.save_analysis(
            final_report=state["final_report"],
            job_title=(state.get("job_requirements") or {}).get("job_title"),
            candidate_name=(state.get("parsed_resume") or {}).get("candidate_name"),
            match_score=(state.get("match_score") or {}).get("score"),
        )
    except utils.DatabaseError as exc:
        warnings.append(f"The analysis completed but could not be saved to history: {exc}")

    return {
        "analysis_id": analysis_id,
        "workflow_status": state["workflow_status"],
        "warnings": warnings,
        "parsed_resume": state.get("parsed_resume"),
        "job_requirements": state.get("job_requirements"),
        "skill_matches": state.get("skill_matches"),
        "match_score": state.get("match_score"),
        "gap_analysis": state.get("gap_analysis"),
        "improvement_suggestions": state.get("improvement_suggestions"),
        "final_report": state.get("final_report"),
    }


@app.get("/history")
def history(limit: int = 50) -> list[dict]:
    """List previous analyses, newest first."""
    return _db_call(utils.get_history, limit=limit)


@app.get("/history/{analysis_id}")
def history_item(analysis_id: str) -> dict:
    """Return one previous analysis including its Markdown report."""
    record = _db_call(utils.get_analysis, analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return record


@app.delete("/history/{analysis_id}")
def delete_history_item(analysis_id: str) -> dict:
    """Delete a previous analysis."""
    if not _db_call(utils.delete_analysis, analysis_id):
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return {"deleted": analysis_id}

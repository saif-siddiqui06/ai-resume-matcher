"""Tests for the resume matcher. All LLM calls are mocked; no API key is needed.

All names, emails and companies below are synthetic.
"""

from __future__ import annotations

import io
import os
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utils  # noqa: E402
import workflow  # noqa: E402
from models import (  # noqa: E402
    BulletRewrite,
    GapAnalysis,
    ImprovementSuggestions,
    JobRequirements,
    MatchStatus,
    ParsedResume,
    ReportNarrative,
    SkillMatch,
    SkillMatchResult,
)

# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #

SAMPLE_RESUME = """Alex Rivera
alex.rivera@example.com | Springfield
SUMMARY
Backend developer with 3 years of experience building Python web services.
SKILLS
Python, Flask, PostgreSQL, Git, Docker, REST APIs
EXPERIENCE
Software Engineer, Example Corp (2021 - 2024)
- Built REST APIs in Flask serving internal analytics dashboards
- Containerized services with Docker and maintained CI pipelines
PROJECTS
Expense Tracker - Personal web app built with Flask and SQLite
EDUCATION
B.Sc. Computer Science, Example State University, 2021
"""

SAMPLE_JD = """Senior Python Backend Engineer
Requirements: Python, FastAPI, PostgreSQL, Docker, Kubernetes, 5+ years of backend experience.
Nice to have: AWS, Redis.
Responsibilities: design scalable APIs, mentor junior engineers.
"""


def make_parsed_resume() -> ParsedResume:
    return ParsedResume.model_validate({
        "candidate_name": "Alex Rivera",
        "contact": {"email": "alex.rivera@example.com", "location": "Springfield"},
        "summary": "Backend developer with 3 years of experience building Python web services.",
        "education": [{"degree": "B.Sc. Computer Science", "institution": "Example State University",
                       "year": "2021"}],
        "experience": [{
            "title": "Software Engineer", "company": "Example Corp", "duration": "2021 - 2024",
            "bullets": ["Built REST APIs in Flask serving internal analytics dashboards",
                        "Containerized services with Docker and maintained CI pipelines"],
        }],
        "projects": [{"name": "Expense Tracker", "description": "Personal web app",
                      "technologies": ["Flask", "SQLite"]}],
        "technical_skills": ["Python", "Flask", "PostgreSQL", "Git", "Docker", "REST APIs", "python"],
    })


def make_job_requirements() -> JobRequirements:
    return JobRequirements(
        job_title="Senior Python Backend Engineer",
        seniority_level="Senior",
        required_skills=["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "python"],
        preferred_skills=["AWS", "Redis", "Docker"],
        experience_requirements=["5+ years of backend experience"],
        responsibilities=["Design scalable APIs", "Mentor junior engineers"],
    )


def make_match_result() -> SkillMatchResult:
    return SkillMatchResult(
        required=[
            SkillMatch(skill="Python", status=MatchStatus.EXACT, evidence="Python, Flask, PostgreSQL"),
            SkillMatch(skill="FastAPI", status=MatchStatus.PARTIAL, evidence="Built REST APIs in Flask",
                       transferable_from="Flask"),
            SkillMatch(skill="PostgreSQL", status=MatchStatus.EXACT, evidence="Python, Flask, PostgreSQL"),
            SkillMatch(skill="Docker", status=MatchStatus.EXACT, evidence="Containerized services with Docker"),
            # Kubernetes intentionally omitted -> must be reconciled to MISSING
        ],
        preferred=[
            SkillMatch(skill="AWS", status=MatchStatus.MISSING),
            SkillMatch(skill="Redis", status=MatchStatus.MISSING),
        ],
    )


def make_gaps() -> GapAnalysis:
    return GapAnalysis(
        missing_qualifications=["Kubernetes experience"],
        experience_gaps=["Role asks for 5+ years; resume shows about 3 years"],
        underhighlighted_items=["Expense Tracker project demonstrates end-to-end ownership"],
        specific_improvements=["Mention API design work more prominently in the summary"],
    )


def make_improvements() -> ImprovementSuggestions:
    return ImprovementSuggestions(
        strengths_summary="Solid Python backend experience with Flask, PostgreSQL and Docker.",
        bullet_rewrites=[
            BulletRewrite(original="Built REST APIs in Flask serving internal analytics dashboards",
                          improved="Designed and built Flask REST APIs powering internal analytics dashboards",
                          reason="Emphasises API design, a key responsibility."),
            # Fabricated: this bullet is not in the resume -> must be filtered out
            BulletRewrite(original="Led a team of 10 engineers", improved="Led 10 engineers", reason="x"),
            # Adds a new metric -> must be filtered out
            BulletRewrite(original="Containerized services with Docker and maintained CI pipelines",
                          improved="Containerized 25 services with Docker", reason="x"),
        ],
        projects_to_highlight=["Expense Tracker"],
        skills_to_highlight=["Python", "Docker"],
        supported_keywords=["Python", "PostgreSQL", "Kubernetes", "REST APIs"],
    )


def make_narrative() -> ReportNarrative:
    return ReportNarrative(
        candidate_overview="Alex Rivera is a backend developer with about 3 years of Python experience.",
        job_requirements_summary="A senior backend role focused on Python APIs and container platforms.",
        score_explanation="Most core backend skills are present; Kubernetes is the main gap.",
        recommended_next_steps=["Build a small FastAPI project", "Deploy a demo app to Kubernetes"],
    )


class FakeLLM:
    """Minimal stand-in for a LangChain chat model with structured output.

    ``responses`` maps a schema class to either a single response or a list of
    responses returned in order. A response that is an Exception is raised.
    """

    def __init__(self, responses: dict):
        self.responses = {k: (list(v) if isinstance(v, list) else v) for k, v in responses.items()}
        self.calls: list[str] = []

    def with_structured_output(self, schema, **_kwargs):
        llm = self

        class _Runner:
            def invoke(self, _messages):
                llm.calls.append(schema.__name__)
                response = llm.responses[schema]
                if isinstance(response, list):
                    response = response.pop(0) if len(response) > 1 else response[0]
                if isinstance(response, Exception):
                    raise response
                return response

        return _Runner()


def full_responses(overrides: dict | None = None) -> dict:
    responses = {
        ParsedResume: make_parsed_resume(),
        JobRequirements: make_job_requirements(),
        SkillMatchResult: make_match_result(),
        GapAnalysis: make_gaps(),
        ImprovementSuggestions: make_improvements(),
        ReportNarrative: make_narrative(),
    }
    responses.update(overrides or {})
    return responses


# --------------------------------------------------------------------------- #
# Text extraction and validation
# --------------------------------------------------------------------------- #


def _make_pdf(text: str) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _make_docx(lines: list[str], table_row: list[str] | None = None) -> bytes:
    import docx

    document = docx.Document()
    for line in lines:
        document.add_paragraph(line)
    if table_row:
        table = document.add_table(rows=1, cols=len(table_row))
        for cell, value in zip(table.rows[0].cells, table_row):
            cell.text = value
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_extract_text_from_pdf():
    text = utils.extract_resume_text(_make_pdf(SAMPLE_RESUME), "resume.pdf")
    assert "Alex Rivera" in text
    assert "PostgreSQL" in text


def test_extract_text_from_docx_includes_tables():
    data = _make_docx(SAMPLE_RESUME.splitlines(), table_row=["Kotlin", "Swift"])
    text = utils.extract_resume_text(data, "Resume.DOCX")
    assert "Example Corp" in text
    assert "Kotlin | Swift" in text


def test_unsupported_format_rejected():
    with pytest.raises(utils.ResumeParsingError, match="Unsupported file format"):
        utils.extract_resume_text(b"hello", "resume.txt")


def test_corrupt_pdf_rejected():
    with pytest.raises(utils.ResumeParsingError, match="Could not read the PDF"):
        utils.extract_resume_text(b"not really a pdf", "resume.pdf")


@pytest.mark.parametrize("content", ["", "   \n\n  ", "Too short"])
def test_empty_resume_rejected(content):
    with pytest.raises(utils.ResumeParsingError, match="empty or unreadable"):
        utils.validate_resume_text(content)


def test_empty_file_and_empty_pdf_rejected():
    with pytest.raises(utils.ResumeParsingError, match="empty"):
        utils.extract_resume_text(b"", "resume.pdf")
    with pytest.raises(utils.ResumeParsingError, match="empty or unreadable"):
        utils.extract_resume_text(_make_pdf(" "), "resume.pdf")


def test_empty_job_description_rejected():
    with pytest.raises(ValueError, match="job description is empty"):
        utils.validate_job_description("   ")


def test_clean_text_normalises_whitespace():
    raw = "Name\x00\r\n\r\n\r\n•  Python\t\tDocker   "
    assert utils.clean_text(raw) == "Name\n\n- Python Docker"


# --------------------------------------------------------------------------- #
# Structured output models
# --------------------------------------------------------------------------- #


def test_parsed_resume_validation():
    parsed = make_parsed_resume()
    assert parsed.candidate_name == "Alex Rivera"
    assert parsed.has_matchable_content()
    # Missing information stays empty instead of being invented
    assert parsed.contact.phone is None and parsed.certifications == []
    with pytest.raises(ValidationError):
        ParsedResume.model_validate({"technical_skills": "not-a-list"})
    with pytest.raises(ValidationError):
        ParsedResume.model_validate({"experience": [{"company": "No title"}]})


def test_job_requirements_has_requirements():
    assert make_job_requirements().has_requirements()
    assert not JobRequirements(job_title="Engineer").has_requirements()


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def test_parse_resume_node_success():
    llm = FakeLLM(full_responses())
    update = workflow.parse_resume({"resume_text": SAMPLE_RESUME}, llm)
    assert "workflow_status" not in update
    assert update["parsed_resume"]["candidate_name"] == "Alex Rivera"
    # duplicate "python" removed
    assert update["parsed_resume"]["technical_skills"].count("Python") == 1
    assert "python" not in update["parsed_resume"]["technical_skills"]


def test_parse_resume_node_invalid_structured_output():
    bad = None
    try:
        ParsedResume.model_validate({"technical_skills": 5})
    except ValidationError as exc:
        bad = exc
    llm = FakeLLM(full_responses({ParsedResume: bad}))
    update = workflow.parse_resume({"resume_text": SAMPLE_RESUME}, llm)
    assert update["workflow_status"] == workflow.FAILED
    assert "invalid structured output" in update["error_message"]


def test_parse_resume_node_empty_resume_skips_llm():
    llm = FakeLLM(full_responses())
    update = workflow.parse_resume({"resume_text": "  "}, llm)
    assert update["workflow_status"] == workflow.FAILED
    assert "could not read the resume" in update["error_message"]
    assert llm.calls == []


def test_analyze_job_node_extracts_and_dedupes():
    llm = FakeLLM(full_responses())
    update = workflow.analyze_job({"job_description": SAMPLE_JD, "warnings": []}, llm)
    reqs = update["job_requirements"]
    assert reqs["job_title"] == "Senior Python Backend Engineer"
    assert reqs["required_skills"] == ["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes"]
    # "Docker" is required, so it is removed from preferred
    assert reqs["preferred_skills"] == ["AWS", "Redis"]
    assert "workflow_status" not in update


def test_analyze_job_node_no_requirements_is_validation_error():
    llm = FakeLLM(full_responses({JobRequirements: JobRequirements(job_title="Something")}))
    update = workflow.analyze_job({"job_description": "We are hiring. Apply now!"}, llm)
    assert update["workflow_status"] == workflow.FAILED
    assert "no identifiable requirements" in update["error_message"]


def test_analyze_job_node_empty_description():
    llm = FakeLLM(full_responses())
    update = workflow.analyze_job({"job_description": ""}, llm)
    assert update["workflow_status"] == workflow.FAILED
    assert "empty" in update["error_message"]
    assert llm.calls == []


# --------------------------------------------------------------------------- #
# Matching and scoring
# --------------------------------------------------------------------------- #


def test_reconcile_matches_fills_missing_and_downgrades_ungrounded():
    llm_matches = [
        SkillMatch(skill="python", status=MatchStatus.EXACT, evidence="Python"),
        SkillMatch(skill="Go", status=MatchStatus.EXACT, evidence=""),          # no evidence
        SkillMatch(skill="Rust", status=MatchStatus.EXACT, evidence="Rust"),    # not a job skill
    ]
    result = utils.reconcile_matches(["Python", "Go", "Java", "Python"], llm_matches)
    assert [(m.skill, m.status) for m in result] == [
        ("Python", MatchStatus.EXACT),
        ("Go", MatchStatus.MISSING),
        ("Java", MatchStatus.MISSING),
    ]


def test_score_with_exact_partial_and_missing():
    required = [
        SkillMatch(skill="Python", status=MatchStatus.EXACT, evidence="x"),
        SkillMatch(skill="FastAPI", status=MatchStatus.PARTIAL, evidence="x"),
        SkillMatch(skill="Docker", status=MatchStatus.EXACT, evidence="x"),
        SkillMatch(skill="Kubernetes", status=MatchStatus.MISSING),
    ]
    preferred = [SkillMatch(skill="AWS", status=MatchStatus.MISSING)]
    score = utils.calculate_match_score(required, preferred)
    assert score.score == 62.5  # (2 + 0.5) / 4 * 100
    assert (score.matched_required, score.partial_required, score.missing_required) == (2, 1, 1)
    assert score.total_preferred == 1


def test_score_counts_each_required_skill_once_and_ignores_preferred():
    required = [
        SkillMatch(skill="Python", status=MatchStatus.EXACT, evidence="x"),
        SkillMatch(skill="python ", status=MatchStatus.EXACT, evidence="x"),
        SkillMatch(skill="Go", status=MatchStatus.MISSING),
    ]
    preferred_all_matched = [SkillMatch(skill="AWS", status=MatchStatus.EXACT, evidence="x")]
    with_pref = utils.calculate_match_score(required, preferred_all_matched)
    without_pref = utils.calculate_match_score(required)
    assert with_pref.score == without_pref.score == 50.0
    assert with_pref.total_required == 2


def test_score_with_no_required_skills():
    score = utils.calculate_match_score([], [SkillMatch(skill="AWS", status=MatchStatus.EXACT, evidence="x")])
    assert score.score is None
    assert "No required skills" in score.message


def test_score_all_missing_and_all_matched():
    missing = [SkillMatch(skill=s, status=MatchStatus.MISSING) for s in ("A", "B")]
    matched = [SkillMatch(skill=s, status=MatchStatus.EXACT, evidence="x") for s in ("A", "B")]
    assert utils.calculate_match_score(missing).score == 0.0
    assert utils.calculate_match_score(matched).score == 100.0


def test_match_skills_node_reports_missing_and_partial():
    state = {
        "resume_text": SAMPLE_RESUME,
        "parsed_resume": make_parsed_resume().model_dump(mode="json"),
        "job_requirements": workflow.analyze_job(
            {"job_description": SAMPLE_JD}, FakeLLM(full_responses()))["job_requirements"],
    }
    update = workflow.match_skills(state, FakeLLM(full_responses()))
    statuses = {m["skill"]: m["status"] for m in update["skill_matches"]["required"]}
    assert statuses == {"Python": "exact", "FastAPI": "partial", "PostgreSQL": "exact",
                        "Docker": "exact", "Kubernetes": "missing"}
    assert update["match_score"]["score"] == 70.0  # (3 + 0.5) / 5 * 100


def test_match_skills_node_insufficient_information():
    llm = FakeLLM(full_responses())
    state = {
        "parsed_resume": ParsedResume(candidate_name="Sam Example").model_dump(mode="json"),
        "job_requirements": make_job_requirements().model_dump(mode="json"),
        "warnings": [],
    }
    update = workflow.match_skills(state, llm)
    assert update["workflow_status"] == workflow.INSUFFICIENT
    assert "Insufficient information" in update["warnings"][0]
    assert update["match_score"]["score"] == 0.0
    assert llm.calls == []


def test_filter_grounded_suggestions_removes_fabrications():
    matches = {"required": [{"skill": "Python", "status": "exact"},
                            {"skill": "Kubernetes", "status": "missing"}], "preferred": []}
    result = workflow.filter_grounded_suggestions(make_improvements(), SAMPLE_RESUME, matches)
    assert [rw.original for rw in result.bullet_rewrites] == [
        "Built REST APIs in Flask serving internal analytics dashboards"]
    assert "Kubernetes" not in result.supported_keywords
    assert {"Python", "PostgreSQL", "REST APIs"} <= set(result.supported_keywords)


# --------------------------------------------------------------------------- #
# Full workflow
# --------------------------------------------------------------------------- #


def test_complete_workflow_execution():
    llm = FakeLLM(full_responses())
    state = workflow.run_analysis(SAMPLE_RESUME, SAMPLE_JD, llm=llm)

    assert state["workflow_status"] == workflow.COMPLETED
    assert state["error_message"] is None
    assert state["match_score"]["score"] == 70.0
    assert state["report_attempts"] == 1
    assert workflow.find_missing_sections(state["final_report"]) == []
    assert "**70.0%**" in state["final_report"]
    assert "Kubernetes" in state["final_report"]
    assert llm.calls == ["ParsedResume", "JobRequirements", "SkillMatchResult", "GapAnalysis",
                         "ImprovementSuggestions", "ReportNarrative"]


def test_workflow_stops_when_resume_parsing_fails():
    llm = FakeLLM(full_responses({ParsedResume: RuntimeError("Connection timed out")}))
    state = workflow.run_analysis(SAMPLE_RESUME, SAMPLE_JD, llm=llm)
    assert state["workflow_status"] == workflow.FAILED
    assert "Resume parsing failed because the AI service returned an error" in state["error_message"]
    assert state["final_report"] is None
    assert llm.calls == ["ParsedResume"]


def test_workflow_handles_llm_failure_mid_workflow():
    llm = FakeLLM(full_responses({GapAnalysis: RuntimeError("Rate limit exceeded")}))
    state = workflow.run_analysis(SAMPLE_RESUME, SAMPLE_JD, llm=llm)
    assert state["workflow_status"] == workflow.FAILED
    assert "Gap analysis" in state["error_message"] and "Rate limit" in state["error_message"]
    assert "ImprovementSuggestions" not in llm.calls


def test_workflow_validation_error_for_job_without_requirements():
    llm = FakeLLM(full_responses({JobRequirements: JobRequirements()}))
    state = workflow.run_analysis(SAMPLE_RESUME, "Great company, apply today!", llm=llm)
    assert state["workflow_status"] == workflow.FAILED
    assert "no identifiable requirements" in state["error_message"]
    assert "SkillMatchResult" not in llm.calls


def test_workflow_no_required_skills_returns_message_not_score():
    reqs = JobRequirements(job_title="Data Analyst", preferred_skills=["SQL"],
                           responsibilities=["Build reports"])
    matches = SkillMatchResult(preferred=[SkillMatch(skill="SQL", status=MatchStatus.MISSING)])
    llm = FakeLLM(full_responses({JobRequirements: reqs, SkillMatchResult: matches}))
    state = workflow.run_analysis(SAMPLE_RESUME, SAMPLE_JD, llm=llm)
    assert state["workflow_status"] == workflow.COMPLETED_WITH_WARNINGS
    assert state["match_score"]["score"] is None
    assert "Not available" in state["final_report"]
    assert "no identifiable required skills" in state["final_report"]


def test_workflow_insufficient_information_skips_detailed_steps():
    llm = FakeLLM(full_responses({ParsedResume: ParsedResume(candidate_name="Sam Example")}))
    state = workflow.run_analysis(SAMPLE_RESUME, SAMPLE_JD, llm=llm)
    assert state["workflow_status"] == workflow.COMPLETED_WITH_WARNINGS
    assert "GapAnalysis" not in llm.calls and "ImprovementSuggestions" not in llm.calls
    assert workflow.find_missing_sections(state["final_report"]) == []
    assert "Insufficient information" in state["final_report"]


def test_report_regenerated_once_when_sections_missing():
    llm = FakeLLM(full_responses({ReportNarrative: [ReportNarrative(), make_narrative()]}))
    state = workflow.run_analysis(SAMPLE_RESUME, SAMPLE_JD, llm=llm)
    assert state["report_attempts"] == 2
    assert llm.calls.count("ReportNarrative") == 2
    assert state["workflow_status"] == workflow.COMPLETED
    assert "Alex Rivera is a backend developer" in state["final_report"]


def test_report_always_terminates_with_fallback():
    llm = FakeLLM(full_responses({ReportNarrative: ReportNarrative()}))  # always empty
    state = workflow.run_analysis(SAMPLE_RESUME, SAMPLE_JD, llm=llm)
    assert llm.calls.count("ReportNarrative") == workflow.MAX_REPORT_ATTEMPTS
    assert state["workflow_status"] == workflow.COMPLETED
    assert workflow.find_missing_sections(state["final_report"]) == []
    assert "Build a small project that demonstrates Kubernetes" in state["final_report"]


def test_run_analysis_without_api_key_returns_failed_state(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    state = workflow.run_analysis(SAMPLE_RESUME, SAMPLE_JD)
    assert state["workflow_status"] == workflow.FAILED
    assert "OPENAI_API_KEY" in state["error_message"]


# --------------------------------------------------------------------------- #
# SQLite persistence
# --------------------------------------------------------------------------- #


def test_database_crud(tmp_path):
    db = str(tmp_path / "test.db")
    utils.init_db(db)
    first = utils.save_analysis("# Report A", "Backend Engineer", "Alex Rivera", 70.0, db_path=db)
    second = utils.save_analysis("# Report B", "Data Analyst", None, None, db_path=db)

    history = utils.get_history(db_path=db)
    assert {row["id"] for row in history} == {first, second}
    assert "final_report" not in history[0]

    record = utils.get_analysis(first, db_path=db)
    assert record["final_report"] == "# Report A" and record["match_score"] == 70.0

    assert utils.delete_analysis(first, db_path=db) is True
    assert utils.delete_analysis(first, db_path=db) is False
    assert utils.get_analysis(first, db_path=db) is None


def test_database_errors_are_wrapped(tmp_path):
    bad_path = str(tmp_path / "missing_dir" / "x.db")
    with pytest.raises(utils.DatabaseError):
        utils.init_db(bad_path)


# --------------------------------------------------------------------------- #
# FastAPI backend
# --------------------------------------------------------------------------- #


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """TestClient with a temporary database and a mocked LLM."""
    from fastapi.testclient import TestClient

    import api

    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    responses = full_responses()
    monkeypatch.setattr(api, "run_analysis",
                        lambda resume, jd: workflow.run_analysis(resume, jd, llm=FakeLLM(responses)))
    with TestClient(api.app) as client:
        client.responses = responses  # allow tests to tweak LLM responses
        yield client


def test_api_analyze_and_history_roundtrip(api_client):
    files = {"resume": ("resume.docx", _make_docx(SAMPLE_RESUME.splitlines()),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    response = api_client.post("/analyze", files=files, data={"job_description": SAMPLE_JD})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["match_score"]["score"] == 70.0
    assert body["analysis_id"]

    history = api_client.get("/history").json()
    assert history[0]["id"] == body["analysis_id"]
    assert history[0]["candidate_name"] == "Alex Rivera"

    record = api_client.get(f"/history/{body['analysis_id']}").json()
    assert record["final_report"] == body["final_report"]

    assert api_client.delete(f"/history/{body['analysis_id']}").status_code == 200
    assert api_client.get(f"/history/{body['analysis_id']}").status_code == 404


def test_api_rejects_invalid_inputs(api_client):
    bad_format = api_client.post("/analyze", files={"resume": ("resume.txt", b"hello", "text/plain")},
                                 data={"job_description": SAMPLE_JD})
    assert bad_format.status_code == 400 and "Unsupported file format" in bad_format.json()["detail"]

    pdf = {"resume": ("resume.pdf", _make_pdf(SAMPLE_RESUME), "application/pdf")}
    empty_jd = api_client.post("/analyze", files=pdf, data={"job_description": " "})
    assert empty_jd.status_code == 400 and "job description is empty" in empty_jd.json()["detail"]


def test_api_reports_llm_failure(api_client):
    api_client.responses[ParsedResume] = RuntimeError("Service unavailable")
    pdf = {"resume": ("resume.pdf", _make_pdf(SAMPLE_RESUME), "application/pdf")}
    response = api_client.post("/analyze", files=pdf, data={"job_description": SAMPLE_JD})
    assert response.status_code == 502
    assert "AI service" in response.json()["detail"]
    assert api_client.get("/history").json() == []

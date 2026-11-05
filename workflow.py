"""LangGraph workflow for resume / job-description matching.

Graph overview::

    START -> parse_resume -> analyze_job -> match_skills -> analyze_gaps
          -> generate_improvements -> generate_report -> finalize -> END

Conditional edges after every node stop the workflow on errors, skip the
gap/improvement steps when there is not enough resume content to match, and
regenerate the report once if required sections are missing.

Each node is a plain function ``node(state, llm) -> dict`` so it can be
unit-tested in isolation with a fake LLM.
"""

from __future__ import annotations

import json
import os
import re
from functools import partial
from typing import Any, Optional, TypedDict, TypeVar

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from models import (
    GapAnalysis,
    ImprovementSuggestions,
    JobRequirements,
    MatchScore,
    MatchStatus,
    ParsedResume,
    ReportNarrative,
    SkillMatch,
    SkillMatchResult,
)
from utils import (
    ResumeParsingError,
    calculate_match_score,
    dedupe_skills,
    normalize_skill,
    reconcile_matches,
    validate_job_description,
    validate_resume_text,
)

MAX_REPORT_ATTEMPTS = 2  # first attempt + one regeneration

# Status values
RUNNING = "running"
FAILED = "failed"
INSUFFICIENT = "insufficient_information"
COMPLETED = "completed"
COMPLETED_WITH_WARNINGS = "completed_with_warnings"

REQUIRED_SECTIONS = [
    "1. Candidate Overview",
    "2. Job Requirements Summary",
    "3. Overall Skill Match Score",
    "4. Matched Skills",
    "5. Partially Matched Skills",
    "6. Missing Skills",
    "7. Experience and Qualification Gaps",
    "8. Resume Improvement Suggestions",
    "9. Recommended Next Steps",
]

GROUNDING_RULES = (
    "Use ONLY information that is explicitly present in the provided resume and job description. "
    "Never invent or assume skills, employers, dates, certifications, achievements or numbers. "
    "If something is not present, leave the field empty or null."
)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


class ResumeMatchState(TypedDict, total=False):
    """Shared state passed between workflow nodes.

    Structured values are stored as plain dicts (``model_dump``) so the final
    state is JSON-serialisable for the API.
    """

    resume_text: str
    job_description: str
    parsed_resume: Optional[dict]
    job_requirements: Optional[dict]
    skill_matches: Optional[dict]          # {"required": [...], "preferred": [...]}
    match_score: Optional[dict]            # MatchScore
    gap_analysis: Optional[dict]
    improvement_suggestions: Optional[dict]
    final_report: Optional[str]
    workflow_status: str
    error_message: Optional[str]
    # Extra bookkeeping fields
    report_narrative: Optional[dict]
    report_attempts: int
    warnings: list[str]


# --------------------------------------------------------------------------- #
# LLM helpers
# --------------------------------------------------------------------------- #

T = TypeVar("T", bound=BaseModel)


def get_llm() -> Any:
    """Create the OpenAI chat model from environment variables."""
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
        timeout=120,
        max_retries=2,
    )


def call_structured(llm: Any, schema: type[T], system: str, user: str) -> T:
    """Call the LLM and return a validated instance of ``schema``."""
    structured = llm.with_structured_output(schema, method="function_calling")
    result = structured.invoke([("system", system), ("human", user)])
    if isinstance(result, dict):  # some providers return dicts
        result = schema.model_validate(result)
    if not isinstance(result, schema):
        raise ValueError(f"LLM returned an unexpected result for {schema.__name__}.")
    return result


def _failure(step: str, exc: Exception) -> dict:
    """Build a state update describing a failed step with a user-friendly message."""
    if isinstance(exc, ResumeParsingError):
        kind = "could not read the resume"
    elif isinstance(exc, ValidationError):
        kind = "returned invalid structured output"
    elif isinstance(exc, ValueError):
        kind = "failed validation"
    else:
        kind = "failed because the AI service returned an error"
    return {"workflow_status": FAILED, "error_message": f"{step} {kind}: {exc}"}


def _dump(model: BaseModel) -> dict:
    return model.model_dump(mode="json")


def _as_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Node 1: Resume parser
# --------------------------------------------------------------------------- #


def parse_resume(state: ResumeMatchState, llm: Any) -> dict:
    """Extract structured candidate information from the raw resume text."""
    try:
        resume_text = validate_resume_text(state.get("resume_text", ""))
        parsed = call_structured(
            llm,
            ParsedResume,
            system=(
                "You are a precise resume parser. Extract the candidate's name, contact details, "
                "summary, education, work experience (copy bullet points verbatim), projects, "
                "technical skills, certifications and achievements. " + GROUNDING_RULES
            ),
            user=f"RESUME:\n{resume_text}",
        )
        parsed.technical_skills = dedupe_skills(parsed.technical_skills)
        return {"resume_text": resume_text, "parsed_resume": _dump(parsed)}
    except Exception as exc:
        return _failure("Resume parsing", exc)


# --------------------------------------------------------------------------- #
# Node 2: Job description analyzer
# --------------------------------------------------------------------------- #


def analyze_job(state: ResumeMatchState, llm: Any) -> dict:
    """Extract structured requirements from the job description."""
    try:
        job_description = validate_job_description(state.get("job_description", ""))
        reqs = call_structured(
            llm,
            JobRequirements,
            system=(
                "You analyse job descriptions. Extract the job title, seniority level, REQUIRED "
                "technical skills (must-have), PREFERRED technical skills (nice-to-have/bonus), "
                "experience, education and certification requirements, and key responsibilities. "
                "List each skill as a short name (e.g. 'Python', 'Docker', 'REST APIs'). "
                "If the description does not say whether a skill is required or preferred, treat it "
                "as required. " + GROUNDING_RULES
            ),
            user=f"JOB DESCRIPTION:\n{job_description}",
        )
    except Exception as exc:
        return _failure("Job description analysis", exc)

    reqs.required_skills = dedupe_skills(reqs.required_skills)
    required_keys = {normalize_skill(s) for s in reqs.required_skills}
    reqs.preferred_skills = [s for s in dedupe_skills(reqs.preferred_skills)
                             if normalize_skill(s) not in required_keys]

    if not reqs.has_requirements():
        return {
            "job_description": job_description,
            "job_requirements": _dump(reqs),
            "workflow_status": FAILED,
            "error_message": (
                "Validation error: no identifiable requirements were found in the job description. "
                "Please paste a complete job description that lists skills or responsibilities."
            ),
        }

    update: dict = {"job_description": job_description, "job_requirements": _dump(reqs)}
    if not reqs.required_skills:
        update["warnings"] = state.get("warnings", []) + [
            "The job description lists no identifiable required skills, so no match score was calculated."
        ]
    return update


# --------------------------------------------------------------------------- #
# Node 3: Skill matching engine
# --------------------------------------------------------------------------- #


def match_skills(state: ResumeMatchState, llm: Any) -> dict:
    """Classify each job skill as exact / partial / missing and compute the score.

    The LLM only classifies skills and quotes evidence. The score itself is
    calculated deterministically in :func:`utils.calculate_match_score`.
    """
    resume = ParsedResume.model_validate(state["parsed_resume"])
    reqs = JobRequirements.model_validate(state["job_requirements"])

    # Not enough resume content: skip the LLM, mark everything missing and warn.
    if not resume.has_matchable_content():
        required = reconcile_matches(reqs.required_skills, [])
        preferred = reconcile_matches(reqs.preferred_skills, [])
        score = calculate_match_score(required, preferred)
        return {
            "skill_matches": {"required": [_dump(m) for m in required],
                              "preferred": [_dump(m) for m in preferred]},
            "match_score": _dump(score),
            "workflow_status": INSUFFICIENT,
            "warnings": state.get("warnings", []) + [
                "Insufficient information: no skills, work experience or projects could be extracted "
                "from the resume, so skill matching may be inaccurate and detailed suggestions were skipped."
            ],
        }

    try:
        result = call_structured(
            llm,
            SkillMatchResult,
            system=(
                "You compare a candidate's resume against job skills. For EVERY skill in the "
                "REQUIRED and PREFERRED lists return one entry, using the skill name exactly as given. "
                "status rules: 'exact' = the resume clearly shows this skill (or an obvious synonym, "
                "e.g. 'JS' for 'JavaScript'); 'partial' = the resume shows a closely related or "
                "transferable skill (set transferable_from) or only limited exposure; 'missing' = no "
                "evidence. evidence MUST be a short quote copied from the resume for exact/partial "
                "matches, and null for missing. " + GROUNDING_RULES
            ),
            user=(
                f"REQUIRED SKILLS: {_as_json(reqs.required_skills)}\n"
                f"PREFERRED SKILLS: {_as_json(reqs.preferred_skills)}\n\n"
                f"PARSED RESUME:\n{_as_json(_dump(resume))}\n\n"
                f"RAW RESUME TEXT:\n{state.get('resume_text', '')}"
            ),
        )
    except Exception as exc:
        return _failure("Skill matching", exc)

    required = reconcile_matches(reqs.required_skills, result.required + result.preferred)
    preferred = reconcile_matches(reqs.preferred_skills, result.preferred + result.required)
    score = calculate_match_score(required, preferred)
    return {
        "skill_matches": {"required": [_dump(m) for m in required],
                          "preferred": [_dump(m) for m in preferred]},
        "match_score": _dump(score),
    }


# --------------------------------------------------------------------------- #
# Node 4: Resume gap analyzer
# --------------------------------------------------------------------------- #


def analyze_gaps(state: ResumeMatchState, llm: Any) -> dict:
    """Identify missing qualifications, experience gaps and weak resume sections."""
    try:
        gaps = call_structured(
            llm,
            GapAnalysis,
            system=(
                "You are a career coach performing a resume gap analysis. Identify missing "
                "qualifications, experience gaps, existing projects/achievements that are relevant "
                "but under-highlighted, weak or irrelevant resume sections, and specific improvements. "
                "Do NOT recommend adding skills or experience the candidate does not have; for missing "
                "skills, suggest learning them or building a project instead. " + GROUNDING_RULES
            ),
            user=(
                f"PARSED RESUME:\n{_as_json(state['parsed_resume'])}\n\n"
                f"JOB REQUIREMENTS:\n{_as_json(state['job_requirements'])}\n\n"
                f"SKILL MATCHES:\n{_as_json(state['skill_matches'])}"
            ),
        )
        return {"gap_analysis": _dump(gaps)}
    except Exception as exc:
        return _failure("Gap analysis", exc)


# --------------------------------------------------------------------------- #
# Node 5: Resume improvement generator
# --------------------------------------------------------------------------- #


def _squash(text: str) -> str:
    """Lower-case and keep only letters/digits, for loose text comparison."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def filter_grounded_suggestions(
    suggestions: ImprovementSuggestions, resume_text: str, skill_matches: dict
) -> ImprovementSuggestions:
    """Drop suggestions that are not supported by the resume (anti-fabrication guard).

    * Bullet rewrites must reference a bullet that exists in the resume and
      must not introduce numbers that were not in the original bullet.
    * Keywords must appear in the resume or be a matched/partially matched skill.
    """
    resume_squashed = _squash(resume_text)

    rewrites = []
    for rw in suggestions.bullet_rewrites:
        if _squash(rw.original) not in resume_squashed:
            continue
        new_numbers = set(re.findall(r"\d+", rw.improved)) - set(re.findall(r"\d+", rw.original))
        if new_numbers:
            continue
        rewrites.append(rw)

    matched = {
        normalize_skill(m["skill"])
        for group in ("required", "preferred")
        for m in skill_matches.get(group, [])
        if m["status"] in (MatchStatus.EXACT.value, MatchStatus.PARTIAL.value)
    }
    keywords = [
        k for k in dedupe_skills(suggestions.supported_keywords)
        if normalize_skill(k) in matched or _squash(k) in resume_squashed
    ]
    return suggestions.model_copy(update={"bullet_rewrites": rewrites, "supported_keywords": keywords})


def generate_improvements(state: ResumeMatchState, llm: Any) -> dict:
    """Generate personalised, resume-grounded improvement suggestions."""
    try:
        suggestions = call_structured(
            llm,
            ImprovementSuggestions,
            system=(
                "You help candidates tailor their resume to a job. Provide: a concise summary of the "
                "candidate's strongest matching qualifications; improved wording for EXISTING resume "
                "bullets (copy the original bullet verbatim, do not add new facts, tools or numbers); "
                "existing projects and skills to highlight; job-description keywords that the resume "
                "genuinely supports; and general suggestions. " + GROUNDING_RULES
            ),
            user=(
                f"RAW RESUME TEXT:\n{state.get('resume_text', '')}\n\n"
                f"JOB REQUIREMENTS:\n{_as_json(state['job_requirements'])}\n\n"
                f"SKILL MATCHES:\n{_as_json(state['skill_matches'])}\n\n"
                f"GAP ANALYSIS:\n{_as_json(state.get('gap_analysis'))}"
            ),
        )
    except Exception as exc:
        return _failure("Improvement generation", exc)

    suggestions = filter_grounded_suggestions(
        suggestions, state.get("resume_text", ""), state.get("skill_matches") or {}
    )
    return {"improvement_suggestions": _dump(suggestions)}


# --------------------------------------------------------------------------- #
# Node 6: Final report generator
# --------------------------------------------------------------------------- #


def _bullets(items: list[str], empty: str = "None identified.") -> str:
    return "\n".join(f"- {i}" for i in items) if items else f"_{empty}_"


def _skill_lines(matches: list[dict], empty: str) -> str:
    if not matches:
        return f"_{empty}_"
    lines = []
    for m in matches:
        line = f"- **{m['skill']}**"
        if m.get("transferable_from"):
            line += f" (related: {m['transferable_from']})"
        if m.get("evidence"):
            line += f" — evidence: \"{m['evidence']}\""
        elif m.get("note"):
            line += f" — {m['note']}"
        lines.append(line)
    return "\n".join(lines)


def render_report(state: ResumeMatchState, narrative: ReportNarrative) -> str:
    """Assemble the Markdown report from structured state and the LLM narrative.

    Numbers and skill lists come from deterministic state, never from the LLM.
    Narrative sections are rendered as-is (possibly empty) so that
    :func:`find_missing_sections` can detect an incomplete LLM response.
    """
    resume = ParsedResume.model_validate(state.get("parsed_resume") or {})
    reqs = JobRequirements.model_validate(state.get("job_requirements") or {})
    score = MatchScore.model_validate(state.get("match_score") or {})
    matches = state.get("skill_matches") or {"required": [], "preferred": []}
    gaps = GapAnalysis.model_validate(state.get("gap_analysis") or {})
    imp = ImprovementSuggestions.model_validate(state.get("improvement_suggestions") or {})

    def by_status(group: str, status: MatchStatus) -> list[dict]:
        return [m for m in matches.get(group, []) if m["status"] == status.value]

    title = reqs.job_title or "the target role"
    name = resume.candidate_name or "Candidate"
    score_text = f"**{score.score}%**" if score.score is not None else "**Not available**"

    out: list[str] = [f"# Resume Match Report: {name} → {title}", ""]

    if state.get("warnings"):
        out += ["> **Warnings**", *[f"> - {w}" for w in state["warnings"]], ""]

    out += ["## 1. Candidate Overview", narrative.candidate_overview.strip(), ""]

    out += ["## 2. Job Requirements Summary", narrative.job_requirements_summary.strip()]
    if narrative.job_requirements_summary.strip():
        out += [
            "",
            f"- **Job title:** {reqs.job_title or 'Not specified'}",
            f"- **Seniority:** {reqs.seniority_level or 'Not specified'}",
            f"- **Required skills:** {', '.join(reqs.required_skills) or 'None identified'}",
            f"- **Preferred skills:** {', '.join(reqs.preferred_skills) or 'None identified'}",
        ]
    out.append("")

    out += [
        "## 3. Overall Skill Match Score",
        f"Required-skill match score: {score_text}",
        "",
        f"- Formula: `{score.formula}`",
        f"- Calculation: {score.message}",
        f"- Preferred skills (not scored): {score.matched_preferred} matched, "
        f"{score.partial_preferred} partial, of {score.total_preferred}",
    ]
    if narrative.score_explanation.strip():
        out += ["", narrative.score_explanation.strip()]
    out.append("")

    out += ["## 4. Matched Skills",
            "**Required**", _skill_lines(by_status("required", MatchStatus.EXACT), "No exact matches."), "",
            "**Preferred**", _skill_lines(by_status("preferred", MatchStatus.EXACT), "No exact matches."), ""]
    out += ["## 5. Partially Matched Skills",
            "**Required**", _skill_lines(by_status("required", MatchStatus.PARTIAL), "No partial matches."), "",
            "**Preferred**", _skill_lines(by_status("preferred", MatchStatus.PARTIAL), "No partial matches."), ""]
    out += ["## 6. Missing Skills",
            "**Required**", _skill_lines(by_status("required", MatchStatus.MISSING), "No missing required skills."), "",
            "**Preferred**", _skill_lines(by_status("preferred", MatchStatus.MISSING), "No missing preferred skills."), ""]

    out += ["## 7. Experience and Qualification Gaps"]
    if state.get("gap_analysis") is None:
        out += ["_Gap analysis was skipped because the resume did not contain enough information._"]
    else:
        out += ["**Missing qualifications**", _bullets(gaps.missing_qualifications), "",
                "**Experience gaps**", _bullets(gaps.experience_gaps), "",
                "**Weak or irrelevant sections**", _bullets(gaps.weak_or_irrelevant_sections)]
    out.append("")

    out += ["## 8. Resume Improvement Suggestions"]
    if state.get("improvement_suggestions") is None:
        out += ["_Improvement suggestions were skipped because the resume did not contain enough information._"]
    else:
        if imp.strengths_summary:
            out += [f"**Strongest matching qualifications:** {imp.strengths_summary}", ""]
        out += ["**Suggested improvements**", _bullets(gaps.specific_improvements + imp.general_suggestions), "",
                "**Projects to highlight**", _bullets(imp.projects_to_highlight + gaps.underhighlighted_items), "",
                "**Skills to highlight**", _bullets(imp.skills_to_highlight), "",
                "**Supported keywords to include**",
                ", ".join(imp.supported_keywords) if imp.supported_keywords else "_None identified._", ""]
        out += ["**Bullet point rewrites**"]
        if imp.bullet_rewrites:
            for rw in imp.bullet_rewrites:
                out += [f"- Original: {rw.original}", f"  - Improved: {rw.improved}", f"  - Why: {rw.reason}"]
        else:
            out += ["_No rewrites suggested._"]
    out.append("")

    steps = narrative.recommended_next_steps
    out += ["## 9. Recommended Next Steps", _bullets(steps) if steps else ""]
    return "\n".join(out).strip() + "\n"


def find_missing_sections(report: str) -> list[str]:
    """Return the required report sections that are absent or have no content."""
    missing = []
    for section in REQUIRED_SECTIONS:
        match = re.search(rf"^## {re.escape(section)}\s*$(.*?)(?=^## |\Z)", report or "", re.M | re.S)
        if not match or not match.group(1).strip():
            missing.append(section)
    return missing


def generate_report(state: ResumeMatchState, llm: Any) -> dict:
    """Write the narrative sections with the LLM and render the Markdown report."""
    attempts = state.get("report_attempts", 0) + 1
    context = {
        "parsed_resume": state.get("parsed_resume"),
        "job_requirements": state.get("job_requirements"),
        "match_score": state.get("match_score"),
        "skill_matches": state.get("skill_matches"),
        "gap_analysis": state.get("gap_analysis"),
        "warnings": state.get("warnings", []),
    }
    try:
        narrative = call_structured(
            llm,
            ReportNarrative,
            system=(
                "You write the narrative parts of a resume-to-job match report. Provide ALL fields: "
                "candidate_overview (2-4 sentences), job_requirements_summary (2-3 sentences), "
                "score_explanation (explain what the given score means; do not change or recompute it), "
                "and recommended_next_steps (3-6 concrete, honest actions). " + GROUNDING_RULES
            ),
            user=f"ANALYSIS DATA:\n{_as_json(context)}",
        )
    except Exception as exc:
        return {**_failure("Report generation", exc), "report_attempts": attempts}

    return {
        "report_narrative": _dump(narrative),
        "final_report": render_report(state, narrative),
        "report_attempts": attempts,
    }


def _fallback_narrative(state: ResumeMatchState, narrative: ReportNarrative) -> ReportNarrative:
    """Fill empty narrative fields with deterministic text built from state."""
    resume = ParsedResume.model_validate(state.get("parsed_resume") or {})
    reqs = JobRequirements.model_validate(state.get("job_requirements") or {})
    missing = [m["skill"] for m in (state.get("skill_matches") or {}).get("required", [])
               if m["status"] == MatchStatus.MISSING.value]
    return ReportNarrative(
        candidate_overview=narrative.candidate_overview.strip() or (
            f"{resume.candidate_name or 'The candidate'} lists {len(resume.experience)} role(s), "
            f"{len(resume.projects)} project(s) and {len(resume.technical_skills)} technical skill(s)."
        ),
        job_requirements_summary=narrative.job_requirements_summary.strip() or (
            f"The role ({reqs.job_title or 'title not specified'}) requires "
            f"{len(reqs.required_skills)} skill(s) and prefers {len(reqs.preferred_skills)} more."
        ),
        score_explanation=narrative.score_explanation,
        recommended_next_steps=narrative.recommended_next_steps or (
            [f"Build a small project that demonstrates {s}." for s in missing[:3]]
            + ["Tailor your resume summary to the job title and its key responsibilities."]
        ),
    )


def finalize(state: ResumeMatchState) -> dict:
    """Terminal node: ensure the report is complete and set the final status."""
    report = state.get("final_report") or ""
    if find_missing_sections(report):
        narrative = ReportNarrative.model_validate(state.get("report_narrative") or {})
        report = render_report(state, _fallback_narrative(state, narrative))

    status = COMPLETED_WITH_WARNINGS if state.get("warnings") else COMPLETED
    return {"final_report": report, "workflow_status": status}


# --------------------------------------------------------------------------- #
# Conditional routing
# --------------------------------------------------------------------------- #


def route_on_failure(next_node: str):
    """Build a router that ends the workflow on failure, otherwise continues."""

    def _route(state: ResumeMatchState) -> str:
        return END if state.get("workflow_status") == FAILED else next_node

    return _route


def route_after_matching(state: ResumeMatchState) -> str:
    """Stop on failure, skip gap/improvement steps when information is insufficient."""
    status = state.get("workflow_status")
    if status == FAILED:
        return END
    if status == INSUFFICIENT:
        return "generate_report"
    return "analyze_gaps"


def route_after_report(state: ResumeMatchState) -> str:
    """Regenerate the report once if sections are missing; always terminate afterwards."""
    if state.get("workflow_status") == FAILED:
        return END
    if find_missing_sections(state.get("final_report") or "") and \
            state.get("report_attempts", 0) < MAX_REPORT_ATTEMPTS:
        return "generate_report"
    return "finalize"


# --------------------------------------------------------------------------- #
# Graph construction and entry point
# --------------------------------------------------------------------------- #


def build_graph(llm: Any = None):
    """Build and compile the LangGraph workflow.

    Args:
        llm: A LangChain chat model (or a test double). Defaults to :func:`get_llm`.
    """
    llm = llm or get_llm()
    graph = StateGraph(ResumeMatchState)

    graph.add_node("parse_resume", partial(parse_resume, llm=llm))
    graph.add_node("analyze_job", partial(analyze_job, llm=llm))
    graph.add_node("match_skills", partial(match_skills, llm=llm))
    graph.add_node("analyze_gaps", partial(analyze_gaps, llm=llm))
    graph.add_node("generate_improvements", partial(generate_improvements, llm=llm))
    graph.add_node("generate_report", partial(generate_report, llm=llm))
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "parse_resume")
    graph.add_conditional_edges("parse_resume", route_on_failure("analyze_job"), ["analyze_job", END])
    graph.add_conditional_edges("analyze_job", route_on_failure("match_skills"), ["match_skills", END])
    graph.add_conditional_edges("match_skills", route_after_matching, ["analyze_gaps", "generate_report", END])
    graph.add_conditional_edges("analyze_gaps", route_on_failure("generate_improvements"),
                                ["generate_improvements", END])
    graph.add_conditional_edges("generate_improvements", route_on_failure("generate_report"),
                                ["generate_report", END])
    graph.add_conditional_edges("generate_report", route_after_report, ["generate_report", "finalize", END])
    graph.add_edge("finalize", END)
    return graph.compile()


def initial_state(resume_text: str, job_description: str) -> ResumeMatchState:
    """Create the starting state for a workflow run."""
    return ResumeMatchState(
        resume_text=resume_text,
        job_description=job_description,
        parsed_resume=None,
        job_requirements=None,
        skill_matches=None,
        match_score=None,
        gap_analysis=None,
        improvement_suggestions=None,
        final_report=None,
        workflow_status=RUNNING,
        error_message=None,
        report_narrative=None,
        report_attempts=0,
        warnings=[],
    )


def run_analysis(resume_text: str, job_description: str, llm: Any = None) -> ResumeMatchState:
    """Run the full workflow and return the final state.

    LLM construction errors (e.g. a missing API key) are returned as a failed
    state rather than raised, so callers always get a terminal state.
    """
    try:
        app = build_graph(llm)
    except Exception as exc:
        state = initial_state(resume_text, job_description)
        state.update(workflow_status=FAILED, error_message=f"Could not start the analysis: {exc}")
        return state
    # Recursion limit is a final safety net against unexpected loops.
    return app.invoke(initial_state(resume_text, job_description), config={"recursion_limit": 25})

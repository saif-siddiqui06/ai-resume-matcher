"""Streamlit frontend for the AI Resume & Job Matcher.

Run the backend first (uvicorn api:app), then:  streamlit run app.py
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 300  # seconds; the workflow makes several LLM calls

st.set_page_config(page_title="AI Resume & Job Matcher", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- #
# Backend helpers
# --------------------------------------------------------------------------- #


def api_request(method: str, path: str, **kwargs) -> tuple[Optional[Any], Optional[str]]:
    """Call the backend and return ``(json, error_message)``."""
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.ConnectionError:
        return None, (f"Cannot reach the backend at {BACKEND_URL}. "
                      "Start it with `uvicorn api:app --reload` and try again.")
    except requests.Timeout:
        return None, "The analysis took too long and timed out. Please try again."

    if response.ok:
        return response.json(), None
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return None, f"{detail} (HTTP {response.status_code})"


def report_filename(title: Optional[str], name: Optional[str]) -> str:
    """Build a safe Markdown filename for downloads."""
    base = "_".join(p for p in (name, title) if p) or "resume_match_report"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")[:80] + ".md"


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def render_skill_list(matches: list[dict], empty: str) -> None:
    """Show skills with their resume evidence in a compact list."""
    if not matches:
        st.caption(empty)
        return
    for m in matches:
        extra = f" _(related: {m['transferable_from']})_" if m.get("transferable_from") else ""
        st.markdown(f"**{m['skill']}**{extra}")
        if m.get("evidence"):
            st.caption(f"Evidence: “{m['evidence']}”")
        elif m.get("note"):
            st.caption(m["note"])


def render_result(result: dict) -> None:
    """Render a full analysis result returned by POST /analyze."""
    resume = result.get("parsed_resume") or {}
    reqs = result.get("job_requirements") or {}
    score = result.get("match_score") or {}
    matches = result.get("skill_matches") or {"required": [], "preferred": []}
    gaps = result.get("gap_analysis") or {}
    imp = result.get("improvement_suggestions") or {}

    for warning in result.get("warnings") or []:
        st.warning(warning)

    # Score and counts
    cols = st.columns(4)
    cols[0].metric("Skill match score", f"{score['score']}%" if score.get("score") is not None else "N/A")
    cols[1].metric("Matched (required)", score.get("matched_required", 0))
    cols[2].metric("Partial (required)", score.get("partial_required", 0))
    cols[3].metric("Missing (required)", score.get("missing_required", 0))
    if score.get("score") is not None:
        st.progress(min(int(score["score"]), 100))
    st.caption(f"Formula: {score.get('formula', '')} → {score.get('message', '')}")

    # Overviews
    left, right = st.columns(2)
    with left:
        st.subheader("👤 Candidate overview")
        st.markdown(f"**{resume.get('candidate_name') or 'Name not found'}**")
        contact = resume.get("contact") or {}
        contact_line = " · ".join(v for v in contact.values() if v)
        if contact_line:
            st.caption(contact_line)
        if resume.get("summary"):
            st.write(resume["summary"])
        for job in resume.get("experience", []):
            st.markdown(f"- {job['title']}" + (f" at {job['company']}" if job.get("company") else "")
                        + (f" ({job['duration']})" if job.get("duration") else ""))
        if resume.get("technical_skills"):
            st.markdown("**Skills:** " + ", ".join(resume["technical_skills"]))
    with right:
        st.subheader("💼 Job requirements")
        st.markdown(f"**{reqs.get('job_title') or 'Title not specified'}**"
                    + (f" · {reqs['seniority_level']}" if reqs.get("seniority_level") else ""))
        st.markdown("**Required:** " + (", ".join(reqs.get("required_skills", [])) or "None identified"))
        st.markdown("**Preferred:** " + (", ".join(reqs.get("preferred_skills", [])) or "None identified"))
        for item in reqs.get("experience_requirements", []) + reqs.get("education_requirements", []):
            st.markdown(f"- {item}")

    # Skills
    st.subheader("🧩 Skill matching")
    by = {s: [m for m in matches.get("required", []) if m["status"] == s] for s in ("exact", "partial", "missing")}
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success(f"Matched ({len(by['exact'])})")
        render_skill_list(by["exact"], "No exact matches.")
    with c2:
        st.warning(f"Partially matched ({len(by['partial'])})")
        render_skill_list(by["partial"], "No partial matches.")
    with c3:
        st.error(f"Missing ({len(by['missing'])})")
        render_skill_list(by["missing"], "No missing required skills.")
    if matches.get("preferred"):
        with st.expander(f"Preferred skills (not scored) — {len(matches['preferred'])}"):
            for m in matches["preferred"]:
                st.markdown(f"- **{m['skill']}**: {m['status']}"
                            + (f" — “{m['evidence']}”" if m.get("evidence") else ""))

    # Gaps and suggestions
    st.subheader("🛠️ Resume improvement suggestions")
    if imp.get("strengths_summary"):
        st.info(imp["strengths_summary"])
    if gaps:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Missing qualifications**")
            for item in gaps.get("missing_qualifications", []) or ["None identified."]:
                st.markdown(f"- {item}")
        with g2:
            st.markdown("**Experience gaps**")
            for item in gaps.get("experience_gaps", []) or ["None identified."]:
                st.markdown(f"- {item}")
    suggestions = gaps.get("specific_improvements", []) + imp.get("general_suggestions", [])
    if suggestions:
        st.markdown("**Suggested improvements**")
        for item in suggestions:
            st.markdown(f"- {item}")
    highlights = imp.get("projects_to_highlight", []) + gaps.get("underhighlighted_items", [])
    if highlights:
        st.markdown("**Existing projects and achievements to highlight**")
        for item in highlights:
            st.markdown(f"- {item}")
    if imp.get("bullet_rewrites"):
        st.markdown("**Bullet point rewrites**")
        for rw in imp["bullet_rewrites"]:
            with st.container(border=True):
                st.markdown(f"~~{rw['original']}~~")
                st.markdown(f"➡️ {rw['improved']}")
                st.caption(rw["reason"])
    if imp.get("supported_keywords"):
        st.markdown("**Keywords supported by your resume:** " + ", ".join(imp["supported_keywords"]))

    # Full report
    st.subheader("📄 Full report")
    st.download_button(
        "⬇️ Download report (Markdown)",
        data=result.get("final_report") or "",
        file_name=report_filename(reqs.get("job_title"), resume.get("candidate_name")),
        mime="text/markdown",
        key=f"download_{result.get('analysis_id')}",
    )
    with st.expander("View full Markdown report"):
        st.markdown(result.get("final_report") or "")


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def analysis_page() -> None:
    """Upload a resume, paste a job description and run the analysis."""
    st.header("Resume Analysis")
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    with col2:
        job_description = st.text_area("Paste the job description", height=220)

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if uploaded is None:
            st.error("Please upload a resume file.")
        elif not job_description.strip():
            st.error("Please paste a job description.")
        else:
            with st.status("Analyzing your resume…", expanded=True) as status:
                st.write("Parsing resume → analysing job → matching skills → "
                         "finding gaps → generating suggestions → writing report")
                result, error = api_request(
                    "POST", "/analyze",
                    files={"resume": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    data={"job_description": job_description},
                )
                if error:
                    status.update(label="Analysis failed", state="error")
                    st.session_state.pop("result", None)
                    st.error(error)
                else:
                    status.update(label="Analysis complete", state="complete", expanded=False)
                    st.session_state["result"] = result

    if st.session_state.get("result"):
        st.divider()
        render_result(st.session_state["result"])


def history_page() -> None:
    """Browse, open, download and delete previous analyses."""
    st.header("Analysis History")
    items, error = api_request("GET", "/history")
    if error:
        st.error(error)
        return
    if not items:
        st.info("No analyses yet. Run one from the Resume Analysis page.")
        return

    st.dataframe(
        [{"Date (UTC)": i["created_at"].replace("T", " ")[:16],
          "Candidate": i.get("candidate_name") or "—",
          "Job title": i.get("job_title") or "—",
          "Score": f"{i['match_score']}%" if i.get("match_score") is not None else "N/A"} for i in items],
        use_container_width=True, hide_index=True,
    )

    labels = {i["id"]: f"{i['created_at'][:16].replace('T', ' ')} — {i.get('candidate_name') or 'Candidate'} "
                       f"→ {i.get('job_title') or 'Job'}" for i in items}
    selected = st.selectbox("Open a report", options=list(labels), format_func=labels.get)
    if not selected:
        return

    record, error = api_request("GET", f"/history/{selected}")
    if error:
        st.error(error)
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        st.download_button("⬇️ Download report (Markdown)", data=record["final_report"],
                           file_name=report_filename(record.get("job_title"), record.get("candidate_name")),
                           mime="text/markdown")
    with c2:
        if st.button("🗑️ Delete", use_container_width=True):
            _, error = api_request("DELETE", f"/history/{selected}")
            if error:
                st.error(error)
            else:
                st.success("Deleted.")
                st.rerun()
    with st.container(border=True):
        st.markdown(record["final_report"])


def main() -> None:
    st.title("📄 AI Resume & Job Matcher")
    st.caption("Upload a resume, paste a job description, and get a grounded, transparent match analysis.")
    page = st.sidebar.radio("Navigation", ["Resume Analysis", "Analysis History"])
    st.sidebar.caption(f"Backend: {BACKEND_URL}")
    if page == "Resume Analysis":
        analysis_page()
    else:
        history_page()


main()

"""Pydantic models used for LLM structured outputs and data exchange.

Every LLM call in the workflow returns one of these models (via LangChain's
``with_structured_output``), so outputs are validated before they are used.
Fields default to empty values so that missing information stays *missing*
instead of being invented.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #


class ContactInfo(BaseModel):
    """Contact details found in the resume. Unknown fields stay ``None``."""

    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None


class Education(BaseModel):
    """A single education entry."""

    degree: str = Field(description="Degree or qualification, e.g. 'B.Sc. Computer Science'")
    institution: Optional[str] = None
    year: Optional[str] = Field(default=None, description="Graduation year or date range")
    details: Optional[str] = None


class WorkExperience(BaseModel):
    """A single job held by the candidate."""

    title: str
    company: Optional[str] = None
    duration: Optional[str] = Field(default=None, description="Date range exactly as written")
    bullets: list[str] = Field(default_factory=list, description="Bullet points copied from the resume")


class Project(BaseModel):
    """A project listed on the resume."""

    name: str
    description: Optional[str] = None
    technologies: list[str] = Field(default_factory=list)


class ParsedResume(BaseModel):
    """Structured view of a resume. Only information present in the text is allowed."""

    candidate_name: Optional[str] = Field(default=None, description="Full name, or null if absent")
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: Optional[str] = Field(default=None, description="Professional summary if present")
    education: list[Education] = Field(default_factory=list)
    experience: list[WorkExperience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    technical_skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)

    def has_matchable_content(self) -> bool:
        """Return True if there is anything (skills, jobs, projects) to match against."""
        return bool(self.technical_skills or self.experience or self.projects)


# --------------------------------------------------------------------------- #
# Job description
# --------------------------------------------------------------------------- #


class JobRequirements(BaseModel):
    """Structured requirements extracted from a job description."""

    job_title: Optional[str] = None
    seniority_level: Optional[str] = Field(
        default=None, description="e.g. Intern, Junior, Mid, Senior, Lead, or null if unclear"
    )
    required_skills: list[str] = Field(
        default_factory=list, description="Technical skills that are explicitly required / must-have"
    )
    preferred_skills: list[str] = Field(
        default_factory=list, description="Nice-to-have / preferred / bonus technical skills"
    )
    experience_requirements: list[str] = Field(default_factory=list)
    education_requirements: list[str] = Field(default_factory=list)
    certification_requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)

    def has_requirements(self) -> bool:
        """Return True if anything identifiable was extracted from the job description."""
        return bool(
            self.required_skills
            or self.preferred_skills
            or self.experience_requirements
            or self.education_requirements
            or self.certification_requirements
            or self.responsibilities
        )


# --------------------------------------------------------------------------- #
# Skill matching
# --------------------------------------------------------------------------- #


class MatchStatus(str, Enum):
    """How well a candidate covers a single job skill."""

    EXACT = "exact"
    PARTIAL = "partial"
    MISSING = "missing"


class SkillMatch(BaseModel):
    """Match result for one skill from the job description."""

    skill: str = Field(description="The skill exactly as listed in the job requirements")
    status: MatchStatus
    evidence: Optional[str] = Field(
        default=None,
        description="Short quote or reference from the resume proving the match; null if missing",
    )
    transferable_from: Optional[str] = Field(
        default=None, description="Related skill the candidate has, for partial matches"
    )
    note: Optional[str] = None


class SkillMatchResult(BaseModel):
    """LLM output for the matching step: one entry per job skill."""

    required: list[SkillMatch] = Field(default_factory=list)
    preferred: list[SkillMatch] = Field(default_factory=list)


class MatchScore(BaseModel):
    """Deterministic score computed in Python (never by the LLM)."""

    score: Optional[float] = Field(default=None, description="0-100, or None if not computable")
    matched_required: int = 0
    partial_required: int = 0
    missing_required: int = 0
    total_required: int = 0
    matched_preferred: int = 0
    partial_preferred: int = 0
    total_preferred: int = 0
    formula: str = "(Matched Required + 0.5 x Partial Required) / Total Required x 100"
    message: str = ""


# --------------------------------------------------------------------------- #
# Gap analysis and improvements
# --------------------------------------------------------------------------- #


class GapAnalysis(BaseModel):
    """Gaps between the resume and the job, grounded in both documents."""

    missing_qualifications: list[str] = Field(default_factory=list)
    experience_gaps: list[str] = Field(default_factory=list)
    underhighlighted_items: list[str] = Field(
        default_factory=list,
        description="Existing projects/achievements that are relevant but not emphasised enough",
    )
    weak_or_irrelevant_sections: list[str] = Field(default_factory=list)
    specific_improvements: list[str] = Field(default_factory=list)


class BulletRewrite(BaseModel):
    """A suggested rewording of an existing resume bullet."""

    original: str = Field(description="The bullet exactly as it appears in the resume")
    improved: str = Field(description="Reworded bullet that adds no new facts or numbers")
    reason: str


class ImprovementSuggestions(BaseModel):
    """Personalised, resume-grounded suggestions."""

    strengths_summary: str = Field(default="", description="Concise summary of strongest matching qualifications")
    bullet_rewrites: list[BulletRewrite] = Field(default_factory=list)
    projects_to_highlight: list[str] = Field(default_factory=list)
    skills_to_highlight: list[str] = Field(default_factory=list)
    supported_keywords: list[str] = Field(
        default_factory=list, description="JD keywords that the resume genuinely supports"
    )
    general_suggestions: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Final report narrative
# --------------------------------------------------------------------------- #


class ReportNarrative(BaseModel):
    """Prose sections of the final report written by the LLM.

    Numbers, skill lists and gaps are filled in by Python from earlier steps,
    so the LLM cannot alter the score or invent matches here.
    """

    candidate_overview: str = ""
    job_requirements_summary: str = ""
    score_explanation: str = ""
    recommended_next_steps: list[str] = Field(default_factory=list)

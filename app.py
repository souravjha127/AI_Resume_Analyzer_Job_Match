import re

import streamlit as st
from pypdf import PdfReader

from rag import (
    create_vector_database,
    search_resume,
    extract_sections,
    detect_section
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer & Job Match Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# THEME-FRIENDLY UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background-color: var(--background-color);
    }

    /* Main content width and spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ======================================================
       HEADINGS
       ====================================================== */

    h1, h2, h3, h4 {
        color: var(--text-color) !important;
    }

    p, li, span, label {
        color: var(--text-color);
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
        border-right: 1px solid var(--border-color);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: var(--text-color) !important;
    }


    /* ======================================================
       METRIC CARDS
       ====================================================== */

    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text-color) !important;
    }


    /* ======================================================
       CONTAINERS / CARDS
       ====================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 12px;
    }


    /* ======================================================
       TEXT INPUTS
       ====================================================== */

    textarea,
    input {
        color: var(--text-color) !important;
        background-color: var(--secondary-background-color) !important;
        border: 1px solid var(--border-color) !important;
    }

    textarea::placeholder,
    input::placeholder {
        color: var(--text-color) !important;
        opacity: 0.65;
    }


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    section[data-testid="stFileUploaderDropzone"] {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 12px;
    }

    section[data-testid="stFileUploaderDropzone"] * {
        color: var(--text-color) !important;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    button[kind="secondary"] {
        border: 1px solid var(--border-color);
    }

    button[kind="secondary"] p,
    button[kind="primary"] p {
        color: inherit !important;
    }


    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {
        color: var(--text-color) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        font-weight: 700;
    }


    /* ======================================================
       EXPANDERS
       ====================================================== */

    details {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 10px;
    }

    details summary {
        color: var(--text-color) !important;
    }


    /* ======================================================
       INFO / SUCCESS / WARNING / ERROR BOXES
       ====================================================== */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span {
        color: var(--text-color) !important;
    }


    /* ======================================================
       PROGRESS BAR
       ====================================================== */

    div[data-testid="stProgress"] {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }


    /* ======================================================
       CODE / PRE TEXT
       ====================================================== */

    pre,
    code {
        color: var(--text-color) !important;
    }


    /* ======================================================
       LINKS
       ====================================================== */

    a {
        color: var(--primary-color) !important;
    }


    /* ======================================================
       DIVIDERS
       ====================================================== */

    hr {
        border-color: var(--border-color);
    }


    /* ======================================================
       PROJECT CARD TEXT
       ====================================================== */

    .project-description {
        color: var(--text-color);
        line-height: 1.6;
    }

    .project-tech {
        color: var(--text-color);
        opacity: 0.8;
        font-size: 0.9rem;
    }


    /* ======================================================
       SMALL TEXT
       ====================================================== */

    .muted-text {
        color: var(--text-color);
        opacity: 0.75;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "qa_answer" not in st.session_state:
    st.session_state.qa_answer = None

if "qa_section" not in st.session_state:
    st.session_state.qa_section = None

if "qa_method" not in st.session_state:
    st.session_state.qa_method = None


# ============================================================
# KNOWN TECHNICAL SKILLS
# ============================================================

KNOWN_SKILLS = [

    "python",
    "java",
    "c++",
    "javascript",
    "typescript",
    "r",

    "sql",
    "mysql",
    "postgresql",
    "postgres",
    "mongodb",

    "excel",
    "advanced excel",
    "power bi",
    "tableau",
    "power query",
    "dax",
    "data analysis",
    "data visualization",
    "business analysis",
    "business intelligence",
    "dashboard",
    "reporting",
    "kpi",
    "data research",
    "statistics",

    "machine learning",
    "deep learning",
    "data science",
    "artificial intelligence",
    "ai",
    "nlp",
    "natural language processing",
    "computer vision",
    "tensorflow",
    "keras",
    "pytorch",
    "scikit-learn",
    "sklearn",
    "opencv",
    "k-means",
    "tf-idf",

    "generative ai",
    "llm",
    "large language model",
    "langchain",
    "rag",
    "retrieval augmented generation",
    "prompt engineering",

    "aws",
    "azure",
    "gcp",
    "data engineering",
    "etl",
    "data warehouse",
    "databricks",

    "git",
    "github",
    "streamlit",
    "jira"
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9+#.\- ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# FIND SKILLS
# ============================================================

def find_skills(text):

    text_lower = text.lower()

    found = []

    for skill in KNOWN_SKILLS:

        if skill.lower() in text_lower:

            found.append(skill)

    return found


# ============================================================
# EXTRACT JOB KEYWORDS
# ============================================================

def extract_keywords(text):

    normalized = normalize_text(text)

    words = normalized.split()

    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "have",
        "has",
        "will",
        "are",
        "you",
        "your",
        "our",
        "their",
        "they",
        "about",
        "into",
        "using",
        "use",
        "work",
        "working",
        "role",
        "job",
        "candidate",
        "looking",
        "required",
        "requirements",
        "responsibilities",
        "experience",
        "years",
        "year",
        "good",
        "strong",
        "skills",
        "skill",
        "ability",
        "knowledge",
        "team",
        "teams",
        "including",
        "such",
        "other",
        "must",
        "should",
        "can",
        "who",
        "all",
        "not",
        "to",
        "of",
        "in",
        "on",
        "a",
        "an",
        "is",
        "be",
        "as",
        "or"
    }

    keywords = []

    for word in words:

        if len(word) < 2:
            continue

        if word in stop_words:
            continue

        if word not in keywords:
            keywords.append(word)

    return keywords


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:

            pages.append(
                f"--- PAGE {page_number + 1} ---\n"
                f"{text}"
            )

    return "\n\n".join(pages).strip()


# ============================================================
# EXPERIENCE FALLBACK
# ============================================================

def extract_experience_fallback(resume_text):

    if not resume_text:
        return None

    text = resume_text.replace("\r", "\n")

    text_lower = text.lower()

    company_keywords = [
        "jus jumpin",
        "jus jumpin private limited"
    ]

    role_keywords = [
        "data analyst",
        "business analyst",
        "data scientist",
        "machine learning engineer",
        "software engineer",
        "software developer",
        "analyst",
        "intern"
    ]

    positions = []

    for keyword in company_keywords + role_keywords:

        start = 0

        while True:

            position = text_lower.find(
                keyword,
                start
            )

            if position == -1:
                break

            positions.append(position)

            start = position + len(keyword)

    if not positions:
        return None

    blocks = []

    for position in sorted(set(positions)):

        start = max(
            0,
            position - 300
        )

        end = min(
            len(text),
            position + 900
        )

        block = text[start:end].strip()

        if block:
            blocks.append(block)

    unique_blocks = []

    seen = set()

    for block in blocks:

        normalized = re.sub(
            r"\s+",
            " ",
            block.lower()
        ).strip()

        if normalized not in seen:

            seen.add(normalized)
            unique_blocks.append(block)

    cleaned_blocks = []

    for block in unique_blocks:

        block = re.sub(
            r"--- PAGE \d+ ---",
            "",
            block,
            flags=re.IGNORECASE
        )

        block = re.sub(
            r"\n{3,}",
            "\n\n",
            block
        )

        block = block.strip()

        if len(block.split()) >= 5:
            cleaned_blocks.append(block)

    if not cleaned_blocks:
        return None

    final_blocks = []

    for block in cleaned_blocks:

        normalized_block = re.sub(
            r"\s+",
            " ",
            block.lower()
        )

        already_contained = False

        for existing in final_blocks:

            normalized_existing = re.sub(
                r"\s+",
                " ",
                existing.lower()
            )

            if (
                normalized_block in normalized_existing
                or normalized_existing in normalized_block
            ):

                already_contained = True
                break

        if not already_contained:
            final_blocks.append(block)

    answer = "\n\n".join(final_blocks)

    if len(answer.split()) < 5:
        return None

    return answer


# ============================================================
# RESUME + JOB ANALYSIS
# ============================================================

def analyze_resume(resume_text, job_description):

    resume_lower = normalize_text(resume_text)

    job_lower = normalize_text(job_description)

    resume_skills = find_skills(resume_lower)

    job_skills = find_skills(job_lower)

    matching_skills = []

    missing_skills = []

    for skill in job_skills:

        if skill.lower() in resume_lower:

            matching_skills.append(skill)

        else:

            missing_skills.append(skill)

    job_keywords = extract_keywords(job_description)

    resume_words = set(resume_lower.split())

    matching_keywords = []

    missing_keywords = []

    for keyword in job_keywords:

        if keyword in resume_words:

            matching_keywords.append(keyword)

        else:

            missing_keywords.append(keyword)

    if job_skills:

        skill_score = (
            len(matching_skills)
            /
            len(job_skills)
        ) * 100

    else:

        skill_score = 0

    if job_keywords:

        keyword_score = (
            len(matching_keywords)
            /
            len(job_keywords)
        ) * 100

    else:

        keyword_score = 0

    if job_skills and job_keywords:

        score = (
            skill_score * 0.7
            +
            keyword_score * 0.3
        )

    elif job_skills:

        score = skill_score

    else:

        score = keyword_score

    score = min(
        100,
        round(score)
    )

    return {
        "score": score,
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "matching_keywords": matching_keywords,
        "missing_keywords": missing_keywords
    }


# ============================================================
# RESUME Q&A
# ============================================================

def get_resume_answer(
    resume_text,
    vector_db,
    question
):

    detected_section = detect_section(question)

    # ========================================================
    # EXPERIENCE
    # ========================================================

    if detected_section == "EXPERIENCE":

        sections = extract_sections(resume_text)

        experience_text = sections.get("EXPERIENCE")

        if experience_text:

            experience_text = re.sub(
                r"--- PAGE \d+ ---",
                "",
                experience_text,
                flags=re.IGNORECASE
            )

            experience_text = re.sub(
                r"\n{3,}",
                "\n\n",
                experience_text
            ).strip()

            if len(experience_text.split()) >= 5:

                return (
                    "EXPERIENCE\n\n"
                    + experience_text,
                    "EXPERIENCE",
                    "Section-aware RAG"
                )

        fallback_answer = extract_experience_fallback(
            resume_text
        )

        if fallback_answer:

            return (
                fallback_answer,
                "EXPERIENCE",
                "Section-aware RAG + PDF fallback"
            )

        return (
            "No detailed experience information "
            "could be extracted from the resume.",
            "EXPERIENCE",
            "Section-aware RAG"
        )

    # ========================================================
    # ALL OTHER QUESTIONS
    # ========================================================

    results = search_resume(
        vector_db,
        question,
        k=3
    )

    if not results:

        return (
            None,
            detected_section,
            "Section-aware RAG"
        )

    # ========================================================
    # FILTER BY SECTION
    # ========================================================

    if detected_section:

        section_results = []

        for result in results:

            content = result.page_content.strip()

            if content.upper().startswith(
                detected_section
            ):

                section_results.append(content)

        if section_results:
            results = section_results

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_results = []

    seen = set()

    for result in results:

        if hasattr(result, "page_content"):

            content = result.page_content.strip()

        else:

            content = str(result).strip()

        normalized = re.sub(
            r"\s+",
            " ",
            content.lower()
        )

        if normalized not in seen:

            seen.add(normalized)
            unique_results.append(content)

    if not unique_results:

        return (
            None,
            detected_section,
            "Section-aware RAG"
        )

    answer = "\n\n".join(unique_results)

    return (
        answer,
        detected_section,
        "Section-aware RAG"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 AI Resume Agent")

    st.write(
        "Your local resume intelligence assistant."
    )

    st.divider()

    st.subheader("🧠 Architecture")

    st.write("📄 PDF Text Extraction")
    st.write("🧩 Resume Section Detection")
    st.write("🔤 HuggingFace Embeddings")
    st.write("🗄️ Chroma Vector Database")
    st.write("🔎 Section-Aware RAG")
    st.write("📊 Local Job Matching")

    st.divider()

    st.subheader("⚡ Features")

    st.write("✓ Resume analysis")
    st.write("✓ Job matching")
    st.write("✓ Skill detection")
    st.write("✓ Keyword analysis")
    st.write("✓ Resume Q&A")

    st.divider()

    st.caption(
        "No OpenAI/OpenRouter API required."
    )

    st.caption(
        "Resume processing happens locally."
    )


# ============================================================
# HERO SECTION
# ============================================================

st.title(
    "🤖 AI Resume Analyzer & Job Match Assistant"
)

st.markdown(
    "### Turn your resume into an intelligent knowledge base."
)

st.write(
    "Upload your resume, analyze it against job descriptions, "
    "and ask natural-language questions about your skills, "
    "experience, projects and education."
)

st.divider()


# ============================================================
# RESUME UPLOAD
# ============================================================

st.header("📄 Resume Intelligence")

upload_col1, upload_col2 = st.columns([2, 1])

with upload_col1:

    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf"],
        help="Upload a text-based PDF resume."
    )

with upload_col2:

    st.info(
        "💡 Tip\n\n"
        "Use your latest resume for the most accurate analysis."
    )


# ============================================================
# PROCESS RESUME
# ============================================================

if uploaded_file:

    if (
        st.session_state.file_name
        != uploaded_file.name
    ):

        st.session_state.resume_text = None
        st.session_state.vector_db = None
        st.session_state.analysis = None
        st.session_state.qa_answer = None
        st.session_state.qa_section = None
        st.session_state.qa_method = None

        try:

            with st.spinner(
                "📄 Reading your resume..."
            ):

                resume_text = extract_pdf_text(
                    uploaded_file
                )

            if not resume_text:

                st.error(
                    "No readable text was found in the PDF."
                )

                st.stop()

            st.session_state.resume_text = resume_text

            st.session_state.file_name = (
                uploaded_file.name
            )

        except Exception as e:

            st.error(
                f"Error reading resume: {e}"
            )

            st.stop()

    if st.session_state.vector_db is None:

        try:

            with st.spinner(
                "🧠 Building resume knowledge base..."
            ):

                st.session_state.vector_db = (
                    create_vector_database(
                        st.session_state.resume_text
                    )
                )

        except Exception as e:

            st.error(
                f"Error creating knowledge base: {e}"
            )

            st.stop()


# ============================================================
# RESUME DASHBOARD
# ============================================================

if st.session_state.resume_text:

    resume_text = st.session_state.resume_text

    sections = extract_sections(resume_text)

    resume_skills = find_skills(resume_text)

    st.success(
        f"✅ Resume ready: "
        f"{st.session_state.file_name}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📄 Characters",
            f"{len(resume_text):,}"
        )

    with col2:

        st.metric(
            "📑 Sections",
            len(sections)
        )

    with col3:

        st.metric(
            "🛠️ Skills",
            len(resume_skills)
        )

    with col4:

        st.metric(
            "🧠 RAG Status",
            "Ready"
        )

    st.divider()

    overview_tab, sections_tab, text_tab = st.tabs(
        [
            "📊 Overview",
            "📑 Sections",
            "🔎 Extracted Text"
        ]
    )

    # ========================================================
    # OVERVIEW
    # ========================================================

    with overview_tab:

        left, right = st.columns(2)

        with left:

            st.subheader("🛠️ Skills Detected")

            if resume_skills:

                for skill in resume_skills:

                    st.write(
                        f"• {skill}"
                    )

            else:

                st.info(
                    "No predefined technical skills detected."
                )

        with right:

            st.subheader("📑 Resume Structure")

            for section_name in sections.keys():

                st.write(
                    f"• {section_name}"
                )

    # ========================================================
    # SECTIONS
    # ========================================================

    with sections_tab:

        st.subheader(
            "Detected Resume Sections"
        )

        for section_name, section_text in sections.items():

            with st.expander(
                f"📌 {section_name}"
            ):

                st.write(section_text)

    # ========================================================
    # EXTRACTED TEXT
    # ========================================================

    with text_tab:

        st.subheader(
            "Extracted PDF Text"
        )

        st.text_area(
            "Extracted text",
            resume_text,
            height=500,
            label_visibility="collapsed"
        )


# ============================================================
# JOB MATCH ANALYZER
# ============================================================

if st.session_state.resume_text:

    st.divider()

    st.header("💼 Job Match Analyzer")

    st.write(
        "Compare your resume with a job description "
        "using local skill and keyword analysis."
    )

    job_description = st.text_area(
        "Paste Job Description",
        height=220,
        placeholder=(
            "Example:\n\n"
            "We are looking for a Data Analyst with "
            "strong SQL, Python, Excel and Power BI skills..."
        )
    )

    if st.button(
        "🔍 Analyze Resume Against Job",
        type="primary",
        use_container_width=True
    ):

        if not job_description.strip():

            st.warning(
                "Please paste a job description first."
            )

        else:

            with st.spinner(
                "Analyzing job requirements..."
            ):

                result = analyze_resume(
                    st.session_state.resume_text,
                    job_description
                )

            st.session_state.analysis = result


# ============================================================
# JOB ANALYSIS RESULTS
# ============================================================

if st.session_state.analysis:

    result = st.session_state.analysis

    st.divider()

    st.header("📊 Job Match Report")

    score_col1, score_col2 = st.columns([1, 2])

    with score_col1:

        st.metric(
            "🎯 Resume Match",
            f"{result['score']}%"
        )

    with score_col2:

        st.write("Match indicator")

        st.progress(
            result["score"] / 100
        )

        st.caption(
            "Based on technical-skill and keyword overlap."
        )

    st.divider()

    skill_col1, skill_col2 = st.columns(2)

    with skill_col1:

        st.subheader("✅ Matching Skills")

        if result["matching_skills"]:

            for skill in result["matching_skills"]:

                st.success(
                    f"✓ {skill}"
                )

        else:

            st.info(
                "No matching skills detected."
            )

    with skill_col2:

        st.subheader(
            "⚠️ Missing / Not Detected"
        )

        if result["missing_skills"]:

            for skill in result["missing_skills"]:

                st.warning(
                    f"• {skill}"
                )

        else:

            st.success(
                "No missing technical skills detected."
            )

    st.divider()

    keyword_col1, keyword_col2 = st.columns(2)

    with keyword_col1:

        st.subheader(
            "🔑 Matching Job Keywords"
        )

        if result["matching_keywords"]:

            st.write(
                ", ".join(
                    result["matching_keywords"]
                )
            )

        else:

            st.info(
                "No strong keyword matches detected."
            )

    with keyword_col2:

        st.subheader(
            "🔍 Missing Keywords"
        )

        if result["missing_keywords"]:

            st.write(
                ", ".join(
                    result["missing_keywords"][:30]
                )
            )

            if len(result["missing_keywords"]) > 30:

                st.caption(
                    "Showing first 30."
                )

        else:

            st.success(
                "No missing keywords detected."
            )

    st.info(
        "ℹ️ This is a local matching indicator, "
        "not an official ATS score."
    )


# ============================================================
# RESUME Q&A
# ============================================================

if st.session_state.resume_text:

    st.divider()

    st.header("💬 Ask Your Resume")

    st.write(
        "Ask questions about your skills, experience, "
        "projects, education, certifications and achievements."
    )

    question = st.text_input(
        "Your question",
        placeholder=(
            "Example: What are my technical skills?"
        )
    )

    st.caption("Suggested questions")

    q1, q2, q3, q4 = st.columns(4)

    with q1:

        if st.button(
            "🛠️ My Skills",
            use_container_width=True
        ):

            question = (
                "What are my technical skills?"
            )

    with q2:

        if st.button(
            "💼 My Experience",
            use_container_width=True
        ):

            question = (
                "What is my work experience?"
            )

    with q3:

        if st.button(
            "🚀 My Projects",
            use_container_width=True
        ):

            question = (
                "What projects have I built?"
            )

    with q4:

        if st.button(
            "🎓 My Education",
            use_container_width=True
        ):

            question = (
                "What is my education?"
            )

    if st.button(
        "🔎 Search Resume",
        type="primary",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "🔎 Searching resume knowledge base..."
            ):

                try:

                    answer, section, method = (
                        get_resume_answer(
                            st.session_state.resume_text,
                            st.session_state.vector_db,
                            question
                        )
                    )

                    st.session_state.qa_answer = answer
                    st.session_state.qa_section = section
                    st.session_state.qa_method = method

                except Exception as e:

                    st.error(
                        f"Search failed: {e}"
                    )


# ============================================================
# Q&A RESULT
# ============================================================

if st.session_state.qa_answer:

    st.divider()

    st.subheader("🤖 Resume Answer")

    info1, info2 = st.columns(2)

    with info1:

        st.caption(
            f"Retrieval: "
            f"{st.session_state.qa_method}"
        )

    with info2:

        if st.session_state.qa_section:

            st.caption(
                f"Section: "
                f"{st.session_state.qa_section}"
            )

    with st.container(border=True):

        st.write(
            st.session_state.qa_answer
        )


# ============================================================
# PROJECT PORTFOLIO
# ============================================================

if st.session_state.resume_text:

    st.divider()

    st.header("🚀 Project Portfolio")

    sections = extract_sections(
        st.session_state.resume_text
    )

    project_section = sections.get("PROJECTS")

    if project_section:

        technology_words = [

            "python",
            "java",
            "c++",
            "javascript",
            "typescript",

            "sql",
            "mysql",
            "postgresql",
            "postgres",
            "mongodb",

            "excel",
            "advanced excel",
            "power bi",
            "tableau",
            "power query",
            "dax",

            "pandas",
            "numpy",
            "matplotlib",
            "seaborn",

            "machine learning",
            "deep learning",
            "data science",
            "artificial intelligence",
            "ai",

            "nlp",
            "natural language processing",
            "computer vision",

            "tensorflow",
            "keras",
            "pytorch",
            "scikit-learn",
            "sklearn",
            "opencv",

            "k-means",
            "tf-idf",

            "streamlit",
            "langchain",
            "rag",
            "llm",
            "generative ai",

            "aws",
            "azure",
            "gcp",

            "etl",
            "data engineering",
            "data warehouse",
            "databricks",

            "git",
            "github",
            "jira"
        ]

        date_pattern = re.compile(
            r"""
            (
                \b(?:19|20)\d{2}\b
                |
                \b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)
                (?:[a-z]+)?
                \s*
                (?:19|20)?\d{2}
                |
                \b(?:19|20)\d{2}
                \s*[-–—]\s*
                (?:19|20)\d{2}
                |
                \b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b
            )
            """,
            flags=re.IGNORECASE | re.VERBOSE
        )

        url_pattern = re.compile(
            r"(https?://|www\.|github\.com|gitlab\.com|"
            r"bitbucket\.org|streamlit\.app)",
            flags=re.IGNORECASE
        )

        bullet_pattern = re.compile(
            r"^\s*(?:•|●|▪|◦|‣|⁃|-|\*|→|➜|➤)\s*"
        )

        metadata_pattern = re.compile(
            r"^\s*(?:"
            r"tech(?:nologies)?\s*stack"
            r"|technologies"
            r"|technology"
            r"|tools"
            r"|tools\s*used"
            r"|technologies\s*used"
            r"|skills"
            r"|skills\s*used"
            r"|role"
            r"|duration"
            r"|date"
            r"|link"
            r"|links"
            r"|github"
            r"|website"
            r"|url"
            r")\s*[:\-]",
            flags=re.IGNORECASE
        )

        description_verbs = [
            "developed",
            "created",
            "built",
            "implemented",
            "designed",
            "analyzed",
            "trained",
            "performed",
            "conducted",
            "deployed",
            "used",
            "worked",
            "managed",
            "processed",
            "predicted",
            "classified",
            "visualized",
            "automated",
            "integrated",
            "evaluated",
            "optimized",
            "generated",
            "extracted",
            "cleaned",
            "transformed"
        ]

        def project_title_score(line):

            text = line.strip()

            if not text:
                return -100

            lower = text.lower()

            score = 0

            word_count = len(text.split())

            if 2 <= word_count <= 7:
                score += 3

            elif word_count <= 10:
                score += 1

            else:
                score -= 5

            if len(text) < 3:
                score -= 5

            if bullet_pattern.match(text):
                score -= 6

            if url_pattern.search(lower):
                score -= 7

            if date_pattern.search(lower):
                score -= 4

            if metadata_pattern.match(text):
                score -= 7

            technology_count = sum(
                1
                for tech in technology_words
                if tech.lower() in lower
            )

            if technology_count >= 3:
                score -= 5

            elif technology_count == 2:
                score -= 2

            if "|" in text:
                score -= 2

            first_word = (
                text.split()[0].lower()
                if text.split()
                else ""
            )

            if first_word in description_verbs:
                score -= 5

            if text.endswith(
                (".", "?", "!")
            ):
                score -= 3

            if text == text.title():
                score += 2

            if text.isupper() and len(text) > 3:
                score += 1

            if ":" in text:
                score -= 2

            return score

        raw_lines = project_section.split("\n")

        lines = []

        for raw_line in raw_lines:

            line = raw_line.strip()

            if not line:
                continue

            line = re.sub(
                r"\s+",
                " ",
                line
            )

            lines.append(line)

        title_candidates = []

        for index, line in enumerate(lines):

            score = project_title_score(line)

            previous_line = (
                lines[index - 1]
                if index > 0
                else ""
            )

            next_line = (
                lines[index + 1]
                if index + 1 < len(lines)
                else ""
            )

            if previous_line:

                previous_lower = previous_line.lower()

                if (
                    "project" in previous_lower
                    and len(previous_line.split()) <= 5
                ):

                    score += 2

            if next_line:

                next_lower = next_line.lower()

                next_technology_count = sum(
                    1
                    for tech in technology_words
                    if tech.lower() in next_lower
                )

                if next_technology_count >= 1:
                    score += 2

            if score >= 3:

                title_candidates.append(
                    {
                        "index": index,
                        "title": line,
                        "score": score
                    }
                )

        filtered_candidates = []

        for candidate in title_candidates:

            title = candidate["title"]

            lower_title = title.lower()

            technology_count = sum(
                1
                for tech in technology_words
                if tech.lower() in lower_title
            )

            if (
                technology_count >= 3
                and len(title.split()) <= 8
            ):
                continue

            if metadata_pattern.match(title):
                continue

            if url_pattern.search(lower_title):
                continue

            if bullet_pattern.match(title):
                continue

            filtered_candidates.append(candidate)

        title_candidates = filtered_candidates

        projects = []

        for i, candidate in enumerate(title_candidates):

            title_index = candidate["index"]

            title = candidate["title"]

            if i + 1 < len(title_candidates):

                next_index = (
                    title_candidates[i + 1]["index"]
                )

            else:

                next_index = len(lines)

            content_lines = []

            for content_index in range(
                title_index + 1,
                next_index
            ):

                content_line = lines[content_index]

                if content_line.lower() == "project":
                    continue

                content_lines.append(content_line)

            content = "\n".join(
                content_lines
            ).strip()

            projects.append(
                {
                    "title": title,
                    "content": content,
                    "score": candidate["score"]
                }
            )

        final_projects = []

        for project in projects:

            title = project["title"]

            normalized_title = re.sub(
                r"[^a-z0-9 ]",
                "",
                title.lower()
            ).strip()

            duplicate = False

            for existing in final_projects:

                existing_title = re.sub(
                    r"[^a-z0-9 ]",
                    "",
                    existing["title"].lower()
                ).strip()

                if normalized_title == existing_title:

                    duplicate = True
                    break

            if not duplicate:
                final_projects.append(project)

        projects = final_projects

        if projects:

            st.caption(
                f"Detected {len(projects)} "
                f"project(s) automatically from the resume."
            )

            project_cols = st.columns(3)

            for index, project in enumerate(projects):

                with project_cols[index % 3]:

                    with st.container(border=True):

                        st.subheader(
                            f"🚀 {project['title']}"
                        )

                        content = project["content"]

                        technologies = []

                        content_lower = content.lower()

                        for technology in technology_words:

                            if technology.lower() in content_lower:

                                technologies.append(
                                    technology
                                )

                        if technologies:

                            technology_display = []

                            for technology in technologies:

                                technology_display.append(
                                    technology.title()
                                )

                            st.caption(
                                " • ".join(
                                    technology_display
                                )
                            )

                        if content:

                            clean_content = content

                            urls = re.findall(
                                r"https?://\S+|www\.\S+",
                                clean_content,
                                flags=re.IGNORECASE
                            )

                            if urls:

                                for url in urls:

                                    clean_content = (
                                        clean_content.replace(
                                            url,
                                            ""
                                        )
                                    )

                            clean_content = re.sub(
                                r"\s+",
                                " ",
                                clean_content
                            ).strip()

                            if clean_content:

                                st.write(
                                    clean_content
                                )

                            if urls:

                                st.caption(
                                    "🔗 Project Link"
                                )

                                for url in urls:

                                    st.write(url)

                        else:

                            st.caption(
                                "Project details detected "
                                "from the resume."
                            )

        else:

            st.info(
                "Projects were detected, but individual "
                "project titles could not be identified automatically."
            )

    else:

        st.info(
            "No PROJECTS section was detected."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🤖 AI Resume Analyzer & Job Match Assistant"
)

st.caption(
    "Python • Streamlit • PyPDF • HuggingFace Embeddings • "
    "Chroma • Section-Aware RAG • Local Processing"
)
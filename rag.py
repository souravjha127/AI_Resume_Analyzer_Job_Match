import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# SECTION NAMES
# ============================================================

SECTION_NAMES = [
    "SUMMARY",
    "PROFILE",
    "EXPERIENCE",
    "WORK EXPERIENCE",
    "PROJECTS",
    "EDUCATION",
    "SKILLS",
    "TECHNICAL SKILLS",
    "CERTIFICATIONS",
    "ACHIEVEMENTS",
    "AWARDS",
    "LANGUAGES",
    "INTERESTS"
]


# ============================================================
# PROJECT / TECHNOLOGY KEYWORDS
# ============================================================

PROJECT_KEYWORDS = [
    "power bi",
    "python",
    "sql",
    "postgresql",
    "postgres",
    "excel",
    "tableau",
    "power query",
    "dax",
    "machine learning",
    "deep learning",
    "tensorflow",
    "keras",
    "nlp",
    "natural language processing",
    "computer vision",
    "opencv",
    "scikit-learn",
    "sklearn",
    "k-means",
    "tf-idf",
    "streamlit",
    "aws",
    "data engineering",
    "data warehouse",
    "langchain",
    "rag",
    "llm",
    "generative ai"
]


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_resume_text(text):

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = text.replace("▪", "•")
    text = text.replace("●", "•")
    text = text.replace("◦", "•")
    text = text.replace("‣", "•")
    text = text.replace("⁃", "•")
    text = text.replace("\uf0b7", "•")

    text = text.replace("\xa0", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# CHECK SECTION HEADING
# ============================================================

def is_section_heading(line):

    if not line:
        return False

    cleaned = line.strip()

    if not cleaned:
        return False

    upper_line = cleaned.upper()

    normalized = re.sub(
        r"[^A-Z ]",
        "",
        upper_line
    ).strip()

    return normalized in SECTION_NAMES


# ============================================================
# EXTRACT SECTIONS
# ============================================================

def extract_sections(text):

    text = clean_resume_text(text)

    if not text:
        return {}

    lines = text.split("\n")

    sections = {}

    current_section = "SUMMARY"

    sections[current_section] = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        upper_line = stripped.upper()

        matched_section = None

        for section in SECTION_NAMES:

            if upper_line == section:

                matched_section = section
                break

            if re.fullmatch(
                rf"{re.escape(section)}\s*[:\-]?",
                upper_line
            ):

                matched_section = section
                break

        if matched_section:

            current_section = matched_section

            if current_section not in sections:

                sections[current_section] = []

            continue

        sections[current_section].append(
            stripped
        )

    final_sections = {}

    for section, content in sections.items():

        section_text = "\n".join(
            content
        ).strip()

        if section_text:

            final_sections[section] = section_text

    return final_sections


# ============================================================
# SPLIT GENERAL TEXT
# ============================================================

def split_text_into_chunks(
    section_name,
    text,
    max_words=120
):

    words = text.split()

    if not words:
        return []

    chunks = []

    current = []

    for word in words:

        current.append(word)

        if len(current) >= max_words:

            chunks.append(
                f"{section_name}\n\n"
                + " ".join(current)
            )

            current = []

    if current:

        chunks.append(
            f"{section_name}\n\n"
            + " ".join(current)
        )

    return chunks


# ============================================================
# EXPERIENCE
# ============================================================

def split_experience(text):

    text = clean_resume_text(text)

    if not text:
        return []

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    if not lines:
        return []

    experience_text = "\n".join(
        lines
    ).strip()

    return [
        "EXPERIENCE\n\n"
        + experience_text
    ]


# ============================================================
# SPLIT PROJECTS
# ============================================================

def split_projects(text):

    project_patterns = [

        r"Customer Review Segmentation",

        r"Flower Recognition",

        r"Flower Recognition Model",

        r"Blinkit Sales Dashboard",

        r"Music Store Data Analysis",

        r"Music Store",

        r"Spam.*Classifier",

        r"PostgreSQL.*Warehouse",

        r"IPL Prediction Dashboard"

    ]

    positions = []

    for pattern in project_patterns:

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            positions.append(
                (
                    match.start(),
                    match.group(0)
                )
            )

    positions.sort(
        key=lambda x: x[0]
    )

    unique_positions = []

    seen_positions = set()

    for position, title in positions:

        if position not in seen_positions:

            seen_positions.add(
                position
            )

            unique_positions.append(
                (
                    position,
                    title
                )
            )

    positions = unique_positions

    chunks = []

    for i, (
        start,
        title
    ) in enumerate(positions):

        if i + 1 < len(positions):

            end = positions[
                i + 1
            ][0]

        else:

            end = len(text)

        project = text[
            start:end
        ].strip()

        if project:

            chunks.append(
                "PROJECTS\n\n"
                + project
            )

    if not chunks:

        chunks.append(
            "PROJECTS\n\n"
            + text
        )

    return chunks


# ============================================================
# SPLIT SKILLS
# ============================================================

def split_skills(text):

    text = clean_resume_text(text)

    chunks = []

    technical_match = re.search(
        r"Technical Skills\s*:\s*(.*?)(?="
        r"Behavioral Skills\s*:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if technical_match:

        technical = (
            technical_match
            .group(1)
            .strip()
        )

        if technical:

            chunks.append(
                "SKILLS\n\n"
                "Technical Skills: "
                + technical
            )

    behavioral_match = re.search(
        r"Behavioral Skills\s*:\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if behavioral_match:

        behavioral = (
            behavioral_match
            .group(1)
            .strip()
        )

        if behavioral:

            chunks.append(
                "SKILLS\n\n"
                "Behavioral Skills: "
                + behavioral
            )

    if not chunks:

        chunks.append(
            "SKILLS\n\n"
            + text
        )

    return chunks


# ============================================================
# CREATE RESUME CHUNKS
# ============================================================

def create_chunks(text):

    sections = extract_sections(
        text
    )

    chunks = []

    for section_name, section_text in sections.items():

        if section_name in [
            "EXPERIENCE",
            "WORK EXPERIENCE"
        ]:

            chunks.extend(
                split_experience(
                    section_text
                )
            )

        elif section_name == "PROJECTS":

            chunks.extend(
                split_projects(
                    section_text
                )
            )

        elif section_name in [
            "SKILLS",
            "TECHNICAL SKILLS"
        ]:

            chunks.extend(
                split_skills(
                    section_text
                )
            )

        else:

            chunks.extend(
                split_text_into_chunks(
                    section_name,
                    section_text
                )
            )

    final_chunks = []

    seen = set()

    for chunk in chunks:

        chunk = chunk.strip()

        if not chunk:
            continue

        normalized = re.sub(
            r"\s+",
            " ",
            chunk.lower()
        )

        if normalized not in seen:

            seen.add(
                normalized
            )

            final_chunks.append(
                chunk
            )

    return final_chunks


# ============================================================
# DETECT QUESTION SECTION
# ============================================================

def detect_section(question):

    question = question.lower().strip()

    # --------------------------------------------------------
    # EXPERIENCE FIRST
    # --------------------------------------------------------

    if any(
        word in question
        for word in [
            "experience",
            "work experience",
            "job experience",
            "employment",
            "professional experience",
            "work history",
            "worked at",
            "previous job",
            "previous work",
            "company",
            "companies",
            "employer",
            "employers",
            "jus jumpin"
        ]
    ):

        return "EXPERIENCE"

    # --------------------------------------------------------
    # PROJECTS
    # --------------------------------------------------------

    if any(
        word in question
        for word in [
            "project",
            "projects",
            "portfolio",
            "dashboard",
            "model",
            "application",
            "applications"
        ]
    ):

        return "PROJECTS"

    # --------------------------------------------------------
    # SKILLS
    # --------------------------------------------------------

    if any(
        word in question
        for word in [
            "skill",
            "skills",
            "technical skill",
            "technical skills",
            "technology",
            "technologies",
            "tech stack",
            "programming language",
            "programming languages",
            "tools"
        ]
    ):

        return "SKILLS"

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    if any(
        word in question
        for word in [
            "education",
            "degree",
            "college",
            "university",
            "btech",
            "b.tech",
            "graduation",
            "graduate",
            "cgpa",
            "academic",
            "qualification",
            "school"
        ]
    ):

        return "EDUCATION"

    # --------------------------------------------------------
    # CERTIFICATIONS
    # --------------------------------------------------------

    if any(
        word in question
        for word in [
            "certification",
            "certifications",
            "certificate",
            "certificates",
            "course",
            "courses"
        ]
    ):

        return "CERTIFICATIONS"

    # --------------------------------------------------------
    # ACHIEVEMENTS
    # --------------------------------------------------------

    if any(
        word in question
        for word in [
            "achievement",
            "achievements",
            "award",
            "awards",
            "hackerrank",
            "hacker rank",
            "leetcode",
            "accomplishment"
        ]
    ):

        return "ACHIEVEMENTS"

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if any(
        word in question
        for word in [
            "summary",
            "profile",
            "about me",
            "introduce",
            "introduction",
            "background",
            "career objective"
        ]
    ):

        return "SUMMARY"

    return None


# ============================================================
# GENERIC EXPERIENCE QUESTION
# ============================================================

def is_generic_experience_question(question):

    question = question.lower().strip()

    generic_phrases = [

        "what is my experience",
        "what is my work experience",
        "tell me about my experience",
        "tell me about my work experience",
        "show my experience",
        "show my work experience",
        "list my experience",
        "list my work experience",
        "describe my experience",
        "describe my work experience",
        "summarize my experience",
        "summarize my work experience",
        "what have i worked on",
        "where have i worked",
        "where did i work",
        "my previous experience",
        "my previous work",
        "my job experience",
        "my professional experience"
    ]

    return any(
        phrase in question
        for phrase in generic_phrases
    )


# ============================================================
# GENERIC PROJECT QUESTION
# ============================================================

def is_generic_project_question(question):

    question = question.lower().strip()

    generic_phrases = [

        "what are my projects",
        "what projects have i built",
        "what projects have i worked on",
        "what projects did i build",
        "what projects did i work on",
        "list my projects",
        "show my projects",
        "tell me about my projects",
        "tell me about my project",
        "describe my projects",
        "describe my project",
        "my project portfolio",
        "projects have i built",
        "projects have i made"
    ]

    return any(
        phrase in question
        for phrase in generic_phrases
    )


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vector_database(text):

    chunks = create_chunks(
        text
    )

    if not chunks:

        raise ValueError(
            "No useful resume information could be extracted."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vector_db


# ============================================================
# NORMALIZE SEARCH TEXT
# ============================================================

def normalize_search_text(text):

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
# FIND PROJECT / TECHNOLOGY TERMS
# ============================================================

def find_query_terms(question):

    question_lower = normalize_search_text(
        question
    )

    found_terms = []

    project_names = [
        "blinkit",
        "customer review segmentation",
        "flower recognition",
        "music store",
        "spam",
        "ipl",
        "postgresql warehouse"
    ]

    for term in project_names:

        if term in question_lower:

            found_terms.append(
                term
            )

    for term in PROJECT_KEYWORDS:

        if term in question_lower:

            found_terms.append(
                term
            )

    return found_terms


# ============================================================
# FILTER PROJECT RESULTS
# ============================================================

def filter_project_results(
    results,
    question
):

    query_terms = find_query_terms(
        question
    )

    if not query_terms:

        return results

    filtered = []

    for result in results:

        content = normalize_search_text(
            result.page_content
        )

        for term in query_terms:

            if term in content:

                filtered.append(
                    result
                )

                break

    return filtered


# ============================================================
# SEARCH RESUME
# ============================================================

def search_resume(
    vector_db,
    question,
    k=3
):

    detected_section = detect_section(
        question
    )

    # ========================================================
    # EXPERIENCE
    # ========================================================

    if (
        detected_section == "EXPERIENCE"
        and is_generic_experience_question(
            question
        )
    ):

        results = vector_db.similarity_search(
            "EXPERIENCE",
            k=10
        )

        experience_results = []

        for result in results:

            content = result.page_content.strip()

            if content.upper().startswith(
                "EXPERIENCE"
            ):

                body = re.sub(
                    r"^EXPERIENCE\s*",
                    "",
                    content,
                    flags=re.IGNORECASE
                ).strip()

                if len(
                    body.split()
                ) >= 5:

                    experience_results.append(
                        result
                    )

        if experience_results:

            return experience_results[:k]

    # ========================================================
    # PROJECTS
    # ========================================================

    if (
        detected_section == "PROJECTS"
        and is_generic_project_question(
            question
        )
    ):

        results = vector_db.similarity_search(
            "PROJECTS",
            k=10
        )

        project_results = []

        for result in results:

            content = result.page_content.strip()

            if content.upper().startswith(
                "PROJECTS"
            ):

                project_results.append(
                    result
                )

        if project_results:

            return project_results[:k]

    # ========================================================
    # NORMAL SECTION-AWARE SEARCH
    # ========================================================

    if detected_section:

        enhanced_question = (
            f"{detected_section}. "
            f"Find information specifically "
            f"from the {detected_section} section. "
            f"Question: {question}"
        )

    else:

        enhanced_question = question

    results = vector_db.similarity_search(
        enhanced_question,
        k=max(k, 5)
    )

    # ========================================================
    # SECTION FILTER
    # ========================================================

    if detected_section:

        section_results = []

        for result in results:

            content = result.page_content.upper()

            if content.startswith(
                detected_section
            ):

                section_results.append(
                    result
                )

        if section_results:

            results = section_results

    # ========================================================
    # PROJECT FILTER
    # ========================================================

    if detected_section == "PROJECTS":

        filtered_results = filter_project_results(
            results,
            question
        )

        if filtered_results:

            results = filtered_results

    # ========================================================
    # EXPERIENCE SAFETY FILTER
    # ========================================================

    if detected_section == "EXPERIENCE":

        valid_results = []

        for result in results:

            content = result.page_content.strip()

            body = re.sub(
                r"^EXPERIENCE\s*",
                "",
                content,
                flags=re.IGNORECASE
            ).strip()

            if len(
                body.split()
            ) >= 5:

                valid_results.append(
                    result
                )

        if valid_results:

            results = valid_results

    return results[:k]
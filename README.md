# AI Resume Analyzer & Job Match Assistant

A portfolio-ready resume intelligence application built with **Python, Streamlit, RAG, Hugging Face Embeddings, and ChromaDB**.

The application analyzes an uploaded resume, extracts structured career information, provides resume-based question answering, and compares a resume against a job description to identify relevant skills, experience, and potential gaps.

---

## 🚀 Live Demo

**Streamlit App:**  
[Open the Live Application](YOUR_STREAMLIT_APP_URL)

> Replace `YOUR_STREAMLIT_APP_URL` with your deployed Streamlit application URL after deployment.

---

## 📌 Overview

Job seekers often need to understand two things:

1. **How strong and complete is my resume?**
2. **How well does my resume match a particular job description?**

This project combines resume analysis with **Retrieval-Augmented Generation (RAG)** techniques to create an interactive resume assistant.

Users can upload their resume in PDF format, explore extracted information, analyze their profile, compare it with a job description, and ask questions about their resume.

---

## ✨ Key Features

### 📄 Resume Analysis

- Upload a resume in PDF format
- Extract text automatically using PyPDF
- Detect major resume sections
- Analyze skills, education, experience, projects, certifications, and achievements
- Present extracted information through an interactive dashboard

### 🎯 Job Match Analysis

- Paste a job description
- Compare resume information with job requirements
- Identify relevant and matching skills
- Identify missing or less-relevant requirements
- Analyze the overall alignment between the resume and the target role

### 🔎 Resume Q&A with RAG

The application uses a section-aware retrieval approach to answer questions from the uploaded resume.

Users can ask questions such as:

- What is my work experience?
- What projects have I worked on?
- What are my technical skills?
- What certifications do I have?
- What is my educational background?

The retrieval pipeline uses:

**Resume → Chunking → Embeddings → Vector Database → Relevant Retrieval**

### 📊 Interactive Resume Dashboard

The application provides dedicated sections for:

- Resume overview
- Resume analysis
- Job matching
- Skills
- Experience
- Projects
- Certifications
- Resume Q&A

---

## 🧠 RAG Architecture

```text
                 Resume PDF
                     │
                     ▼
             PDF Text Extraction
                     │
                     ▼
            Resume Section Detection
                     │
                     ▼
              Intelligent Chunking
                     │
                     ▼
          Hugging Face Embeddings
                     │
                     ▼
              ChromaDB Vector Store
                     │
                     ▼
           Section-Aware Retrieval
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     Resume Q&A            Job Matching
          │                     │
          └──────────┬──────────┘
                     ▼
              Resume Insights

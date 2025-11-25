import streamlit as st
from groq import Groq
from pypdf import PdfReader
from docx import Document
import io
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Configure page
st.set_page_config(page_title="AI Hiring Assistant", page_icon="🤖", layout="wide")

# Initialize session state
if 'resumes_data' not in st.session_state:
    st.session_state.resumes_data = []
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'resume_vectors' not in st.session_state:
    st.session_state.resume_vectors = None
if 'jd_history' not in st.session_state:
    st.session_state.jd_history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'parsed_resumes' not in st.session_state:
    st.session_state.parsed_resumes = []

# Groq API Configuration
def configure_groq(api_key):
    """Configure Groq client"""
    client = Groq(api_key=api_key)
    return client

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(io.BytesIO(pdf_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = Document(io.BytesIO(docx_file.read()))
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# AI-powered resume parsing using Groq
def parse_resume_with_ai(client, resume_text, filename):
    prompt = f"""Analyze this resume and extract structured information in JSON format.

Resume Content:
{resume_text}

Extract the following information and return ONLY a valid JSON object (no markdown, no extra text):
{{
    "name": "candidate's full name",
    "email": "email address",
    "phone": "phone number",
    "skills": ["skill1", "skill2", "skill3"],
    "experience_years": "total years of experience as a number",
    "education": ["degree1", "degree2"],
    "job_titles": ["title1", "title2"],
    "summary": "brief professional summary in 2-3 sentences",
    "key_achievements": ["achievement1", "achievement2"],
    "technologies": ["tech1", "tech2"],
    "certifications": ["cert1", "cert2"]
}}

If any field is not found, use empty string or empty array. Return ONLY the JSON object."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2000
        )
        
        # Clean the response to extract JSON
        json_text = chat_completion.choices[0].message.content.strip()
        json_text = re.sub(r'^```json\s*', '', json_text)
        json_text = re.sub(r'\s*```$', '', json_text)
        parsed_data = json.loads(json_text)
        parsed_data['filename'] = filename
        parsed_data['raw_text'] = resume_text
        return parsed_data
    except Exception as e:
        return {
            "filename": filename,
            "raw_text": resume_text,
            "name": "Unknown",
            "email": "",
            "phone": "",
            "skills": [],
            "experience_years": "0",
            "education": [],
            "job_titles": [],
            "summary": "Unable to parse resume",
            "key_achievements": [],
            "technologies": [],
            "certifications": [],
            "error": str(e)
        }

# AI-powered JD generation using Groq
def generate_jd_from_conversation(client, conversation):
    prompt = f"""Based on this conversation about a job requirement, generate a comprehensive and professional Job Description.

Conversation:
{conversation}

Generate a complete, professional Job Description that includes:
1. Job Title
2. Company Overview (create a professional generic overview)
3. Role Summary
4. Key Responsibilities (at least 5-7 points)
5. Required Qualifications
6. Preferred Qualifications
7. Technical Skills Required
8. Soft Skills Required
9. Experience Level
10. Benefits (create attractive generic benefits)
11. Equal Opportunity Statement

Make it professional, detailed, and attractive to candidates. Format it nicely with proper sections and bullet points."""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=3000
    )
    return chat_completion.choices[0].message.content

# AI-powered candidate scoring using Groq
def score_candidates_with_ai(client, job_requirements, parsed_resumes):
    # Prepare resume summaries
    resume_summaries = []
    for idx, resume in enumerate(parsed_resumes):
        summary = f"""
Candidate {idx + 1}: {resume['filename']}
Name: {resume.get('name', 'Unknown')}
Experience: {resume.get('experience_years', 'Unknown')} years
Skills: {', '.join(resume.get('skills', [])[:10])}
Technologies: {', '.join(resume.get('technologies', [])[:10])}
Job Titles: {', '.join(resume.get('job_titles', [])[:3])}
Summary: {resume.get('summary', 'N/A')}
"""
        resume_summaries.append(summary)
    
    all_resumes_text = "\n---\n".join(resume_summaries)
    
    prompt = f"""You are an expert HR recruiter. Analyze these candidates against the job requirements and provide detailed scoring.

Job Requirements:
{job_requirements}

Candidates:
{all_resumes_text}

For each candidate, provide a comprehensive analysis in JSON format:
{{
    "candidates": [
        {{
            "candidate_number": 1,
            "name": "candidate name",
            "overall_score": 85,
            "skills_match": 90,
            "experience_match": 80,
            "education_match": 85,
            "strengths": ["strength1", "strength2"],
            "weaknesses": ["weakness1", "weakness2"],
            "recommendation": "Highly Recommended/Recommended/Consider/Not Recommended",
            "reasoning": "detailed explanation of scoring",
            "interview_questions": ["question1", "question2", "question3"]
        }}
    ],
    "top_candidates": [1, 2, 3],
    "summary": "overall summary of the candidate pool"
}}

Return ONLY the JSON object, no markdown or extra text."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=4000
        )
        
        json_text = chat_completion.choices[0].message.content.strip()
        json_text = re.sub(r'^```json\s*', '', json_text)
        json_text = re.sub(r'\s*```$', '', json_text)
        scoring_data = json.loads(json_text)
        return scoring_data
    except Exception as e:
        return {"error": str(e), "candidates": [], "top_candidates": [], "summary": "Unable to score candidates"}

# AI conversation handler using Groq
def ai_conversation(client, user_message, context):
    prompt = f"""You are an intelligent AI Hiring Assistant. You help recruiters with:
1. Understanding job requirements
2. Generating job descriptions
3. Analyzing resumes
4. Finding the best candidates
5. Answering questions about candidates

Context:
{context}

User Message: {user_message}

Provide a helpful, professional, and conversational response. If the user is asking about candidates or job requirements, be specific and data-driven."""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=2000
    )
    return chat_completion.choices[0].message.content

# Smart candidate search using Groq
def smart_candidate_search(client, query, parsed_resumes, vectorizer, resume_vectors):
    # Vector-based search
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, resume_vectors)[0]
    
    # Get top candidates
    top_indices = np.argsort(similarities)[-10:][::-1]
    
    # Prepare context for AI
    relevant_candidates = []
    for idx in top_indices:
        if similarities[idx] > 0.05:
            resume = parsed_resumes[int(idx)]
            relevant_candidates.append({
                "index": int(idx),
                "similarity": float(similarities[idx]),
                "name": resume.get('name', 'Unknown'),
                "filename": resume['filename'],
                "skills": resume.get('skills', []),
                "experience": resume.get('experience_years', 'Unknown'),
                "summary": resume.get('summary', 'N/A')
            })
    
    # AI-powered analysis
    candidates_text = json.dumps(relevant_candidates, indent=2)
    prompt = f"""Based on this query: "{query}"

Here are the most relevant candidates found:
{candidates_text}

Provide a detailed analysis:
1. List the best matching candidates with their names
2. Explain why each candidate matches the requirements
3. Highlight their key strengths
4. Provide specific recommendations
5. Suggest next steps (interview questions, verification points)

Be specific and reference actual data from the candidates."""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=2000
    )
    return chat_completion.choices[0].message.content, relevant_candidates

# Process resumes with AI
def process_resumes_with_ai(client, resumes_data):
    parsed_resumes = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, resume in enumerate(resumes_data):
        status_text.text(f"🤖 AI is analyzing {resume['filename']}...")
        parsed = parse_resume_with_ai(client, resume['text'], resume['filename'])
        parsed_resumes.append(parsed)
        progress_bar.progress((idx + 1) / len(resumes_data))
    
    status_text.text("✅ All resumes analyzed!")
    return parsed_resumes

# Main UI
st.title("🤖 AI Hiring Assistant")
st.markdown("### Fully Automated Intelligent Recruitment System")

# Sidebar
with st.sidebar:
    st.header("⚙ Configuration")
    # Hardcoded API key for testing
    api_key = st.text_input("Groq API Key", value="your api key here", type="password", help="Your Groq API key")
    
    if api_key:
        st.success("✅ AI System Online")
        client = configure_groq(api_key)
    else:
        st.warning("⚠ Please enter API key")
        st.markdown("[Get Groq API Key](https://console.groq.com)")
        client = None
    
    st.markdown("---")
    st.markdown("### 📊 System Dashboard")
    st.metric("📄 Resumes Loaded", len(st.session_state.resumes_data))
    st.metric("✅ Resumes Analyzed", len(st.session_state.parsed_resumes))
    st.metric("📝 JDs Generated", len(st.session_state.jd_history))
    
    if st.button("🗑 Reset System"):
        st.session_state.resumes_data = []
        st.session_state.parsed_resumes = []
        st.session_state.vectorizer = None
        st.session_state.resume_vectors = None
        st.session_state.jd_history = []
        st.session_state.chat_history = []
        st.rerun()

# Main Interface
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📝 AI JD Generator", "📂 Resume Intelligence", "🔍 Candidate Search"])

# Tab 1: Home / Dashboard
with tab1:
    st.header("Welcome to AI Hiring Assistant")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Step 1")
        st.markdown("Generate Job Description")
        st.markdown("Tell AI what you need, it creates a professional JD")
    with col2:
        st.markdown("### 📤 Step 2")
        st.markdown("Upload Resumes")
        st.markdown("AI automatically parses and analyzes each resume")
    with col3:
        st.markdown("### 🏆 Step 3")
        st.markdown("Find Best Candidates")
        st.markdown("AI matches, scores, and ranks candidates")
    
    st.markdown("---")
    
    # Quick stats
    if st.session_state.parsed_resumes:
        st.markdown("### 📊 Quick Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate stats
        total_candidates = len(st.session_state.parsed_resumes)
        avg_experience = np.mean([float(r.get('experience_years', 0)) for r in st.session_state.parsed_resumes if r.get('experience_years', '0').replace('.', '').isdigit()])
        
        all_skills = []
        for r in st.session_state.parsed_resumes:
            all_skills.extend(r.get('skills', []))
        top_skills = {}
        for skill in all_skills:
            top_skills[skill] = top_skills.get(skill, 0) + 1
        
        with col1:
            st.metric("Total Candidates", total_candidates)
        with col2:
            st.metric("Avg Experience", f"{avg_experience:.1f} years")
        with col3:
            if top_skills:
                top_skill = max(top_skills.items(), key=lambda x: x[1])
                st.metric("Most Common Skill", top_skill[0])
        with col4:
            st.metric("Unique Skills", len(set(all_skills)))
        
        # Show parsed candidates
        st.markdown("### 👥 Analyzed Candidates")
        for idx, resume in enumerate(st.session_state.parsed_resumes[:5]):
            with st.expander(f"👤 {resume.get('name', 'Unknown')} - {resume['filename']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Experience:** {resume.get('experience_years', 'Unknown')} years")
                    st.markdown(f"**Email:** {resume.get('email', 'N/A')}")
                    st.markdown(f"**Phone:** {resume.get('phone', 'N/A')}")
                with col2:
                    st.markdown(f"**Skills:** {', '.join(resume.get('skills', [])[:5])}")
                    st.markdown(f"**Technologies:** {', '.join(resume.get('technologies', [])[:5])}")
                st.markdown(f"**Summary:** {resume.get('summary', 'N/A')}")

# Tab 2: AI JD Generator
with tab2:
    st.header("🤖 AI-Powered Job Description Generator")
    st.markdown("Describe what you need, and AI will create a complete professional JD")
    
    # Conversational JD generation
    st.markdown("### 💬 Chat with AI to Create JD")
    
    user_input = st.text_area(
        "Describe the position you want to hire for",
        placeholder="Example: I need a senior backend developer with Python and AWS experience for our fintech startup. Should have 5+ years experience and be comfortable with microservices architecture.",
        height=150
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        generate_jd = st.button("🚀 Generate JD", type="primary")
    with col2:
        if st.session_state.jd_history:
            refine_jd = st.button("✨ Refine Last JD")
        else:
            refine_jd = False
    
    if generate_jd and user_input:
        if not client:
            st.error("❌ Please configure Groq API key")
        else:
            with st.spinner("🤖 AI is crafting your Job Description..."):
                jd = generate_jd_from_conversation(client, user_input)
                st.session_state.jd_history.append({
                    "requirement": user_input,
                    "jd": jd
                })
                
                st.markdown("### ✅ Generated Job Description")
                st.markdown(jd)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download JD (TXT)",
                        jd,
                        file_name="job_description.txt",
                        mime="text/plain"
                    )
                with col2:
                    st.download_button(
                        "📥 Download JD (MD)",
                        jd,
                        file_name="job_description.md",
                        mime="text/markdown"
                    )
    
    # Show JD history
    if st.session_state.jd_history:
        st.markdown("---")
        st.markdown("### 📚 Generated JD History")
        for idx, jd_item in enumerate(reversed(st.session_state.jd_history)):
            with st.expander(f"JD #{len(st.session_state.jd_history) - idx}: {jd_item['requirement'][:50]}..."):
                st.markdown(jd_item['jd'])
                st.download_button(
                    "📥 Download",
                    jd_item['jd'],
                    file_name=f"jd_{idx}.txt",
                    key=f"download_jd_{idx}"
                )

# Tab 3: Resume Intelligence
with tab3:
    st.header("📂 Resume Intelligence & Auto-Parsing")
    st.markdown("Upload resumes and let AI automatically extract and analyze all information")
    
    uploaded_files = st.file_uploader(
        "Upload Resume Files (PDF or DOCX)",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help="AI will automatically parse and extract structured information"
    )
    
    if uploaded_files:
        if st.button("🤖 Let AI Analyze Resumes", type="primary"):
            if not client:
                st.error("❌ Please configure Groq API key")
            else:
                # Extract text
                st.session_state.resumes_data = []
                with st.spinner("📄 Extracting text from resumes..."):
                    for file in uploaded_files:
                        try:
                            if file.name.endswith('.pdf'):
                                text = extract_text_from_pdf(file)
                            else:
                                text = extract_text_from_docx(file)
                            
                            st.session_state.resumes_data.append({
                                'filename': file.name,
                                'text': text
                            })
                        except Exception as e:
                            st.error(f"Error reading {file.name}: {str(e)}")
                
                # AI parsing
                st.markdown("### 🤖 AI is analyzing resumes...")
                parsed_resumes = process_resumes_with_ai(client, st.session_state.resumes_data)
                st.session_state.parsed_resumes = parsed_resumes
                
                # Create vectors for search
                texts = [r['raw_text'] for r in parsed_resumes]
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                vectors = vectorizer.fit_transform(texts)
                st.session_state.vectorizer = vectorizer
                st.session_state.resume_vectors = vectors
                
                st.success(f"✅ Successfully analyzed {len(parsed_resumes)} resumes with AI!")
    
    # Display parsed resumes
    if st.session_state.parsed_resumes:
        st.markdown("---")
        st.markdown("### 👥 AI-Analyzed Candidates")
        
        for idx, resume in enumerate(st.session_state.parsed_resumes):
            with st.expander(f"👤 {resume.get('name', 'Unknown')} - {resume['filename']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📋 Basic Information")
                    st.markdown(f"**Name:** {resume.get('name', 'Unknown')}")
                    st.markdown(f"**Email:** {resume.get('email', 'N/A')}")
                    st.markdown(f"**Phone:** {resume.get('phone', 'N/A')}")
                    st.markdown(f"**Experience:** {resume.get('experience_years', 'Unknown')} years")
                    
                    st.markdown("#### 🎓 Education")
                    for edu in resume.get('education', []):
                        st.markdown(f"- {edu}")
                
                with col2:
                    st.markdown("#### 💼 Job Titles")
                    for title in resume.get('job_titles', []):
                        st.markdown(f"- {title}")
                    
                    st.markdown("#### 🏆 Certifications")
                    for cert in resume.get('certifications', []):
                        st.markdown(f"- {cert}")
                
                st.markdown("#### 🛠 Skills")
                st.markdown(", ".join(resume.get('skills', [])))
                
                st.markdown("#### 💻 Technologies")
                st.markdown(", ".join(resume.get('technologies', [])))
                
                st.markdown("#### ✨ Key Achievements")
                for achievement in resume.get('key_achievements', []):
                    st.markdown(f"- {achievement}")
                
                st.markdown("#### 📝 Professional Summary")
                st.markdown(resume.get('summary', 'N/A'))

# Tab 4: Candidate Search
with tab4:
    st.header("🔍 AI-Powered Candidate Search & Ranking")
    
    if not st.session_state.parsed_resumes:
        st.warning("⚠ Please upload and analyze resumes first in the 'Resume Intelligence' tab")
    else:
        # Smart search
        st.markdown("### 💬 Ask AI Anything About Your Candidates")
        
        query = st.text_area(
            "What are you looking for?",
            placeholder="Examples:\n- Find the top 3 Python developers with cloud experience\n- Who has machine learning expertise?\n- Show me candidates with leadership experience\n- Find full-stack developers with 5+ years experience\n- Who knows React and Node.js?",
            height=120
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            search_btn = st.button("🔍 Search", type="primary")
        
        if search_btn and query:
            if not client:
                st.error("❌ Please configure Groq API key")
            else:
                with st.spinner("🤖 AI is analyzing candidates..."):
                    result, candidates = smart_candidate_search(
                        client,
                        query,
                        st.session_state.parsed_resumes,
                        st.session_state.vectorizer,
                        st.session_state.resume_vectors
                    )
                    
                    st.markdown("### 🎯 AI Analysis Results")
                    st.markdown(result)
                    
                    st.markdown("### 📊 Matched Candidates")
                    for candidate in candidates:
                        with st.expander(f"👤 {candidate['name']} ({candidate['filename']}) - Match: {candidate['similarity']*100:.1f}%"):
                            st.markdown(f"**Experience:** {candidate['experience']}")
                            st.markdown(f"**Top Skills:** {', '.join(candidate['skills'][:8])}")
                            st.markdown(f"**Summary:** {candidate['summary']}")
        
        # Advanced: Score all candidates
        st.markdown("---")
        st.markdown("### 🏆 AI-Powered Candidate Scoring")
        
        if st.session_state.jd_history:
            use_jd = st.checkbox("Use latest generated JD as requirement")
            if use_jd:
                job_req = st.session_state.jd_history[-1]['requirement']
                st.info(f"Using: {job_req[:100]}...")
            else:
                job_req = st.text_area("Enter job requirements for scoring", height=100)
        else:
            job_req = st.text_area("Enter job requirements for scoring", height=100)
        
        if st.button("⚡ Score All Candidates with AI"):
            if not client:
                st.error("❌ Please configure Groq API key")
            elif not job_req:
                st.error("❌ Please enter job requirements")
            else:
                with st.spinner("🤖 AI is scoring all candidates... This may take a moment..."):
                    scoring = score_candidates_with_ai(client, job_req, st.session_state.parsed_resumes)
                    
                    if 'error' in scoring:
                        st.error(f"Error: {scoring['error']}")
                    else:
                        st.markdown("### 📊 AI Scoring Results")
                        st.markdown(f"**Summary:** {scoring.get('summary', 'N/A')}")
                        
                        st.markdown("### 🏆 Top Candidates")
                        for cand in scoring.get('candidates', []):
                            with st.expander(f"👤 {cand.get('name', 'Unknown')} - Score: {cand.get('overall_score', 0)}/100"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Overall Score", f"{cand.get('overall_score', 0)}/100")
                                    st.metric("Skills Match", f"{cand.get('skills_match', 0)}/100")
                                    st.metric("Experience Match", f"{cand.get('experience_match', 0)}/100")
                                with col2:
                                    st.metric("Education Match", f"{cand.get('education_match', 0)}/100")
                                    st.markdown(f"**Recommendation:** {cand.get('recommendation', 'N/A')}")
                                
                                st.markdown("**Strengths:**")
                                for strength in cand.get('strengths', []):
                                    st.markdown(f"✅ {strength}")
                                
                                st.markdown("**Areas to Explore:**")
                                for weakness in cand.get('weaknesses', []):
                                    st.markdown(f"⚠ {weakness}")
                                
                                st.markdown("**AI Reasoning:**")
                                st.markdown(cand.get('reasoning', 'N/A'))
                                
                                st.markdown("**Suggested Interview Questions:**")
                                for q in cand.get('interview_questions', []):
                                    st.markdown(f"❓ {q}")

# Footer
st.markdown("---")
st.markdown("### 🤖 Fully Automated AI Hiring System | Powered by Groq")
st.markdown("Using Llama 3.3 70B via Groq API")

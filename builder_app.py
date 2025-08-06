import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
from dotenv import load_dotenv
from langdetect import detect
import easyocr
import numpy as np
import fitz
from io import BytesIO
from fpdf import FPDF
from docx import Document
from fpdf import FPDF
import unicodedata
import re
from datetime import date
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import tempfile
# from weasyprint import HTML


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel(
    'gemini-1.5-flash-8b-001',
    generation_config={
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
)


# Define Gemini Pro Audio model
audio_model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame)
        return frame

def transcribe_with_gemini_from_audio(frames):
    if not frames:
        return "No audio frames captured."

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            # Open a writable WAV container
            container = av.open(temp_audio.name, mode='w', format='wav')
            stream = container.add_stream("pcm_s16le", rate=frames[0].sample_rate)
            stream.layout = frames[0].layout  # Set layout only (not channels directly)

            for frame in frames:
                container.mux(frame)
            container.close()

            # Read the WAV file
            with open(temp_audio.name, "rb") as audio_file:
                audio_data = audio_file.read()

        # Gemini transcription
        try:
            response = audio_model.generate_content([
                genai.content.FileData(mime_type="audio/wav", data=audio_data)
            ])
            return response.text

        except Exception as e:
            st.error(f"Gemini transcription failed: {e}")
            return ""

    except Exception as e:
        st.error(f"Audio processing failed: {e}")
        return ""


def detect_language(text):
    return detect(text)

def read_image(file, lang):
    image = Image.open(file)
    image_np = np.array(image)
    reader = easyocr.Reader([lang, 'en'], gpu=False)
    result = reader.readtext(image_np, detail=0)
    return ' '.join(result)




def clean_text(text):
    # Normalize Unicode and remove unsupported characters for FPDF
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

# ---------- Markdown to HTML Converter ----------
def markdown_to_html(text):
    html = text
    html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)  # Bold
    html = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html)              # Italic
    html = html.replace("\n", "<br>")                              # Newlines to <br>
    return f"<div style='font-family:Arial; font-size:15px;'>{html}</div>"

# ---------- Save as DOCX ----------
def save_as_docx(text, filename="resume.docx"):
    doc = Document()

    lines = text.split("\n")
    for line in lines:
        paragraph = doc.add_paragraph()

        # Bold: **text**
        line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
        # Italic: *text*
        line = re.sub(r"\*(.*?)\*", r"<i>\1</i>", line)

        # Split line by HTML tags for bold and italic
        tokens = re.split(r"(<b>.*?</b>|<i>.*?</i>)", line)
        for token in tokens:
            if token.startswith("<b>") and token.endswith("</b>"):
                run = paragraph.add_run(token[3:-4])
                run.bold = True
            elif token.startswith("<i>") and token.endswith("</i>"):
                run = paragraph.add_run(token[3:-4])
                run.italic = True
            else:
                paragraph.add_run(token)
    doc.save(filename)
    return filename

# ---------- Save as PDF ----------
def save_as_pdf(text, filename="resume.pdf"):
    html = markdown_to_html(text)

    output_path = f"{filename}"
    HTML(string=html).write_pdf(output_path)
    return output_path



def extract_text_from_file(file):
    try:
        if file.name.endswith(".pdf"):
            text = ""
            pdf_bytes = file.read()
            if not pdf_bytes:
                st.error("Error: Uploaded PDF is empty.")
                return None
            try:
                reader = PdfReader(BytesIO(pdf_bytes))
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n\n"
            except:
                pass
            if not text:
                try:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    for page in doc:
                        t = page.get_text("text")
                        if t:
                            text += t + "\n\n"
                except:
                    pass
            if not text:
                try:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    for page in doc:
                        pix = page.get_pixmap()
                        img_bytes = pix.tobytes("png")
                        lang = detect_language(read_image(BytesIO(img_bytes), 'en'))
                        text += read_image(BytesIO(img_bytes), lang) + "\n\n"
                except Exception as e:
                    st.error(f"OCR failed: {e}")
                    return None
            return text

        elif file.name.endswith(".txt"):
            return file.read().decode("utf-8")

        elif file.name.endswith(".docx"):
            doc = Document(file)
            return "\n".join(p.text for p in doc.paragraphs)

        elif file.name.endswith((".jpg", ".jpeg", ".png")):
            temp_text = read_image(file, 'en')
            lang = detect_language(temp_text)
            return read_image(file, lang)

        else:
            st.error("Unsupported file type.")
            return None

    except Exception as e:
        st.error(f"Text extraction failed: {e}")
        return None

def summarize_with_gemini(text, prompt_template):
    try:
        prompt = prompt_template.format(text=text)
        response = model.generate_content(prompt)
        return response.text if response.text else None
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return None

def build_user_input(data):
    lines = []

    # Personal Info
    pi = data.get("personal_info", {})
    if pi:
        lines.append(f"Name: {pi.get('name', '')}")
        lines.append(f"Email: {pi.get('email', '')}")
        lines.append(f"Phone: {pi.get('phone', '')}")
        lines.append(f"Summary: {pi.get('summary', '')}")
    
    # Experience
    if data.get("experience"):
        lines.append("\nExperience:")
        for exp in data["experience"]:
            if isinstance(exp, dict):
                lines.append(f"{exp.get('job_title', 'N/A')} at {exp.get('company', 'N/A')} ({exp.get('start', 'N/A')} to {exp.get('end', 'N/A')})")
                lines.append(f"Description: {exp.get('description', '')}")
            else:
                lines.append(str(exp))  # fallback if not dict
    
    # Education
    if data.get("education"):
        lines.append("\nEducation:")
        for edu in data["education"]:
            if isinstance(edu, dict):
                lines.append(f"{edu.get('degree', 'N/A')} from {edu.get('school', 'N/A')} ({edu.get('year', 'N/A')})")
            else:
                lines.append(str(edu))

    # Skills
    if data.get("skills"):
        if isinstance(data["skills"], list):
            lines.append(f"\nSkills: {', '.join(map(str, data['skills']))}")
        else:
            lines.append(f"\nSkills: {str(data['skills'])}")

    # Certifications
    if data.get("certifications"):
        lines.append("\nCertifications:")
        for cert in data["certifications"]:
            if isinstance(cert, dict):
                lines.append(f"{cert.get('name', 'Unknown')} issued by {cert.get('org', 'Unknown')}")
            else:
                lines.append(str(cert))  # fallback if it's just a string

    return "\n".join([l for l in lines if l.strip()])


def voice_input_section(label: str, key: str):
    st.markdown(f"🎤 **{label}**")

    webrtc_ctx = webrtc_streamer(
        key=f"rec-{key}",
        audio_receiver_size=4096,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
        audio_processor_factory=AudioProcessor,
    )

    transcript = st.empty()

    # Automatically transcribe once user stops recording
    if webrtc_ctx and not webrtc_ctx.state.playing and hasattr(webrtc_ctx.audio_processor, "frames"):
        frames = webrtc_ctx.audio_processor.frames
        if frames:
            with st.spinner("Transcribing with Gemini..."):
                result = transcribe_with_gemini_from_audio(frames)
                st.session_state.resume_data["skills_from_voice"] = result
            transcript.text_area("Transcript", value=result, key=f"text-{key}")
            return result
        else:
            st.warning("No audio recorded.")

    return ""



def main():
    st.set_page_config(page_title="Resume Builder & Enhancer", layout="wide")
    st.title("🧠 Smart Resume Generator & Improver")

    tab1, tab2 = st.tabs(["📝 Create Resume", "🔧 Improve Resume"])

    with tab1:
    
        st.subheader("Step-by-step Resume Builder")

        section = st.radio("Navigate Sections", ["Personal Info", "Experience", "Education", "Skills", "Certifications"])

        if "resume_data" not in st.session_state:
            st.session_state.resume_data = {
                "personal_info": {}, "experience": [], "education": [], "skills": [], "certifications": []
            }

        
        if section == "Personal Info":
            st.subheader("👤 Personal Information")
            voice_summary = voice_input_section("Summary via Voice", "voice_summary")
            with st.form("personal_form"):
                name = st.text_input("Full Name")
                email = st.text_input("Email")
                phone = st.text_input("Phone Number")
                summary = st.text_area("Summary")
                

                if st.form_submit_button("Save"):
                    full_summary = summary + "\n" + voice_summary if voice_summary else summary
                    st.session_state.resume_data["personal_info"] = {
                        "name": name, "email": email, "phone": phone, "summary": full_summary
                    }
                    st.success("✅ Saved Personal Info")

    
        elif section == "Experience":
            st.subheader("💼 Experience")
            voice_desc = voice_input_section("Experience Description via Voice", "voice_exp_desc")
            with st.form("experience_form"):
                job_title = st.text_input("Job Title")
                company = st.text_input("Company Name")
                start_date = st.date_input("Start Date", value=date(2020, 1, 1))
                end_date = st.date_input("End Date", value=date.today())
                description = st.text_area("Job Description")
                

                if st.form_submit_button("Add Experience"):
                    full_desc = description + "\n" + voice_desc if voice_desc else description
                    st.session_state.resume_data["experience"].append({
                        "job_title": job_title,
                        "company": company,
                        "start": str(start_date),
                        "end": str(end_date),
                        "description": full_desc,
                    })
                    st.success("✅ Experience Added")

        elif section == "Education":
            st.subheader("🎓 Education")
            voice_education = voice_input_section("Education via Voice", "voice_education")
            with st.form("education_form"):
                degree = st.text_input("Degree")
                school = st.text_input("School")
                year = st.text_input("Year")
                
                if st.form_submit_button("Add Education"):
                    full_degree = f"{degree} ({voice_education})" if voice_education else degree
                    st.session_state.resume_data["education"].append({
                        "degree": full_degree, "school": school, "year": year
                    })
                    st.success("✅ Education Added")

        elif section == "Skills":
            st.subheader("🛠️ Skills")
            voice_skills = voice_input_section("Skills via Voice", "voice_skills")

            with st.form("skills_form"):
                skills = st.text_area("List your skills")

                if st.form_submit_button("Save"):
                    full_skills = skills + "\n" + voice_skills if voice_skills else skills
                    st.session_state.resume_data["skills"] = full_skills
                    st.success("✅ Saved Skills")


        elif section == "Certifications":
            st.subheader("📜 Certifications")
            voice_certifications = voice_input_section("Certifications via Voice", "voice_certifications")

            with st.form("certifications_form"):
                certifications = st.text_area("List your certifications")

                if st.form_submit_button("Save"):
                    full_certifications = certifications + "\n" + voice_certifications if voice_certifications else certifications
                    st.session_state.resume_data["certifications"] = full_certifications
                    st.success("✅ Saved Certifications")
        # Generate Resume
        elif section == "Preview":
            st.subheader("📝 Resume Preview")
            resume = st.session_state.resume_data

            def edit_button(target_section):
                if st.button(f"✏️ Edit {target_section}", key=f"edit_{target_section}"):
                    st.session_state.current_section = target_section
                    st.experimental_rerun()

            if resume["personal_info"]:
                st.markdown("## 👤 Personal Information")
                st.markdown(f"""
                **Name:** {resume["personal_info"].get("name", "")}  
                **Email:** {resume["personal_info"].get("email", "")}  
                **Phone:** {resume["personal_info"].get("phone", "")}  
                **Summary:** {resume["personal_info"].get("summary", "")}
                """)
                edit_button("Personal Info")

            if resume["experience"]:
                st.markdown("## 💼 Experience")
                for i, exp in enumerate(resume["experience"]):
                    st.markdown(f"""
                    **{exp.get("job_title", "")}** at *{exp.get("company", "")}*  
                    _{exp.get("start_date", "")} – {exp.get("end_date", "")}_  
                    - {exp.get("description", "")}
                    """)
                edit_button("Experience")

            if resume["education"]:
                st.markdown("## 🎓 Education")
                for edu in resume["education"]:
                    st.markdown(f"""
                    **{edu.get("degree", "")}**, *{edu.get("institution", "")}*  
                    _{edu.get("start_year", "")} – {edu.get("end_year", "")}_
                    """)
                edit_button("Education")

            if resume["skills"]:
                st.markdown("## 🧠 Skills")
                st.markdown(", ".join(resume["skills"]))
                edit_button("Skills")

            if resume["certifications"]:
                st.markdown("## 📜 Certifications")
                st.markdown(", ".join(resume["certifications"]))
                edit_button("Certifications")

                

        create_prompt_template = (
            "You are a professional resume writer. Based on the description below, generate a structured and detailed resume with proper sections like Summary, Skills, Experience, Projects, and Education.\n\n"
            "Description:\n{text}\n\n"
            "Return in resume format as plain text."
        )
        
        if st.button("✨ Generate Resume"):
            combined_text = build_user_input(st.session_state.resume_data)
            
            if combined_text.strip():
                with st.spinner("Generating resume..."):
                    resume = summarize_with_gemini(combined_text, create_prompt_template)
                if resume:
                    st.success("✅ Resume Generated")
                    st.markdown(markdown_to_html(resume), unsafe_allow_html=True)

                    docx_path = save_as_docx(resume, "generated_resume.docx")
                    st.download_button("📄 Download as DOCX", data=open(docx_path, "rb"), file_name="generated_resume.docx")
                else:
                    st.error("Resume generation failed.")
            else:
                st.warning("Please fill in the form sections or provide input.")

    with tab2:
        st.subheader("Upload your existing resume and get it improved and more impactful")
        file = st.file_uploader("Upload Resume (PDF, DOCX, TXT, Image)", type=["pdf", "docx", "txt", "jpg", "jpeg", "png"])

        improve_prompt_template = (
            "You are an expert resume editor. Improve and rewrite the resume below to make it more impactful, clean, and suitable for top jobs.\n\n"
            "Resume:\n{text}\n\n"
            "Return the improved resume as plain text."
        )

        if file and st.button("🚀 Improve Resume"):
            with st.spinner("Extracting and improving resume..."):
                raw_text = extract_text_from_file(file)
                if raw_text:
                    improved = summarize_with_gemini(raw_text, improve_prompt_template)
                    if improved:
                        st.success("✅ Improved Resume")
                        st.markdown(markdown_to_html(improved), unsafe_allow_html=True)

                        
                        docx_path = save_as_docx(improved, "improved_resume.docx")
                        # pdf_path = save_as_pdf(improved, "improved_resume.pdf")

                        st.download_button("📄 Download as DOCX", data=open(docx_path, "rb"), file_name="improved_resume.docx")
                        # st.download_button("📄 Download as PDF", data=open(pdf_path, "rb"), file_name="improved_resume.pdf")
                        # st.download_button("📄 Download as TXT", data=improved, file_name="improved_resume.txt")

                    else:
                        st.error("Improvement failed.")
                else:
                    st.error("Could not extract text from the uploaded file.")

if __name__ == "__main__":
    main()

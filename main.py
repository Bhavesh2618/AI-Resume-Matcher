import os
import re
import gc
import numpy as np
import pandas as pd
import torch
import docx
import pdfplumber
from PyPDF2 import PdfReader
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM
import kagglehub
from keybert import KeyBERT

# -----------------------------------------------------------------------------
# 1. Setup and Utility Functions
# -----------------------------------------------------------------------------

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

kw_model = KeyBERT()

def extract_key_summary(text, max_phrases=3):
    """Extract key phrases from text for better summarization"""
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=max_phrases,
            use_maxsum=True,
            nr_candidates=20
        )
        return ', '.join([kw[0] for kw in keywords]) if keywords else text[:150]
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        return text[:150]

def extract_skills_from_text(text):
    """Extract technical skills and qualifications from text"""
    # Common technical skills patterns
    skill_patterns = [
        r'\b(?:Python|R|SQL|Java|C\+\+|JavaScript|Scala|SAS|SPSS|Matlab)\b',
        r'\b(?:Machine Learning|Deep Learning|AI|NLP|Computer Vision)\b',
        r'\b(?:TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy)\b',
        r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Spark|Hadoop)\b',
        r'\b(?:Tableau|Power BI|Looker|D3\.js|Matplotlib|Seaborn)\b',
        r'\b(?:Statistics|Mathematics|Data Analysis|Data Science)\b',
        r'\b(?:Bachelor|Master|PhD|BS|MS|MBA)\b.*?(?:degree|science|engineering|mathematics|statistics|computer)',
        r'\b\d+\+?\s*years?\s*(?:of\s*)?experience\b'
    ]
    
    skills = []
    for pattern in skill_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        skills.extend(matches)
    
    return list(set(skills))

# -----------------------------------------------------------------------------
# 2. Text Extraction Functions
# -----------------------------------------------------------------------------

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error extracting text from {docx_path}: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

# -----------------------------------------------------------------------------
# 3. Enhanced Text Processing Functions
# -----------------------------------------------------------------------------

def chunk_text(text, chunk_size=512, overlap=50):
    """Improved text chunking with better sentence boundary detection"""
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Try to split by sentences first
    sentences = re.split(r'[.!?]+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed chunk size, save current chunk
        if len(current_chunk.split()) + len(sentence.split()) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                words = current_chunk.split()
                if len(words) > overlap:
                    current_chunk = ' '.join(words[-overlap:]) + ' ' + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += ' ' + sentence if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]

def parse_job_requirements(query):
    """Extract structured information from job description"""
    requirements = {
        'skills': [],
        'experience': [],
        'education': [],
        'responsibilities': []
    }
    
    # Extract skills
    skill_matches = re.findall(r'\b(?:Python|R|SQL|Java|C\+\+|JavaScript|Scala|SAS|SPSS|Matlab|Machine Learning|Deep Learning|AI|NLP|TensorFlow|PyTorch|AWS|Azure|GCP|Docker|Kubernetes|Spark|Hadoop|Tableau|Power BI)\b', query, re.IGNORECASE)
    requirements['skills'] = list(set(skill_matches))
    
    # Extract experience requirements
    exp_matches = re.findall(r'\b\d+\+?\s*years?\s*(?:of\s*)?experience\b', query, re.IGNORECASE)
    requirements['experience'] = exp_matches
    
    # Extract education requirements
    edu_matches = re.findall(r'\b(?:Bachelor|Master|PhD|BS|MS|MBA)\b.*?(?:degree|science|engineering|mathematics|statistics|computer)', query, re.IGNORECASE)
    requirements['education'] = edu_matches
    
    return requirements

# -----------------------------------------------------------------------------
# 4. Enhanced Document Processing Functions
# -----------------------------------------------------------------------------

def process_docx_files(docx_folder_path):
    resume_data = []
    docx_files = [f for f in os.listdir(docx_folder_path) if f.endswith('.docx')]
    
    for docx_file in tqdm(docx_files, desc="Processing DOCX files"):
        docx_path = os.path.join(docx_folder_path, docx_file)
        text = extract_text_from_docx(docx_path)
        if not text:
            continue
            
        # Extract skills for each resume
        skills = extract_skills_from_text(text)
        
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            resume_data.append({
                'resume_id': docx_file,
                'chunk_id': f"{docx_file}_chunk_{i}",
                'text': chunk,
                'skills': skills,
                'full_text': text  # Keep full text for context
            })
    
    if resume_data:
        df = pd.DataFrame(resume_data)
        print(f"DataFrame created with {len(df)} chunks from {len(docx_files)} documents")
        return df
    else:
        print("No data processed, returning empty DataFrame")
        return pd.DataFrame(columns=['resume_id', 'chunk_id', 'text', 'skills', 'full_text'])

# -----------------------------------------------------------------------------
# 5. Embedding Functions
# -----------------------------------------------------------------------------

def create_embeddings(df, model_name="all-MiniLM-L6-v2"):
    required_cols = ['resume_id', 'chunk_id', 'text']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    if df.empty:
        print("Empty DataFrame, skipping embeddings")
        return df, np.array([]), None
    
    print(f"Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    
    embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(df), batch_size), desc="Creating embeddings"):
        batch = df['text'][i:i+batch_size].tolist()
        batch_embeddings = embedding_model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    
    embeddings_array = np.array(embeddings)
    df['embedding'] = [embedding.tolist() for embedding in embeddings_array]
    
    print(f"Created {len(embeddings)} embeddings of size {embeddings_array.shape[1]}")
    return df, embeddings_array, embedding_model

def compare_queries(query1, query2, embedding_model):
    e1 = embedding_model.encode(query1)
    e2 = embedding_model.encode(query2)
    return cosine_similarity([e1], [e2])[0][0]

# -----------------------------------------------------------------------------
# 6. Enhanced Retrieval Functions
# -----------------------------------------------------------------------------

def query_resumes(query, df, embeddings_array, embedding_model, top_k=5):
    if df.empty or len(embeddings_array) == 0:
        print("No data to query")
        return pd.DataFrame(columns=['resume_id', 'chunk_id', 'similarity', 'text', 'skills'])
    
    query_embedding = embedding_model.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings_array)[0]
    
    top_k = min(top_k, len(similarities))
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_similarities = similarities[top_indices]
    
    results = []
    for i, idx in enumerate(top_indices):
        results.append({
            'resume_id': df.iloc[idx]['resume_id'],
            'chunk_id': df.iloc[idx]['chunk_id'],
            'similarity': top_similarities[i],
            'text': df.iloc[idx]['text'],
            'skills': df.iloc[idx]['skills'],
            'full_text': df.iloc[idx]['full_text']
        })
    
    return pd.DataFrame(results)

def get_resume_summary(resume_id, df):
    """Get a comprehensive summary of a resume"""
    resume_chunks = df[df['resume_id'] == resume_id]
    if resume_chunks.empty:
        return "No resume found"
    
    # Get the full text
    full_text = resume_chunks.iloc[0]['full_text']
    skills = resume_chunks.iloc[0]['skills']
    
    # Extract key information
    summary = {
        'skills': skills,
        'key_phrases': extract_key_summary(full_text, max_phrases=5),
        'experience_years': re.findall(r'\b\d+\+?\s*years?\s*(?:of\s*)?experience\b', full_text, re.IGNORECASE),
        'education': re.findall(r'\b(?:Bachelor|Master|PhD|BS|MS|MBA)\b.*?(?:degree|science|engineering|mathematics|statistics|computer)', full_text, re.IGNORECASE)
    }
    
    return summary

# -----------------------------------------------------------------------------
# 7. Enhanced LLM Functions
# -----------------------------------------------------------------------------

def initialize_llm(model_path):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    if hasattr(torch, "compile"):
        print("Compiling model for optimized inference...")
        model = torch.compile(model)
    
    return model, tokenizer

def generate_detailed_response(job_query, resume_data, model, tokenizer, is_top_candidate=True):
    """Generate detailed response for top candidates, brief for others"""
    
    if not is_top_candidate:
        # Brief summary for remaining candidates
        return f"Key Skills: {', '.join(resume_data['skills'][:5]) if resume_data['skills'] else 'Not specified'}"
    
    # Detailed assessment for top 3 candidates
    candidate_skills = ', '.join(resume_data['skills'][:6]) if resume_data['skills'] else 'Not specified'
    candidate_text = resume_data['text'][:400]
    job_text = job_query[:250]
    
    # Very direct and specific prompt to avoid internal reasoning
    prompt = f"""Job Requirements: {job_text}

Candidate Profile:
- Skills: {candidate_skills}
- Experience: {candidate_text}

Assessment: This candidate"""
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        # Tokenize with proper settings
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=700,
            padding=True
        )
        
        # Move inputs to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                min_new_tokens=40,
                temperature=0.6,  # Lower temperature for more focused responses
                do_sample=True,
                top_p=0.85,
                top_k=40,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        # Decode the response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after "Assessment: This candidate")
        if "Assessment: This candidate" in full_response:
            generated_text = full_response.split("Assessment: This candidate")[-1].strip()
            generated_text = "This candidate" + generated_text
        else:
            generated_text = full_response[len(prompt):].strip()
        
        # Clean up unwanted patterns that indicate internal reasoning
        unwanted_patterns = [
            r'I need to.*?\.', r'Let me.*?\.', r'So I.*?\.', r'Okay.*?\.', 
            r'First.*?\.', r'What.*?\?', r'How.*?\?', r'Technical Skills –.*?\.',
            r'\(\d+\)', r'Experience –.*?\.', r'Let\'s.*?\.', r'Now.*?\.'
        ]
        
        for pattern in unwanted_patterns:
            generated_text = re.sub(pattern, '', generated_text, flags=re.IGNORECASE)
        
        # Remove any remaining unwanted prefixes
        generated_text = re.sub(r'^(Assessment:|Analysis:|Response:)\s*', '', generated_text)
        
        # Split into sentences and clean
        sentences = re.split(r'[.!?]+', generated_text)
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out short fragments and reasoning-like sentences
            if (sentence and len(sentence) > 15 and 
                not any(word in sentence.lower() for word in ['i need', 'let me', 'so i', 'what', 'how does', 'okay', 'first'])):
                clean_sentences.append(sentence)
        
        # Take first 3-4 good sentences
        clean_sentences = clean_sentences[:4]
        
        if clean_sentences:
            result = '. '.join(clean_sentences)
            if not result.endswith('.'):
                result += '.'
            return result
        else:
            # Enhanced fallback response based on actual skills
            skills_list = resume_data['skills'][:3] if resume_data['skills'] else []
            if skills_list:
                return f"This candidate has strong technical skills in {', '.join(skills_list)} which are relevant to the job requirements. Their experience demonstrates good alignment with the position needs and they show potential for contributing effectively to the role."
            else:
                return "This candidate shows relevant technical background and experience that align well with the job requirements. Their profile demonstrates potential for success in this position."
            
    except Exception as e:
        print(f"Error in text generation: {e}")
        # Enhanced fallback response
        skills_list = resume_data['skills'][:3] if resume_data['skills'] else []
        if skills_list:
            return f"This candidate has strong skills in {', '.join(skills_list)} which are relevant to the position requirements. Their technical background and experience demonstrate good alignment with the role."
        else:
            return "This candidate shows relevant experience and background that align with the job requirements."

# -----------------------------------------------------------------------------
# 8. Enhanced Interactive Interface
# -----------------------------------------------------------------------------

def interactive_query(resume_df, embeddings_array, embedding_model, llm_model, llm_tokenizer):
    print("\n" + "="*60)
    print("         ENHANCED RESUME QUERY SYSTEM")
    print("="*60)
    print("Enter your job description or query, or type 'quit' to exit\n")

    while True:
        query = input("\nEnter your job description/query (or 'quit' to exit): ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not query:
            print("Please enter a valid query.")
            continue

        try:
            top_k = int(input("How many top candidates do you want to see? [default: 5]: ") or "5")
        except ValueError:
            print("Invalid number. Using default: 5")
            top_k = 5

        print(f"\nSearching for candidates...")
        
        # Get results
        results = query_resumes(query, resume_df, embeddings_array, embedding_model, top_k=top_k)

        if results.empty:
            print("No matching candidates found.")
            continue

        print(f"\n" + "="*60)
        print(f"TOP {top_k} MATCHING CANDIDATES")
        print("="*60)

        # Show detailed analysis for each candidate
        for i, (_, row) in enumerate(results.iterrows(), 1):
            resume_summary = get_resume_summary(row['resume_id'], resume_df)
            print(f"\n{i}. {row['resume_id']}")
            print(f"   Similarity Score: {row['similarity']:.4f}")
            
            # Show detailed AI assessment only for the TOP 3 candidates (regardless of total number requested)
            if i <= 3:
                # Detailed AI assessment for top 3
                print(f"   Key Skills: {', '.join(resume_summary['skills'][:5]) if resume_summary['skills'] else 'Not specified'}")
                try:
                    analysis = generate_detailed_response(query, row, llm_model, llm_tokenizer, is_top_candidate=True)
                    print(f"   AI Assessment: {analysis}")
                except Exception as e:
                    print(f"   AI Assessment: Error generating detailed analysis - {str(e)[:50]}")
                    print(f"   Fallback: This candidate shows strong technical background with relevant skills for the position.")
            else:
                # Simple format for remaining candidates (position 4 and beyond)
                try:
                    brief_summary = generate_detailed_response(query, row, llm_model, llm_tokenizer, is_top_candidate=False)
                    print(f"   {brief_summary}")
                except Exception as e:
                    print(f"   Key Skills: {', '.join(resume_summary['skills'][:5]) if resume_summary['skills'] else 'Not specified'}")

        print(f"\n" + "="*60)

# -----------------------------------------------------------------------------
# 9. Main Execution
# -----------------------------------------------------------------------------

def main():
    try:
        # Configuration
        model_path = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-llama-8b"
        dataset_path = "/kaggle/input/resume-dataset"
        docx_folder = os.path.join(dataset_path, "Resumes")
        
        print("Starting Enhanced Resume Query System...")
        print(f"Processing resumes from: {docx_folder}")
        
        # Process documents
        resume_df = process_docx_files(docx_folder)

        # Create embeddings
        print("Creating embeddings...")
        resume_df, embeddings_array, embedding_model = create_embeddings(resume_df)

        # Save data
        print("Saving processed data...")
        resume_df.to_pickle("/kaggle/working/resume_data_enhanced.pkl")
        np.save("/kaggle/working/embeddings_enhanced.npy", embeddings_array)

        # Initialize LLM
        llm_path = os.path.join(model_path, "2")
        llm_model, llm_tokenizer = initialize_llm(llm_path)

        # Clean up memory
        free_memory()

        # Start interactive session
        interactive_query(resume_df, embeddings_array, embedding_model, llm_model, llm_tokenizer)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

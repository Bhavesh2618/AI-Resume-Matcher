# AI Resume Matcher

An intelligent resume screening and candidate matching system powered by advanced NLP and Large Language Models (LLMs). This tool helps recruiters and hiring managers efficiently identify the best candidates by analyzing resumes against job descriptions using semantic similarity and AI-powered assessments.

## 🚀 Features

- **Intelligent Resume Processing**: Extracts and processes text from PDF and DOCX resume files
- **Semantic Search**: Uses sentence transformers to find candidates based on meaning, not just keywords
- **AI-Powered Assessment**: Generates detailed candidate evaluations using LLMs
- **Skill Extraction**: Automatically identifies technical skills, experience levels, and qualifications
- **Interactive Query System**: User-friendly interface for searching and ranking candidates
- **Scalable Architecture**: Handles large volumes of resumes with optimized memory usage

## 🛠️ Technologies Used

- **Python 3.8+**
- **Machine Learning**: SentenceTransformers, Scikit-learn, PyTorch
- **NLP**: KeyBERT for keyword extraction, Custom text processing
- **LLM Integration**: Transformers library with DeepSeek models
- **Document Processing**: PyPDF2, pdfplumber, python-docx
- **Data Processing**: Pandas, NumPy
- **Optimization**: BitsAndBytes for memory efficiency

## 📋 Requirements

```bash
pip install torch transformers sentence-transformers
pip install pandas numpy scikit-learn
pip install python-docx PyPDF2 pdfplumber
pip install keybert bitsandbytes
pip install tqdm kagglehub
```

## 🏃‍♂️ Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Bhavesh2618/ai-resume-matcher.git
   cd ai-resume-matcher
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Place resume files (PDF/DOCX) in a `resumes/` folder
   - Update the `docx_folder` path in the main script

4. **Run the system**
   ```bash
   python main.py
   ```

## 💡 How It Works

1. **Document Processing**: Extracts text from resume files and chunks them for better processing
2. **Embedding Creation**: Generates semantic embeddings using sentence transformers
3. **Similarity Matching**: Compares job descriptions with resume content using cosine similarity
4. **AI Assessment**: Provides detailed candidate evaluations using LLM analysis
5. **Ranking & Results**: Returns top candidates with similarity scores and AI insights

## 🎯 Use Cases

- **Recruitment Agencies**: Streamline candidate screening process
- **HR Departments**: Quickly identify qualified candidates from large applicant pools
- **Hiring Managers**: Get AI-powered insights on candidate fit
- **Job Platforms**: Enhance matching algorithms for better job-candidate pairing

## 📊 Sample Output

```
TOP 2 MATCHING CANDIDATES
============================================================

1. Mohammad Resume.docx
   Similarity Score: 0.4011
   Key Skills: SAS, R, Java, SQL, AWS
   AI Assessment: This candidatehas experience in performing gap analysis and working closely with senior business analysts. However, there's a lack of specific examples or case studies showcasing how they applied these skills in real-world scenarios. Feedback: The candidate is technically skilled but needs more concrete examples from their work history to demonstrate effectiveness.

2. Robinson.docx
   Similarity Score: 0.3980
   Key Skills: data analysis, 10 years of experience, JAVA, SQL
   AI Assessment: This candidatehas over 10+ years in Data Analysis. He is very strong in Java and SQL, which are essential for any data-related work. His experience includes working with large financial institutions, so he should have a good grasp of handling sensitive data. The fact that he interacted with business users and stakeholders indicates he can communicate effectively, which is crucial in collaborative environments.
```

## 🔧 Configuration

Key configuration options in the main script:

- `model_name`: Choose embedding model (default: "all-MiniLM-L6-v2")
- `chunk_size`: Text chunk size for processing (default: 512)
- `top_k`: Number of candidates to return (default: 5)
- `llm_model_path`: Path to your LLM model

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SentenceTransformers for semantic similarity
- Transformers library for LLM integration
- KeyBERT for keyword extraction
- The open-source community for various supporting libraries

## 📞 Contact

Mail ID - pidugubhaveshkumar@gmail.com

Project Link: [https://github.com/Bhavesh2618/ai-resume-matcher](https://github.com/Bhavesh2618/ai-resume-matcher)

---

⭐ **Star this repository if you find it helpful!**

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

1. Naren Sr.BA.docx
   Similarity Score: 0.5356
   Key Skills: Java, SQL, Power BI, 9+ years of experience, data analysis
   AI Assessment: This candidatehas strong technical skills with a solid background in data engineering. They have extensive experience with data analysis and ETL processes, which is crucial for building efficient data pipelines. Their ability to work with databases and write complex SQL queries will be beneficial in optimizing data retrieval and performance. Additionally, their proficiency with tools like Power BI and Tableau suggests they can effectively visualize and communicate data insights, which are important traits for any data engineer.

2. jane_smith_resume.docx
   Similarity Score: 0.8231
   Key Skills: Data Science, R, Statistics, Tableau, SQL
   AI Assessment: This candidate shows excellent analytical skills and 
   data visualization experience, making them a strong fit for 
   data-driven roles.
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

Your Name - pidugubhaveshkumar@gmail.com

Project Link: [https://github.com/Bhavesh2618/ai-resume-matcher](https://github.com/Bhavesh2618/ai-resume-matcher)

---

⭐ **Star this repository if you find it helpful!**

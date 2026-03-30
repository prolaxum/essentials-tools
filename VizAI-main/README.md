# VizAI - AI-Powered Business Data Visualization and Interactive Analytics Tool

An interactive web application that bridges the gap between complex data science and business intelligence. Built for non-technical users to extract actionable insights from raw data files through automated profiling, custom Plotly visualizations, and a conversational AI interface.

## 🚀 Features
* **Automated Data Profiling:** Instant generation of summary statistics, data type separation, and missing value detection.
* **Smart Data Cleaning:** Automatic handling of blank rows and imputation of missing values (mean for numeric, 'Unknown' for categorical).
* **Interactive Visualizations:** Manual routing for dynamic Plotly bar charts, line graphs, and scatter plots.
* **Conversational AI:** Natural language querying powered by Google Gemini 2.5 Flash and PandasAI to generate Python code, execute it against the dataset, and return text or chart responses.

## 🛠️ Technology Stack
* **Frontend/Framework:** Streamlit
* **Data Processing:** Pandas
* **Visualization:** Plotly Express
* **AI/LLM:** Google Gemini 2.5 Flash (via LiteLLM)
* **AI Agent:** PandasAI

## ⚙️ Local Setup Instructions

1. **Clone the repository:**
   git clone <your-github-repo-url>
   cd <your-repo-name>
   
Create and activate a virtual environment (Python 3.11 recommended):
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt
Environment Variables:

Create a .env file in the root directory and add your Google Gemini API Key:
GEMINI_API_KEY="your_api_key_here"

Run the application:
streamlit run app.py

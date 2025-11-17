import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import docx

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

# -------- Monster scraper --------
def scrape_monster_jobs(query='data scientist'):
    base_url = f"https://www.monster.com/jobs/search/?q={query.replace(' ', '-')}&page="
    job_list = []

    for page in range(1, 3):
        url = base_url + str(page)
        res = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(res.text, 'html.parser')
        cards = soup.find_all('section', class_='card-content')

        for card in cards:
            title_tag = card.find('h2', class_='title')
            company_tag = card.find('div', class_='company')
            location_tag = card.find('div', class_='location')
            date_tag = card.find('time')
            link_tag = title_tag.find('a', href=True) if title_tag else None

            title = title_tag.text.strip() if title_tag else "N/A"
            company = company_tag.text.strip() if company_tag else "N/A"
            location = location_tag.text.strip() if location_tag else "N/A"
            date = date_tag.text.strip() if date_tag else "N/A"
            link = link_tag['href'] if link_tag else ""

            job_list.append({
                "Title": title,
                "Company": company,
                "Location": location,
                "Description": title,
                "DatePosted": date,
                "SkillLevel": "Mid",
                "Link": link,
                "Source": "Monster"
            })
    return pd.DataFrame(job_list)

# -------- Apna scraper --------
def scrape_apna_jobs(query='data scientist'):
    base_url = f"https://apna.co/jobs?q={query.replace(' ', '%20')}&page="
    job_list = []

    for page in range(1, 3):
        url = base_url + str(page)
        res = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(res.text, 'html.parser')

        # Apna uses script tags to load jobs; we will look for job cards
        jobs = soup.find_all('a', class_='job-card')  # may need update if Apna changed structure

        for job in jobs:
            title_tag = job.find('h3', class_='job-title')
            company_tag = job.find('p', class_='company-name')
            location_tag = job.find('span', class_='location')
            # Date and skilllevel are not always available here
            title = title_tag.text.strip() if title_tag else "N/A"
            company = company_tag.text.strip() if company_tag else "N/A"
            location = location_tag.text.strip() if location_tag else "N/A"
            date = "N/A"
            skill_level = "Mid"
            link = "https://apna.co" + job['href'] if job.has_attr('href') else ""

            job_list.append({
                "Title": title,
                "Company": company,
                "Location": location,
                "Description": title,
                "DatePosted": date,
                "SkillLevel": skill_level,
                "Link": link,
                "Source": "Apna"
            })
    return pd.DataFrame(job_list)

# -------- Cuvette scraper (Example) --------
def scrape_cuvette_jobs(query='data scientist'):
    base_url = f"https://cuvette.io/jobs?q={query.replace(' ', '+')}&page="
    job_list = []

    for page in range(1, 3):
        url = base_url + str(page)
        res = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(res.text, 'html.parser')

        jobs = soup.find_all('a', class_='job-listing-link')  # hypothetical class

        for job in jobs:
            title_tag = job.find('h2')
            company_tag = job.find('div', class_='company-name')
            location_tag = job.find('div', class_='job-location')

            title = title_tag.text.strip() if title_tag else "N/A"
            company = company_tag.text.strip() if company_tag else "N/A"
            location = location_tag.text.strip() if location_tag else "N/A"
            date = "N/A"
            skill_level = "Mid"
            link = "https://cuvette.io" + job['href'] if job.has_attr('href') else ""

            job_list.append({
                "Title": title,
                "Company": company,
                "Location": location,
                "Description": title,
                "DatePosted": date,
                "SkillLevel": skill_level,
                "Link": link,
                "Source": "Cuvette"
            })
    return pd.DataFrame(job_list)

# -------- Indeed scraper --------
def scrape_indeed_jobs(query='data scientist'):
    base_url = "https://www.indeed.com/jobs"
    job_list = []

    for start in [0, 10]:  # 2 pages
        params = {'q': query, 'start': start}
        res = requests.get(base_url, params=params, headers=HEADERS)
        soup = BeautifulSoup(res.text, 'html.parser')
        cards = soup.find_all('div', class_='job_seen_beacon')

        for card in cards:
            title_tag = card.find('h2', class_='jobTitle')
            company_tag = card.find('span', class_='companyName')
            location_tag = card.find('div', class_='companyLocation')
            date_tag = card.find('span', class_='date')
            link_tag = title_tag.find('a', href=True) if title_tag else None

            title = title_tag.text.strip() if title_tag else "N/A"
            company = company_tag.text.strip() if company_tag else "N/A"
            location = location_tag.text.strip() if location_tag else "N/A"
            date = date_tag.text.strip() if date_tag else "N/A"
            link = "https://www.indeed.com" + link_tag['href'] if link_tag else ""

            job_list.append({
                "Title": title,
                "Company": company,
                "Location": location,
                "Description": title,
                "DatePosted": date,
                "SkillLevel": "Mid",
                "Link": link,
                "Source": "Indeed"
            })
    return pd.DataFrame(job_list)

# -------- Load or scrape data --------
def load_jobs_data(query='data scientist'):
    try:
        df = pd.read_csv('data/jobs.csv')
        if df.empty:
            raise Exception("Empty CSV")
    except Exception:
        # Scrape from all sources and combine
        df1 = scrape_monster_jobs(query)
        df2 = scrape_apna_jobs(query)
        df3 = scrape_cuvette_jobs(query)
        df4 = scrape_indeed_jobs(query)

        df = pd.concat([df1, df2, df3, df4], ignore_index=True)

        os.makedirs('data', exist_ok=True)
        df.to_csv('data/jobs.csv', index=False)
    return df

df = load_jobs_data()

for col in ['Title', 'Company', 'Location', 'Description', 'DatePosted', 'SkillLevel', 'Link', 'Source']:
    if col not in df.columns:
        df[col] = 'N/A'

df['Skills'] = df['Title'].str.lower() + ' ' + df['Description'].str.lower()

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Skills'])

def get_job_recommendations(user_input, top_n=5):
    user_input = user_input.lower()
    user_vec = tfidf.transform([user_input])
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    matched_df = df.iloc[top_indices][cosine_sim[top_indices] > 0]

    return matched_df[['Title', 'Company', 'Location', 'Description', 'DatePosted', 'SkillLevel', 'Link', 'Source']].to_dict(orient='records')

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return " ".join([para.text for para in doc.paragraphs])

def extract_skills(text):
    text = text.lower()
    possible_skills = ['python', 'java', 'c++', 'machine learning', 'data analysis', 'django', 'flask', 'sql', 'tensorflow',
                       'keras', 'pandas', 'numpy', 'react', 'aws', 'api', 'backend', 'frontend', 'docker', 'kubernetes']
    return " ".join([skill for skill in possible_skills if skill in text])

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    if request.method == 'POST':
        job_title = request.form.get('job_title', '').strip()
        file = request.files.get('resume')

        if file and file.filename != '':
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            if filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(filename)
            elif filename.lower().endswith('.docx'):
                text = extract_text_from_docx(filename)
            else:
                error = "Unsupported file type. Please upload PDF or DOCX."
                return render_template('index.html', error=error)

            user_skills = extract_skills(text)
            recommendations = get_job_recommendations(user_skills)
            if not recommendations:
                message = "No matching jobs found for your resume skills."
                return render_template('result.html', job_title="Your Resume Skills", message=message)
            return render_template('result.html', recommendations=recommendations, job_title="Your Resume Skills")

        elif job_title:
            recommendations = get_job_recommendations(job_title)
            if not recommendations:
                message = f"No matching jobs found for '{job_title}'."
                return render_template('result.html', job_title=job_title, message=message)
            return render_template('result.html', recommendations=recommendations, job_title=job_title)

        else:
            error = "Please enter skills/job title or upload a resume."

    return render_template('index.html', error=error)
@app.route('/apply')
def apply():
    return render_template('apply.html')

@app.route('/submit-application', methods=['POST'])
def submit_application():
    name = request.form.get('name')
    email = request.form.get('email')
    resume = request.files.get('resume')

    if not name or not email or not resume:
        return "All fields are required.", 400

    # Save resume
    filename = os.path.join(app.config['UPLOAD_FOLDER'], resume.filename)
    resume.save(filename)

    print(f"Received application from {name} ({email}), Resume saved to {filename}")

    return render_template('success.html', name=name)


if __name__ == '__main__':
    app.run(debug=False)

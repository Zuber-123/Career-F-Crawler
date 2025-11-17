import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load jobs dataset
df = pd.read_csv('data/jobs.csv')
df['Skills'] = df['Skills'].str.lower()
df['Title'] = df['Title'].str.lower()

# TF-IDF vectorizer on skills
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Skills'])

def get_job_recommendations(user_input, top_n=10, location=None, company=None, skill_level=None, sort_by=None):
    user_input = user_input.lower()
    user_vec = tfidf.transform([user_input])
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()

    df['Similarity'] = cosine_sim

    # Filter & sort
    filtered_df = df.copy()
    if location:
        filtered_df = filtered_df[filtered_df['Location'].str.contains(location, case=False, na=False)]
    if company:
        filtered_df = filtered_df[filtered_df['Company'].str.contains(company, case=False, na=False)]
    if skill_level:
        filtered_df = filtered_df[filtered_df['SkillLevel'].str.contains(skill_level, case=False, na=False)]

    if sort_by == 'relevance':
        filtered_df = filtered_df.sort_values(by='Similarity', ascending=False)
    elif sort_by == 'date':
        filtered_df = filtered_df.sort_values(by='DatePosted', ascending=False)
    elif sort_by == 'company':
        filtered_df = filtered_df.sort_values(by='Company')
    else:
        filtered_df = filtered_df.sort_values(by='Similarity', ascending=False)

    # Fix links in filtered jobs (ensure full URL)
    filtered_df = filtered_df.copy()
    filtered_df['Link'] = filtered_df['Link'].apply(lambda x: x if x.startswith(('http://', 'https://')) else ('https://www.naukri.com' + x))

    # Take top matches by skill similarity
    top_jobs = filtered_df[filtered_df['Similarity'] > 0].head(top_n)

    # Fallback if no skill match: use job title keyword match
    if top_jobs.empty:
        title_matches = df[df['Title'].str.contains(user_input, case=False, na=False)]
        top_jobs = title_matches.head(top_n)

    return top_jobs[['Title', 'Company', 'Location', 'Description', 'DatePosted', 'SkillLevel', 'Link']].to_dict(orient='records')

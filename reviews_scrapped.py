import openai
from openai import OpenAI
import tiktoken
import re
import pandas as pd
import uuid
from datetime import datetime
from google.cloud import bigquery
from google_play_scraper import Sort,reviews
from datetime import datetime
import os
import json

#big query id, dataset and table for upload purposes
project_id = ''
dataset_name = "play_store_reviews"
table_name = "reviews"

#Key from OpenAI developer account
client = OpenAI(api_key="")

# packages = ['com.android.google','com.brave.browser','com.opera.browser', 'org.mozilla.firefox']
package = 'com.microsoft.emmx'
#To upload to big query, no other reason
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ""
checkpoint_file = 'edge_reviews.json'
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as file:
        checkpoint_data = json.load(file)
else:
    checkpoint_data = {package: []}

def load_checkpoint(file_path="edge_reviews.json"):
    try:
        with open(file_path, "r") as file:
            print('here')
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return an empty dictionary if the file doesn't exist or can't be parsed
        return {}

# Save updated checkpoint data
def save_checkpoint(package, new_data, file_path="checkpoint.json"):
    checkpoint_data = load_checkpoint(file_path)  # Load existing data
    if package not in checkpoint_data:
        checkpoint_data[package] = []  # Initialize if not present

    # Extend existing data with new batch analyses
    checkpoint_data[package].extend(new_data)

    # Save the updated data back to the file
    with open(file_path, "w") as file:
        json.dump(checkpoint_data, file)


def fetch_latest_timestamp(project_id, dataset_id, table_id):
    client = bigquery.Client(project=project_id)
    query = f"""
    SELECT MAX(Timestamp) AS latest_timestamp
    FROM `{project_id}.{dataset_id}.{table_id}`
    """
    query_job = client.query(query)
    result = query_job.result()
    latest_timestamp = None
    for row in result:
        latest_timestamp = row.latest_timestamp
    return latest_timestamp

def filter_new_reviews(analyses, latest_timestamp):
    new_reviews = []
    for analysis in analyses:
        review_timestamp = analysis.get('Timestamp', datetime.utcnow())
        if review_timestamp > latest_timestamp:
            new_reviews.append(analysis)
    return new_reviews

def upload_to_bigquery(df, project_id, dataset_id, table_id):
    client = bigquery.Client(project=project_id)
    table_ref = client.dataset(dataset_id).table(table_id)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    client.load_table_from_dataframe(df, table_ref, job_config=job_config).result()
    print(f"Uploaded {len(df)} rows to BigQuery.")

def process_reviews_and_upload(analyses, package, project_id, dataset_id, table_id):
    latest_timestamp = fetch_latest_timestamp(project_id, dataset_id, table_id)
    if not latest_timestamp:
        latest_timestamp = datetime.min

    new_reviews = filter_new_reviews(analyses, latest_timestamp)

    data = []
    for analysis in analyses:
      if analysis.get('Understandable') is None:
          understandable = 'N/A'
      else:
          understandable = analysis.get('Understandable', 'N/A').strip()
      if analysis.get('Sentiment') is None:
          sentiment = 'Neutral'
      else:
          sentiment = analysis.get('Sentiment', 'Neutral').strip()
      if analysis.get('Key Issues') is None:
          key_issues = 'Unknown Issue'
      else:
          key_issues = analysis.get('Key Issues', 'Unknown Issue').strip()
      if analysis.get('Analysis') is None:
          understandable = 'No analysis provided'
      else:
           gpt_analysis = analysis.get('Analysis', 'No analysis provided').strip()
      if analysis.get('Review') is None:
          review_text = 'No Review Text'
      else:
          review_text = analysis.get('Review', 'No Review Text').strip()
      if analysis.get('Date') is None:
          date_str = None
      else:
          date_str = analysis.get('Date', None)

      # Strip extra space
      understandable = understandable.strip().lower()
      sentiment = sentiment.strip()
      key_issues = key_issues.strip()
      gpt_analysis = gpt_analysis.strip()
      date_str = date_str.strip()
      review_text = review_text.strip()

      print(analysis)
      if date_str:
          try:
              date = pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
          except ValueError:
              print(f"Error parsing date: {date_str}")
              date = None
      else:
          date = None

      if understandable.lower() == "yes":
          data.append([review_text, date, sentiment, gpt_analysis, key_issues, package])

    df = pd.DataFrame(data, columns=['Review', 'Timestamp', 'Sentiment', 'Analysis', 'Key Issues', 'Package'])

    df['Review'] = df['Review'].astype(str)
    df['Sentiment'] = df['Sentiment'].astype(str)
    df['Analysis'] = df['Analysis'].astype(str)
    df['Key Issues'] = df['Key Issues'].astype(str)
    df['Package'] = df['Package'].astype(str)


    if not df.empty:
        upload_to_bigquery(df, project_id, dataset_id, table_id)
    else:
        print("No new reviews to upload; all data is duplicated or outdated.")

# Estimate token based on text length since gpt has a limit
def estimate_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    num_tokens = len(encoding.encode(text))
    return num_tokens

# Need full review summary for overall analysis to get a decent understanding of our overall
def get_full_review_summary(reviews):
    combined_reviews = "\n".join([f"Review {i+1}: {review['content']}" for i, review in enumerate(reviews)])
    prompt = (
        "Based on the following reviews, provide a summary of the key issues, positive highlights, "
        "and overall sentiment trends. Summarize any common complaints or praises and suggest areas "
        "for improvement:\n\n" + combined_reviews
    )

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes reviews."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-4o-mini",
            timeout=480 #Longer timeout needed for longer text otherwise we just get server error
        )
    except Exception as e:
        print(f"Error generating overall summary: {e}")
        return "Overall summary could not be generated due to an error."

    return response.choices[0].message.content

def batch_reviews(reviews, max_tokens=4096, system_message_overhead=150):
    effective_max_tokens = max_tokens - system_message_overhead
    batches = []
    current_batch = []
    current_tokens = 0

    #Have to do reviews in batch since there are too many
    for review in reviews:
        if review.get('content') is None:
            continue
        review_text = "Review: " + review.get('content', '')
        review_date = review.get('at')
        review_score = review.get('score')
        review_text += f"\n{review_date}\n{review_score}"
        if not isinstance(review_text, str):
            continue
        review_tokens = estimate_tokens(review_text)
        if current_tokens + review_tokens > effective_max_tokens:
            if current_batch:
                batches.append(current_batch)

            current_batch = [review_text]
            current_tokens = review_tokens
        else:
            current_batch.append(review_text)
            current_tokens += review_tokens


    if current_batch:
        batches.append(current_batch)

    return batches


def analyze_batch(reviews_batch, max_retries=3, initial_delay=5):
    all_results = []
    prompt = "\n\n".join([f"Review {i+1}: {review}" for i, review in enumerate(reviews_batch)])
    prompt += (
        "Analyze the above customer reviews and provide a structured analysis for each review. "
        f"Return ALL of the {len(reviews_batch)} reviews given to you."
        "Please note: UNDERSTANDABLE is a field that should denote that a review could be translated from english to another language and still make sense. The review must be words FROM THE ENGLISH LANGUAGE. IT CANNOT BE JUST RANDOM LETTERS, OR JUST RANDOM NAMES."
        "For each review, include the following format:\n\n"
        "Understandable: Yes / No -> indicates whether or not the comment makes sense in English and therefore can be found in the ENGLISH DICTIONARY.  E,g: you have makred 'Abdul Qadir Lashari' as understandable. THAT IS NOT UNDERSTANDABLE. IT IS NOT AN ENGLISH WORD.\n"
        "Review: The review comment\n"
        "Date: The date passed to you"
        "Score: The score passed to you as an integer"
        "Sentiment: [Sentiment - Clearly state if the review is Positive or Negative. Make sure to say its either or. If the review seems to be 'in between' but has somewhat of a negative sense, then put negative. If you can't decide, output N/A]\n"
        "Key Issues: [List any issues such as battery, performance, login problems]\n"
        "Category of Issues: [List the category from one or more of the following: Performance, Accessibility, UI design, Privacy, Usability, Other] -- Only list this if negative issues are found.\n"
        "Analysis: [Detailed summary of the review content] -- Only do this for long comments that contain detailed information, otherwise state 'N/A'.\n"
        "Make sure that EACH review is separated by two new lines character for easy parsing and NO special charaters. This means: \\n\\n between every review and no ** or ---- between lines. Easy parsing of each review is a MUST . \n\n"
    )

    retries = 0
    delay = initial_delay
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                messages=[{"role": "system", "content": "You are an assistant that analyzes customer reviews."},
                            {"role": "user", "content": prompt}],
                model="gpt-4o-mini",
                timeout=480
            )
            response_content = response.choices[0].message.content
            all_results.append(response_content)
            break
        except Exception as e:
            print(f"Error during OpenAI request: {e}")
            retries += 1
            print(f"Retrying... ({retries}/{max_retries}) in {delay} seconds.")
            time.sleep(delay)
            delay *= 2

        if retries == max_retries:
            print(f"Failed to get response after {max_retries} retries. Skipping this batch.")
            continue
    return all_results



def remove_words_before_first_keyword(text, keyword):
    keyword = keyword.strip().lower()
    position = text.lower().find(keyword)

    if position != -1:
        return text[position:]
    else:
        return text

def clean_line(line):
    #Keep colons, hyphens, and spaces since reviews contain those but other characters we can ommit
    cleaned_line = re.sub(r'[^\w\s:.-]', '', line)
    #Some reviews have multiple spaces and it throws the model off at time, try to normalize it to one space
    cleaned_line = re.sub(r' +', ' ', cleaned_line)
    return cleaned_line

def should_ignore_line(line):
    if line.strip() == "---":
      return True
    return line.strip() == "" or line.strip() == "---" or not re.search(r'^(Understandable|Review|Sentiment|Key Issues|Category of Issues|Analysis):', line, re.IGNORECASE)



def parse_chat_completion_response(response_content):
    reviews = re.split('\n\n', response_content)
    parsed_reviews=[]

    #GPT pattern response should contain all of these "sections"
    understandable_pattern = re.compile(r'Understandable:\s*(.*?)(?=\n|$)', re.IGNORECASE)
    review_text_pattern = re.compile(r'Review:\s*(.*?)(?=\n|$)', re.DOTALL | re.IGNORECASE)
    score_pattern = re.compile(r'Score:\s*(.*?)(?=\n|$)', re.DOTALL | re.IGNORECASE)
    sentiment_pattern = re.compile(r'Sentiment:\s*(.*?)(?=\n|$)', re.IGNORECASE)
    key_issues_pattern = re.compile(r'Key Issues:\s*(.*?)(?=\n|$)', re.DOTALL | re.IGNORECASE)
    category_pattern = re.compile(r'Category of Issues:\s*(.*?)(?=\n|$)', re.DOTALL | re.IGNORECASE)
    analysis_pattern = re.compile(r'Analysis:\s*(.*?)(?=\n|$)', re.DOTALL | re.IGNORECASE)
    date_pattern = re.compile(r'Date:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', re.IGNORECASE)

    if reviews[0].lower().startswith("understandable") is False:
      reviews = reviews[1:]
    for review in reviews:
        date_match = date_pattern.search(review)
        understandable_match = understandable_pattern.search(review)
        review_match = review_text_pattern.search(review)
        score_match = score_pattern.search(review)
        sentiment_match = sentiment_pattern.search(review)
        key_issues_match = key_issues_pattern.search(review)
        category_match = category_pattern.search(review)
        analysis_match = analysis_pattern.search(review)
        temp = { "Review": review_match.group(1) if review_match else None,
                 "Date": date_match.group(1) if date_match else None,
                 "Score": score_match.group(1) if score_match else -1,
                                    "Understandable": understandable_match.group(1) if understandable_match else None,
                                    "Sentiment": sentiment_match.group(1) if sentiment_match else None,
                                    "Key Issues": key_issues_match.group(1) if key_issues_match else None,
                                    "Category of Issues": category_match.group(1) if category_match else None,
                                    "Analysis": analysis_match.group(1) if analysis_match else None,
                 }
        parsed_reviews.append(temp)
    return parsed_reviews



#Sort can be sorted by score, date, etc... We just take the top X we want to analyze.
all_reviews = {}
print(f"Fetching reviews for {package}")
result, _ = reviews(package, sort=Sort.NEWEST, count=20000)
all_analyses = []


review_batches = batch_reviews(result, max_tokens=4000)
print(f"Number of batches created: {len(review_batches)}")

for i, batch in enumerate(review_batches):
    print(f"Analyzing batch {i+1}/{len(review_batches)} for package {package}...")
    print(f'current batch of size {len(batch)}')
    batch_analyses = analyze_batch(batch)
    all_analyses.extend(batch_analyses)
    save_checkpoint(package, batch_analyses, "edge_reviews.json")


# for package_name, package_reviews in all_analyses.items():
#     combined_reviews = ''.join(package_reviews)
#     cleaned_reviews = remove_words_before_first_keyword(combined_reviews, "Understandable")
#     parsed_text = parse_chat_completion_response(cleaned_reviews)
#     process_reviews_and_upload(parsed_text, package_name, project_id, dataset_name, table_name)

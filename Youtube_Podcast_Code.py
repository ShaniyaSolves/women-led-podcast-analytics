#!/usr/bin/env python
# coding: utf-8

# # 1.  Imports & Set Up 

# In[ ]:


import pandas as pd
from datetime import datetime, timezone
import isodate
import re
import numpy as np
from scipy.stats import mannwhitneyu

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pydantic import BaseModel, Field
from typing import Literal


get_ipython().run_line_magic('pip', 'install -q -U google-genai')
import getpass
from google import genai

get_ipython().run_line_magic('pip', 'install -q google-api-python-client pandas isodate')
import os
from googleapiclient.discovery import build


# In[1]:


os.environ["YOUTUBE_API_KEY"] = " INSERT KEY "

youtube = build(
    "youtube",
    "v3",
    developerKey=os.environ["YOUTUBE_API_KEY"]
)

response = youtube.channels().list(
    part="snippet,statistics,contentDetails",
    forHandle="@ABETTERYOUBYFERNANDA"
).execute()

print(response["items"][0]["snippet"]["title"])
print(response["items"][0]["statistics"])


# In[2]:


#Insert Key
gemini_key = getpass.getpass("Gemini API key: ")

gemini = genai.Client(api_key=gemini_key)

response = gemini.models.generate_content(
    model="gemini-3.6-flash",
    contents="Reply with exactly: Gemini API works"
)

print(response.text)


# # 2. Import Youtube Channels

# In[3]:


def collect_channel_videos(youtube, handle, max_videos=30):
    # 1. Find the channel and its uploads playlist
    channel_response = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        forHandle=handle
    ).execute()
    if not channel_response["items"]:
        raise ValueError(f"No channel found for {handle}")
    channel = channel_response["items"][0]
    channel_name = channel["snippet"]["title"]
    uploads_playlist = channel["contentDetails"]["relatedPlaylists"]["uploads"]

    # 2. Get video IDs from the uploads playlist
    video_ids = []
    page_token = None
    while len(video_ids) < max_videos:
        playlist_response = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist,
            maxResults=min(50, max_videos - len(video_ids)),
            pageToken=page_token
        ).execute()
        video_ids.extend(
            item["contentDetails"]["videoId"]
            for item in playlist_response["items"]
        )
        page_token = playlist_response.get("nextPageToken")
        if not page_token:
            break

    # 3. Retrieve metadata and performance statistics
    rows = []
    now = datetime.now(timezone.utc)
    for start in range(0, len(video_ids), 50):
        batch = video_ids[start:start + 50]
        response = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch)
        ).execute()
        for video in response["items"]:
            snippet = video["snippet"]
            stats = video.get("statistics", {})
            published = datetime.fromisoformat(
                snippet["publishedAt"].replace("Z", "+00:00")
            )
            age_days = max((now - published).total_seconds() / 86400, 1)
            views = int(stats["viewCount"]) if "viewCount" in stats else None
            likes = int(stats["likeCount"]) if "likeCount" in stats else None
            comments = int(stats["commentCount"]) if "commentCount" in stats else None
            rows.append({
                "channel": channel_name,
                "video_id": video["id"],
                "title": snippet["title"],
                "published_at": published,
                "duration_minutes": round(
                    isodate.parse_duration(
                        video["contentDetails"]["duration"]
                    ).total_seconds() / 60,
                    2
                ),
                "views": views,
                "likes": likes,
                "comments": comments,
                "age_days": round(age_days, 1),
                "views_per_day": round(views / age_days, 2) if views is not None else None,
                "likes_per_1000_views": (
                    round(1000 * likes / views, 2)
                    if views and likes is not None else None
                ),
                "comments_per_1000_views": (
                    round(1000 * comments / views, 2)
                    if views and comments is not None else None
                )
            })
    return pd.DataFrame(rows).sort_values(
        "published_at",
        ascending=False
    ).reset_index(drop=True)


# In[4]:


def add_video_descriptions(youtube, dataframe):
    """Add public YouTube descriptions to an existing episode dataframe."""
    description_map = {}
    video_ids = dataframe["video_id"].tolist()
    for start in range(0, len(video_ids), 50):
        response = youtube.videos().list(
            part="snippet",
            id=",".join(video_ids[start:start + 50])
        ).execute()
        for video in response.get("items", []):
            description_map[video["id"]] = (
                video["snippet"].get("description", "")
            )
    result = dataframe.copy()
    result["description"] = (
        result["video_id"]
        .map(description_map)
        .fillna("")
    )
    # Limit description length so sponsorship text and links do not dominate.
    result["classification_text"] = (
        "TITLE:\n"
        + result["title"].fillna("")
        + "\n\nDESCRIPTION:\n"
        + result["description"].fillna("").str[:2000]
    )
    return result


# # 3. Call Gemini

# In[5]:


MODEL_NAME = "gemini-3.5-flash-lite"

def call_gemini_structured(prompt, schema, max_attempts=6):
    for attempt in range(max_attempts):
        try:
            response = gemini.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema
                )
            )
            return schema.model_validate_json(response.text)
        except (ServerError, ClientError) as error:
            if isinstance(error, ClientError) and error.status_code != 429:
                raise
            if attempt == max_attempts - 1:
                raise
            wait_seconds = min(
                (2 ** attempt) + random.uniform(0, 1),
                30
            )
            print(
                f"Gemini temporarily unavailable; "
                f"retrying in {wait_seconds:.1f} seconds..."
            )
            time.sleep(wait_seconds)


# In[6]:


from pydantic import BaseModel

class TopicCategory(BaseModel):
    name: str
    definition: str

class TopicSet(BaseModel):
    categories: list[TopicCategory]


# # 4. Feature Extraction 

# In[7]:


CHANNELS = {
    "A Better You Podcast": "@ABETTERYOUBYFERNANDA",
    "A Really Good Cry": "@AReallyGoodCry",
    "The Chic Code": "@TheChicCodePodcast",
    "Sis, Be Real": "@SisBeRealPodcast",
    "Alex & Annie": "@alexannie",
    "Good Hang with Amy Poehler": "@Good-Hang-with-Amy-Poehler",
    "Baby, This Is Keke Palmer": "@BabythisisKekePalmer",
    "Rotten Mango": "@rottenmangopod",
}


# In[8]:


import os
import random
import re
import time

import pandas as pd
from googleapiclient.discovery import build


TARGET_EPISODES = 50
UPLOADS_TO_CHECK = 200
MAX_ATTEMPTS = 6


exclude_pattern = (
    r"(?i)(?:#shorts?\b|#podcastclips?\b|\bpodcast clip\b|"
    r"\bshort clip\b|\btrailer\b|\bteaser\b|\bpromo\b)"
)


def rebuild_youtube_client():
    return build(
        "youtube",
        "v3",
        developerKey=os.environ["YOUTUBE_API_KEY"],
        cache_discovery=False
    )


def collect_channel_with_retries(
    channel_name,
    handle,
    max_videos,
    max_attempts=MAX_ATTEMPTS
):
    global youtube

    for attempt in range(max_attempts):
        try:
            channel_df = collect_channel_videos(
                youtube,
                handle=handle,
                max_videos=max_videos
            )

            print(
                f"Collected {len(channel_df)} uploads "
                f"from {channel_name}"
            )

            return channel_df

        except ValueError:
            # Do not retry an invalid channel handle
            raise

        except Exception as error:
            if attempt == max_attempts - 1:
                raise

            wait_seconds = min(
                (2 ** attempt) + random.uniform(0, 2),
                30
            )

            print(
                f"{channel_name} failed with "
                f"{type(error).__name__}; retrying in "
                f"{wait_seconds:.1f} seconds..."
            )

            time.sleep(wait_seconds)

            # Create a fresh connection after SSL/network failures
            youtube = rebuild_youtube_client()


def add_descriptions_with_retries(
    dataframe,
    max_attempts=MAX_ATTEMPTS
):
    global youtube

    for attempt in range(max_attempts):
        try:
            return add_video_descriptions(
                youtube,
                dataframe
            )

        except Exception as error:
            if attempt == max_attempts - 1:
                raise

            wait_seconds = min(
                (2 ** attempt) + random.uniform(0, 2),
                30
            )

            print(
                f"Description request failed with "
                f"{type(error).__name__}; retrying in "
                f"{wait_seconds:.1f} seconds..."
            )

            time.sleep(wait_seconds)
            youtube = rebuild_youtube_client()


# Start with a fresh YouTube client
youtube = rebuild_youtube_client()

expanded_data = []

for requested_channel, handle in CHANNELS.items():
    channel_df = collect_channel_with_retries(
        channel_name=requested_channel,
        handle=handle,
        max_videos=UPLOADS_TO_CHECK
    )

    channel_df["requested_channel"] = requested_channel
    channel_df["channel_handle"] = handle

    expanded_data.append(channel_df)

    # Brief pause between channels
    time.sleep(1)


expanded_raw = pd.concat(
    expanded_data,
    ignore_index=True
)

# Add descriptions and classification text
expanded_videos = add_descriptions_with_retries(
    expanded_raw
)

# Remove likely clips and non-episodes
expanded_videos["excluded_reason"] = ""

expanded_videos.loc[
    expanded_videos["duration_minutes"] < 10,
    "excluded_reason"
] = "Under 10 minutes"

expanded_videos.loc[
    expanded_videos["title"].str.contains(
        exclude_pattern,
        regex=True,
        na=False
    ),
    "excluded_reason"
] = "Clip, Short, trailer, or promo"

eligible_episodes = expanded_videos[
    expanded_videos["excluded_reason"].eq("")
].copy()

# Keep up to 50 recent full episodes from every podcast
all_episodes = (
    eligible_episodes
    .sort_values(
        ["requested_channel", "published_at"],
        ascending=[True, False]
    )
    .groupby(
        "requested_channel",
        group_keys=False
    )
    .head(TARGET_EPISODES)
    .reset_index(drop=True)
)

episode_counts = (
    all_episodes
    .groupby("requested_channel")
    .size()
    .rename("episode_count")
    .reset_index()
)

display(episode_counts)

print("Final total:", len(all_episodes))


# In[10]:


channel_topic_categories = {}

for channel_name, channel_df in all_episodes.groupby(
    "requested_channel"
):
    episode_records = channel_df[
        ["video_id", "classification_text"]
    ].to_dict(orient="records")

    prompt = f"""
Review these episodes from one podcast and discover 3 to 5 broad,
recurring topic categories.

Rules:
- Categories should be distinct.
- Each category should plausibly contain multiple episodes.
- Avoid categories based on only one episode.
- Together they should cover nearly all episodes.
- Use only the supplied titles and descriptions.
- Do not classify individual episodes yet.

Episodes:
{json.dumps(episode_records, ensure_ascii=False)}
"""

    result = call_gemini_structured(
        prompt,
        TopicSet
    )

    categories = [
        category.model_dump()
        for category in result.categories
    ]

    channel_topic_categories[channel_name] = categories

    print(f"\n{channel_name}")
    for category in categories:
        print(
            f"- {category['name']}: "
            f"{category['definition']}"
        )


# In[11]:


class CombinedEpisodeLabel(BaseModel):
    video_id: str
    topic: str

    topic_evidence: str = Field(
        description="A short exact quote supporting the topic"
    )

    format: Literal["Solo", "Guest", "Unclear"]

    format_evidence: str = Field(
        description="A short exact quote supporting the format"
    )


class CombinedEpisodeLabelSet(BaseModel):
    episodes: list[CombinedEpisodeLabel]


def get_format_definitions(channel_name):
    if channel_name == "The Chic Code":
        return """
- Guest: someone beyond the two regular hosts is featured.
- Solo: only one or both regular hosts appear.
- Unclear: the supplied text does not provide enough evidence.
"""

    return """
- Guest: someone beyond the regular host or hosts is featured.
- Solo: only the regular host or hosts appear.
- Unclear: the supplied text does not provide enough evidence.
"""


all_labels = []

for channel_name, channel_df in all_episodes.groupby(
    "requested_channel"
):
    categories = channel_topic_categories[channel_name]

    valid_topic_names = [
        category["name"]
        for category in categories
    ]

    print(f"Classifying {channel_name}...")

    for start in range(0, len(channel_df), 10):
        batch = channel_df.iloc[start:start + 10]

        records = batch[
            ["video_id", "classification_text"]
        ].to_dict(orient="records")

        prompt = f"""
Classify every podcast episode below.

Podcast:
{channel_name}

Use exactly one of these topic categories:
{json.dumps(categories, ensure_ascii=False, indent=2)}

Format definitions:
{get_format_definitions(channel_name)}

Rules:
- Use one exact topic category name from the supplied list.
- Provide a short exact quote supporting the topic.
- Provide a short exact quote supporting the format.
- Do not invent or paraphrase evidence.
- If format is unclear, leave format_evidence empty.
- Return exactly one result for every video_id.

Episodes:
{json.dumps(records, ensure_ascii=False)}
"""

        result = call_gemini_structured(
            prompt,
            CombinedEpisodeLabelSet
        )

        batch_labels = [
            label.model_dump()
            for label in result.episodes
        ]

        expected_ids = set(batch["video_id"])
        returned_ids = {
            label["video_id"]
            for label in batch_labels
        }

        if returned_ids != expected_ids:
            raise ValueError(
                f"Returned video IDs do not match the batch "
                f"for {channel_name}"
            )

        for label in batch_labels:
            if label["topic"] not in valid_topic_names:
                raise ValueError(
                    f"Unexpected topic for {channel_name}: "
                    f"{label['topic']}"
                )

            label["requested_channel"] = channel_name

        all_labels.extend(batch_labels)

        time.sleep(1)


all_labels_df = pd.DataFrame(all_labels)

all_episodes_labeled = (
    all_episodes
    .merge(
        all_labels_df.drop(
            columns="requested_channel"
        ),
        on="video_id",
        how="left",
        validate="one_to_one"
    )
)

display(
    all_episodes_labeled[
        [
            "requested_channel",
            "title",
            "topic",
            "format",
            "format_evidence"
        ]
    ]
)

print("\nFormat counts:")

display(
    all_episodes_labeled
    .groupby(
        ["requested_channel", "format"]
    )
    .size()
    .rename("episode_count")
    .reset_index()
)


# In[13]:


# IMPORTANT: rebuild from the newly expanded and labeled dataset
final_df = all_episodes_labeled.copy()

print("Expanded episodes:", len(all_episodes))
print("Expanded labeled episodes:", len(all_episodes_labeled))

display(
    final_df.groupby("requested_channel")
    .size()
    .rename("episode_count")
    .reset_index()
)


def normalize_text(text):
    text = "" if pd.isna(text) else str(text)

    return (
        re.sub(r"\s+", " ", text)
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .strip()
        .lower()
    )


def find_evidence_source(row, evidence_column):
    evidence = normalize_text(row.get(evidence_column, ""))

    if not evidence:
        return "None"

    sources = {
        "Title": row.get("title", ""),
        "Description": row.get("description", "")
    }

    for source_name, source_text in sources.items():
        if evidence in normalize_text(source_text):
            return source_name

    return "None"


# Recreate transcript-review flags without requesting transcripts
final_df["format_evidence_clean"] = (
    final_df["format_evidence"]
    .fillna("")
    .str.strip()
)

evidence_frequency = (
    final_df.groupby(
        ["requested_channel", "format_evidence_clean"]
    )["video_id"]
    .transform("size")
)

final_df["format_evidence_repeated"] = (
    final_df["format_evidence_clean"].ne("")
    & evidence_frequency.ge(3)
)

final_df["needs_transcript"] = (
    final_df["format"].eq("Unclear")
    | final_df["format_evidence_clean"].eq("")
    | final_df["format_evidence_repeated"]
)

# Validate evidence
final_df["topic_source"] = final_df.apply(
    lambda row: find_evidence_source(
        row,
        "topic_evidence"
    ),
    axis=1
)

final_df["format_source"] = final_df.apply(
    lambda row: find_evidence_source(
        row,
        "format_evidence"
    ),
    axis=1
)

final_df["topic_evidence_valid"] = (
    final_df["topic_source"] != "None"
)

final_df["format_evidence_valid"] = (
    final_df["format_source"] != "None"
)

# Basic episode features
final_df["published_at"] = pd.to_datetime(
    final_df["published_at"],
    utc=True
)

final_df["upload_day"] = (
    final_df["published_at"].dt.day_name()
)

final_df["upload_hour_utc"] = (
    final_df["published_at"].dt.hour
)

final_df["title_word_count"] = (
    final_df["title"]
    .fillna("")
    .str.split()
    .str.len()
)

final_df["video_url"] = (
    "https://www.youtube.com/watch?v="
    + final_df["video_id"]
)

# Log-transform and normalize engagement within each podcast
engagement_metrics = [
    "views_per_day",
    "likes_per_1000_views",
    "comments_per_1000_views"
]

for column in engagement_metrics:
    final_df[f"log_{column}"] = np.log1p(
        final_df[column].clip(lower=0)
    )

    log_column = f"log_{column}"

    final_df[f"{log_column}_z"] = (
        final_df.groupby("requested_channel")[log_column]
        .transform(
            lambda values: (
                (values - values.mean()) / values.std(ddof=0)
                if values.std(ddof=0) > 0
                else 0
            )
        )
    )

final_df["engagement_score"] = final_df[
    [
        "log_views_per_day_z",
        "log_likes_per_1000_views_z",
        "log_comments_per_1000_views_z"
    ]
].mean(axis=1)

# Reliable format labels use direct, non-repeated evidence
final_df["format_reliable"] = (
    final_df["format"].isin(["Solo", "Guest"])
    & final_df["format_evidence_valid"]
    & ~final_df["needs_transcript"]
)

analysis_columns = [
    "requested_channel",
    "channel_handle",
    "video_id",
    "video_url",
    "title",
    "published_at",
    "upload_day",
    "upload_hour_utc",
    "title_word_count",
    "duration_minutes",
    "age_days",
    "views",
    "likes",
    "comments",
    "views_per_day",
    "likes_per_1000_views",
    "comments_per_1000_views",
    "engagement_score",
    "topic",
    "topic_source",
    "topic_evidence_valid",
    "format",
    "format_source",
    "format_evidence_valid",
    "format_reliable"
]

analysis_ready = final_df[analysis_columns].copy()

analysis_ready.to_csv(
    "podcast_analysis_ready.csv",
    index=False
)

print("\nRows saved:", len(analysis_ready))

display(
    analysis_ready.groupby("requested_channel")
    .size()
    .rename("saved_episodes")
    .reset_index()
)


# In[14]:


# Audit dataset coverage and usable format comparisons

channel_summary = (
    analysis_ready
    .groupby("requested_channel")
    .agg(
        episodes=("video_id", "size"),
        topic_categories=("topic", "nunique"),
        reliable_format_labels=("format_reliable", "sum"),
        missing_engagement=("engagement_score", lambda x: x.isna().sum())
    )
    .reset_index()
)

reliable_rows = analysis_ready[
    analysis_ready["format_reliable"]
]

format_counts = pd.crosstab(
    reliable_rows["requested_channel"],
    reliable_rows["format"]
)

for column in ["Solo", "Guest"]:
    if column not in format_counts.columns:
        format_counts[column] = 0

eligible_format_channels = format_counts[
    (format_counts["Solo"] >= 3)
    & (format_counts["Guest"] >= 3)
].index.tolist()

display(channel_summary)
display(format_counts)

print(
    "Channels usable for guest-versus-solo comparison:",
    eligible_format_channels
)


# In[15]:


format_df = analysis_ready[
    (analysis_ready["requested_channel"] == "A Really Good Cry")
    & analysis_ready["format_reliable"]
].copy()

metrics = {
    "views_per_day": "Views per day",
    "likes_per_1000_views": "Likes per 1,000 views",
    "comments_per_1000_views": "Comments per 1,000 views",
    "engagement_score": "Combined engagement score"
}

results = []

for column, label in metrics.items():
    guest = format_df.loc[
        format_df["format"] == "Guest",
        column
    ].dropna()

    solo = format_df.loc[
        format_df["format"] == "Solo",
        column
    ].dropna()

    statistic, p_value = mannwhitneyu(
        guest,
        solo,
        alternative="two-sided"
    )

    rank_biserial = (
        2 * statistic / (len(guest) * len(solo))
    ) - 1

    results.append({
        "metric": label,
        "guest_median": guest.median(),
        "solo_median": solo.median(),
        "median_difference": guest.median() - solo.median(),
        "rank_biserial_effect": rank_biserial,
        "p_value": p_value
    })

format_results = pd.DataFrame(results)

display(
    format_results.round(3)
)


# In[16]:


p_values = format_results["p_value"].to_numpy()
order = np.argsort(p_values)
adjusted = np.empty(len(p_values))

previous = 0

for rank, index in enumerate(order):
    corrected = (len(p_values) - rank) * p_values[index]
    previous = max(previous, corrected)
    adjusted[index] = min(previous, 1)

format_results["holm_adjusted_p"] = adjusted
format_results["statistically_clear"] = (
    format_results["holm_adjusted_p"] < 0.05
)

display(format_results.round(3))


# ## Quick Visuals

# In[17]:


plot_metrics = {
    "views_per_day": "Views per Day",
    "likes_per_1000_views": "Likes per 1,000 Views",
    "comments_per_1000_views": "Comments per 1,000 Views"
}

for column, label in plot_metrics.items():
    guest_values = format_df.loc[
        format_df["format"] == "Guest",
        column
    ].dropna()

    solo_values = format_df.loc[
        format_df["format"] == "Solo",
        column
    ].dropna()

    plt.figure(figsize=(6, 4))

    plt.boxplot(
        [solo_values, guest_values],
        tick_labels=["Solo", "Guest"],
        showmeans=True
    )

    plt.ylabel(label)
    plt.title(f"{label}: Solo vs. Guest Episodes")
    plt.tight_layout()

    filename = (
        label.lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace(".", "")
        + "_by_format.png"
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# In[18]:


topic_results = (
    analysis_ready
    .groupby(["requested_channel", "topic"])
    .agg(
        episode_count=("video_id", "size"),
        median_views_per_day=("views_per_day", "median"),
        median_likes_per_1000=("likes_per_1000_views", "median"),
        median_comments_per_1000=("comments_per_1000_views", "median"),
        median_engagement_score=("engagement_score", "median")
    )
    .reset_index()
)

# Keep topics represented by at least 3 episodes
topic_results = topic_results[
    topic_results["episode_count"] >= 3
].copy()

topic_results = topic_results.sort_values(
    ["requested_channel", "median_engagement_score"],
    ascending=[True, False]
)

display(topic_results.round(2))


# In[19]:


import numpy as np
import pandas as pd
from scipy.stats import kruskal


metrics = {
    "views_per_day": "Views per day",
    "likes_per_1000_views": "Likes per 1,000 views",
    "comments_per_1000_views": "Comments per 1,000 views"
}

topic_tests = []

for channel_name, channel_df in analysis_ready.groupby(
    "requested_channel"
):
    # Only retain topics with at least 3 episodes
    valid_topics = (
        channel_df["topic"]
        .value_counts()
        .loc[lambda counts: counts >= 3]
        .index
    )

    test_df = channel_df[
        channel_df["topic"].isin(valid_topics)
    ]

    for metric, metric_label in metrics.items():
        groups = [
            group[metric].dropna().to_numpy()
            for _, group in test_df.groupby("topic")
        ]

        groups = [
            values for values in groups
            if len(values) >= 3
        ]

        # Skip tests with insufficient groups or no variation
        if len(groups) < 2:
            continue

        combined_values = np.concatenate(groups)

        if np.all(combined_values == combined_values[0]):
            continue

        statistic, p_value = kruskal(*groups)

        n = sum(len(values) for values in groups)
        k = len(groups)

        epsilon_squared = max(
            (statistic - k + 1) / (n - k),
            0
        )

        topic_tests.append({
            "podcast": channel_name,
            "metric": metric_label,
            "topic_count": k,
            "episode_count": n,
            "kruskal_h": statistic,
            "effect_size": epsilon_squared,
            "p_value": p_value
        })


topic_test_results = pd.DataFrame(topic_tests)

# Holm correction across all topic tests
p_values = topic_test_results["p_value"].to_numpy()
order = np.argsort(p_values)
adjusted = np.empty(len(p_values))
previous = 0

for rank, index in enumerate(order):
    corrected = (len(p_values) - rank) * p_values[index]
    previous = max(previous, corrected)
    adjusted[index] = min(previous, 1)

topic_test_results["holm_adjusted_p"] = adjusted
topic_test_results["statistically_clear"] = (
    topic_test_results["holm_adjusted_p"] < 0.05
)

display(
    topic_test_results
    .sort_values("holm_adjusted_p")
    .round(3)
)


# In[20]:


get_ipython().run_line_magic('pip', 'install -q statsmodels')


# In[21]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


regression_df = analysis_ready.copy()

# Compare each episode with what is typical for its own podcast
regression_df["duration_within_channel"] = (
    regression_df["duration_minutes"]
    - regression_df.groupby("requested_channel")[
        "duration_minutes"
    ].transform("mean")
)

regression_df["title_words_within_channel"] = (
    regression_df["title_word_count"]
    - regression_df.groupby("requested_channel")[
        "title_word_count"
    ].transform("mean")
)

regression_df["log_age_days"] = np.log1p(
    regression_df["age_days"]
)

regression_df["age_within_channel"] = (
    regression_df["log_age_days"]
    - regression_df.groupby("requested_channel")[
        "log_age_days"
    ].transform("mean")
)

# Standardize predictors for easier comparison
predictors = [
    "duration_within_channel",
    "title_words_within_channel",
    "age_within_channel"
]

for predictor in predictors:
    standard_deviation = regression_df[predictor].std(ddof=0)

    regression_df[f"{predictor}_z"] = (
        regression_df[predictor] / standard_deviation
    )

# Log-transform engagement outcomes
outcomes = {
    "views_per_day": "Views per day",
    "likes_per_1000_views": "Likes per 1,000 views",
    "comments_per_1000_views": "Comments per 1,000 views"
}

results = []

for outcome, outcome_label in outcomes.items():
    regression_df[f"log_{outcome}"] = np.log1p(
        regression_df[outcome].clip(lower=0)
    )

    formula = (
        f"log_{outcome} ~ "
        "duration_within_channel_z + "
        "title_words_within_channel_z + "
        "age_within_channel_z + "
        "C(requested_channel)"
    )

    model = smf.ols(
        formula=formula,
        data=regression_df
    ).fit(cov_type="HC3")

    for predictor, predictor_label in [
        (
            "duration_within_channel_z",
            "Episode duration"
        ),
        (
            "title_words_within_channel_z",
            "Title word count"
        ),
        (
            "age_within_channel_z",
            "Video age"
        )
    ]:
        results.append({
            "outcome": outcome_label,
            "predictor": predictor_label,
            "coefficient": model.params[predictor],
            "standard_error": model.bse[predictor],
            "p_value": model.pvalues[predictor]
        })


continuous_results = pd.DataFrame(results)

# Holm correction across the nine predictor tests
p_values = continuous_results["p_value"].to_numpy()
order = np.argsort(p_values)
adjusted = np.empty(len(p_values))
previous = 0

for rank, index in enumerate(order):
    corrected = (
        len(p_values) - rank
    ) * p_values[index]

    previous = max(previous, corrected)
    adjusted[index] = min(previous, 1)

continuous_results["holm_adjusted_p"] = adjusted
continuous_results["statistically_clear"] = (
    continuous_results["holm_adjusted_p"] < 0.05
)

display(
    continuous_results
    .sort_values("holm_adjusted_p")
    .round(3)
)


# In[22]:


# upload_day_counts = pd.crosstab(
#     analysis_ready["requested_channel"],
#     analysis_ready["upload_day"]
# )

# display(upload_day_counts)

# print("\nUnique upload days per podcast:")
# display(
#     analysis_ready.groupby("requested_channel")["upload_day"]
#     .nunique()
#     .rename("unique_upload_days")
#     .reset_index()
# )


# In[23]:


import matplotlib.pyplot as plt

topic_plot = (
    topic_results
    .sort_values(
        ["requested_channel", "median_engagement_score"],
        ascending=[True, True]
    )
    .copy()
)

topic_plot["label"] = (
    topic_plot["requested_channel"]
    + " — "
    + topic_plot["topic"]
)

plt.figure(figsize=(10, 9))

plt.barh(
    topic_plot["label"],
    topic_plot["median_engagement_score"]
)

plt.axvline(0, linewidth=1)

plt.xlabel("Median Engagement Score\n(relative to other episodes from the same podcast)")
plt.ylabel("")
plt.title("Topic Performance Within Each Podcast")

plt.tight_layout()
plt.savefig(
    "topic_performance_by_podcast.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# In[24]:


top_topics = (
    topic_results
    .sort_values(
        ["requested_channel", "median_engagement_score"],
        ascending=[True, False]
    )
    .groupby("requested_channel", as_index=False)
    .first()
    [
        [
            "requested_channel",
            "topic",
            "episode_count",
            "median_views_per_day",
            "median_likes_per_1000",
            "median_comments_per_1000",
            "median_engagement_score"
        ]
    ]
)

top_topics = top_topics.rename(
    columns={
        "requested_channel": "podcast",
        "topic": "highest_scoring_topic"
    }
)

display(top_topics.round(2))

top_topics.to_csv(
    "top_topic_by_podcast.csv",
    index=False
)


# In[25]:


# Save the final analytical outputs

format_results.to_csv(
    "guest_vs_solo_results.csv",
    index=False
)

topic_results.to_csv(
    "topic_performance_results.csv",
    index=False
)

topic_test_results.to_csv(
    "topic_significance_tests.csv",
    index=False
)

continuous_results.to_csv(
    "episode_characteristic_regressions.csv",
    index=False
)

top_topics.to_csv(
    "top_topic_by_podcast.csv",
    index=False
)

channel_summary.to_csv(
    "podcast_sample_summary.csv",
    index=False
)

print("Final datasets and results saved.")


# In[ ]:





# In[26]:


import zipfile
from pathlib import Path

files_to_share = [
    "podcast_analysis_ready.csv",
    "guest_vs_solo_results.csv",
    "topic_performance_results.csv",
    "topic_significance_tests.csv",
    "episode_characteristic_regressions.csv",
    "top_topic_by_podcast.csv",
    "podcast_sample_summary.csv",
]

# Include all saved charts
files_to_share += [
    str(path)
    for path in Path(".").glob("*.png")
]

existing_files = [
    file for file in files_to_share
    if Path(file).exists()
]

with zipfile.ZipFile(
    "podcast_project_results.zip",
    "w",
    compression=zipfile.ZIP_DEFLATED
) as zip_file:
    for file in existing_files:
        zip_file.write(file, arcname=Path(file).name)

print("Created podcast_project_results.zip")
print("Included files:")
for file in existing_files:
    print("-", file)


# In[ ]:





# # Extra

# In[27]:


# --------------------------------------------------
# Prepare data
# --------------------------------------------------

time_ml_df = final_df.copy()

time_ml_df["published_at"] = pd.to_datetime(
    time_ml_df["published_at"],
    utc=True
)

time_ml_df["format_ml"] = np.where(
    time_ml_df["format_reliable"],
    time_ml_df["format"],
    "Unknown"
)

time_ml_df["log_age_days"] = np.log1p(
    time_ml_df["age_days"]
)


def clean_text(text):
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


time_ml_df["text"] = (
    time_ml_df["title"].map(clean_text)
    + " "
    + time_ml_df["description"].map(clean_text).str[:2000]
)

# Predict log likes per 1,000 views
time_ml_df["target"] = np.log1p(
    time_ml_df["likes_per_1000_views"].clip(lower=0)
)

required_columns = [
    "requested_channel",
    "published_at",
    "text",
    "duration_minutes",
    "title_word_count",
    "upload_hour_utc",
    "log_age_days",
    "upload_day",
    "format_ml",
    "target"
]

time_ml_df = (
    time_ml_df
    .dropna(subset=required_columns)
    .sort_values(
        ["requested_channel", "published_at"]
    )
    .reset_index(drop=True)
)


# --------------------------------------------------
# Chronological split within every podcast
# --------------------------------------------------

def make_time_split(dataframe, train_fraction=0.80):
    train_indices = []
    test_indices = []

    for _, group in dataframe.groupby(
        "requested_channel",
        sort=False
    ):
        split_position = int(
            np.floor(len(group) * train_fraction)
        )

        # Preserve at least two episodes for testing
        split_position = min(
            max(split_position, 1),
            len(group) - 2
        )

        train_indices.extend(
            group.index[:split_position]
        )

        test_indices.extend(
            group.index[split_position:]
        )

    return (
        np.array(train_indices),
        np.array(test_indices)
    )


outer_train_index, outer_test_index = make_time_split(
    time_ml_df,
    train_fraction=0.80
)

train_df = time_ml_df.loc[
    outer_train_index
].copy()

test_df = time_ml_df.loc[
    outer_test_index
].copy()


# Inner chronological split for choosing alpha
inner_train_index, inner_validation_index = make_time_split(
    train_df.reset_index(drop=True),
    train_fraction=0.80
)

inner_df = train_df.reset_index(drop=True)

inner_train_df = inner_df.loc[
    inner_train_index
]

inner_validation_df = inner_df.loc[
    inner_validation_index
]


# --------------------------------------------------
# Text + metadata model
# --------------------------------------------------

numeric_features = [
    "duration_minutes",
    "title_word_count",
    "upload_hour_utc",
    "log_age_days"
]

categorical_features = [
    "upload_day",
    "format_ml",
    "requested_channel"
]

feature_columns = (
    ["text"]
    + numeric_features
    + categorical_features
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.95,
                max_features=5000,
                sublinear_tf=True
            ),
            "text"
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


def build_model(alpha):
    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "ridge",
            Ridge(
                alpha=alpha,
                solver="lsqr"
            )
        )
    ])


# --------------------------------------------------
# Select alpha using earlier training episodes only
# --------------------------------------------------

alpha_results = []

for alpha in [0.1, 1.0, 10.0, 100.0]:
    model = build_model(alpha)

    model.fit(
        inner_train_df[feature_columns],
        inner_train_df["target"]
    )

    validation_predictions = model.predict(
        inner_validation_df[feature_columns]
    )

    alpha_results.append({
        "alpha": alpha,
        "validation_mae": mean_absolute_error(
            inner_validation_df["target"],
            validation_predictions
        )
    })

alpha_results = pd.DataFrame(alpha_results)

best_alpha = alpha_results.loc[
    alpha_results["validation_mae"].idxmin(),
    "alpha"
]

print("Best alpha:", best_alpha)
display(alpha_results.round(3))


# --------------------------------------------------
# Fit earlier episodes and predict newest episodes
# --------------------------------------------------

final_model = build_model(best_alpha)

final_model.fit(
    train_df[feature_columns],
    train_df["target"]
)

test_predictions_log = final_model.predict(
    test_df[feature_columns]
)

test_actual = np.expm1(
    test_df["target"].to_numpy()
)

test_predictions = np.clip(
    np.expm1(test_predictions_log),
    0,
    None
)


# Podcast-specific baseline: median of earlier episodes
training_medians = (
    train_df.groupby("requested_channel")[
        "likes_per_1000_views"
    ]
    .median()
)

baseline_predictions = (
    test_df["requested_channel"]
    .map(training_medians)
    .to_numpy()
)


print(
    "\nOverall model MAE:",
    round(
        mean_absolute_error(
            test_actual,
            test_predictions
        ),
        3
    )
)

print(
    "Overall baseline MAE:",
    round(
        mean_absolute_error(
            test_actual,
            baseline_predictions
        ),
        3
    )
)

print(
    "Overall model R²:",
    round(
        r2_score(
            test_df["target"],
            test_predictions_log
        ),
        3
    )
)


# --------------------------------------------------
# Results by podcast
# --------------------------------------------------

test_results = test_df[
    [
        "requested_channel",
        "video_id",
        "published_at"
    ]
].copy()

test_results["actual_likes_per_1000"] = test_actual
test_results["predicted_likes_per_1000"] = test_predictions
test_results["baseline_likes_per_1000"] = baseline_predictions

podcast_time_results = []

for podcast, group in test_results.groupby(
    "requested_channel"
):
    podcast_time_results.append({
        "podcast": podcast,
        "test_episodes": len(group),
        "model_mae": mean_absolute_error(
            group["actual_likes_per_1000"],
            group["predicted_likes_per_1000"]
        ),
        "baseline_mae": mean_absolute_error(
            group["actual_likes_per_1000"],
            group["baseline_likes_per_1000"]
        )
    })

podcast_time_results = pd.DataFrame(
    podcast_time_results
)

podcast_time_results["model_beats_baseline"] = (
    podcast_time_results["model_mae"]
    < podcast_time_results["baseline_mae"]
)

display(
    podcast_time_results.round(3)
)


# In[ ]:





# # Build Report

# In[31]:


get_ipython().run_line_magic('pip', 'install -q -U plotly')

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


# =========================================================
# SETTINGS
# =========================================================

RANDOM_SEED = 42
N_BOOTSTRAPS = 10

ACCENT_PINK = "#E84A8A"
ACCENT_TEAL = "#1B9AAA"
ACCENT_GOLD = "#E9A23B"
DARK_TEXT = "#18212F"
MUTED_TEXT = "#667085"
GRID_COLOR = "#E7E9EE"
BACKGROUND = "#FAFAFC"

rng = np.random.default_rng(RANDOM_SEED)


# =========================================================
# VISUAL 1 — BOOTSTRAP EFFECT PLOT
# Guest versus solo as percent differences
# =========================================================

format_viz_df = analysis_ready[
    (analysis_ready["requested_channel"] == "A Really Good Cry")
    & analysis_ready["format_reliable"]
].copy()

bootstrap_metrics = {
    "views_per_day": "Views per day",
    "likes_per_1000_views": "Likes per 1,000 views",
    "comments_per_1000_views": "Comments per 1,000 views"
}

bootstrap_rows = []

for column, label in bootstrap_metrics.items():

    guest = format_viz_df.loc[
        format_viz_df["format"] == "Guest",
        column
    ].dropna().to_numpy()

    solo = format_viz_df.loc[
        format_viz_df["format"] == "Solo",
        column
    ].dropna().to_numpy()

    guest_median = np.median(guest)
    solo_median = np.median(solo)

    observed_percent = (
        (guest_median / solo_median) - 1
    ) * 100

    bootstrap_percent = np.empty(N_BOOTSTRAPS)

    for index in range(N_BOOTSTRAPS):

        guest_sample = rng.choice(
            guest,
            size=len(guest),
            replace=True
        )

        solo_sample = rng.choice(
            solo,
            size=len(solo),
            replace=True
        )

        sampled_solo_median = np.median(solo_sample)

        bootstrap_percent[index] = (
            (
                np.median(guest_sample)
                / sampled_solo_median
            ) - 1
        ) * 100

    ci_low, ci_high = np.percentile(
        bootstrap_percent,
        [2.5, 97.5]
    )

    bootstrap_rows.append({
        "metric": label,
        "percent_difference": observed_percent,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "guest_median": guest_median,
        "solo_median": solo_median,
        "guest_n": len(guest),
        "solo_n": len(solo),
        "clear": ci_low > 0 or ci_high < 0
    })

bootstrap_plot_df = pd.DataFrame(bootstrap_rows)

metric_order = [
    "Views per day",
    "Likes per 1,000 views",
    "Comments per 1,000 views"
]

bootstrap_plot_df["metric"] = pd.Categorical(
    bootstrap_plot_df["metric"],
    categories=metric_order[::-1],
    ordered=True
)

bootstrap_plot_df = bootstrap_plot_df.sort_values(
    "metric"
)

bootstrap_plot_df["error_plus"] = (
    bootstrap_plot_df["ci_high"]
    - bootstrap_plot_df["percent_difference"]
)

bootstrap_plot_df["error_minus"] = (
    bootstrap_plot_df["percent_difference"]
    - bootstrap_plot_df["ci_low"]
)

bootstrap_plot_df["color"] = np.where(
    bootstrap_plot_df["clear"],
    ACCENT_TEAL,
    "#A8ADB7"
)

fig_bootstrap = go.Figure()

for _, row in bootstrap_plot_df.iterrows():

    fig_bootstrap.add_trace(
        go.Scatter(
            x=[row["percent_difference"]],
            y=[row["metric"]],
            mode="markers",
            marker=dict(
                size=18,
                color=row["color"],
                line=dict(
                    color="white",
                    width=2
                )
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[row["error_plus"]],
                arrayminus=[row["error_minus"]],
                thickness=3,
                width=8,
                color=row["color"]
            ),
            customdata=[[
                row["ci_low"],
                row["ci_high"],
                row["guest_median"],
                row["solo_median"],
                row["guest_n"],
                row["solo_n"]
            ]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Guest − Solo: %{x:.1f}%<br>"
                "95% CI: %{customdata[0]:.1f}% to "
                "%{customdata[1]:.1f}%<br>"
                "Guest median: %{customdata[2]:.2f}<br>"
                "Solo median: %{customdata[3]:.2f}<br>"
                "Sample: %{customdata[4]} guest, "
                "%{customdata[5]} solo"
                "<extra></extra>"
            ),
            showlegend=False
        )
    )

fig_bootstrap.add_vline(
    x=0,
    line_width=2,
    line_dash="dash",
    line_color=DARK_TEXT
)

fig_bootstrap.add_annotation(
    x=0,
    y=1.13,
    xref="x",
    yref="paper",
    text="← Solo higher       Guest higher →",
    showarrow=False,
    font=dict(
        size=13,
        color=MUTED_TEXT
    )
)

fig_bootstrap.update_layout(
    title=dict(
        text=(
            "<b>Guest Episodes Expand Reach, "
            "Solo Episodes Deepen Interaction</b>"
            "<br><sup>A Really Good Cry · "
            "Median differences with 95% confidence intervals</sup>"
        ),
        x=0.02
    ),
    xaxis_title="Guest versus solo difference (%)",
    yaxis_title="",
    template="plotly_white",
    paper_bgcolor=BACKGROUND,
    plot_bgcolor="white",
    height=470,
    margin=dict(
        l=180,
        r=50,
        t=110,
        b=70
    ),
    font=dict(
        family="Arial",
        color=DARK_TEXT
    )
)

fig_bootstrap.update_xaxes(
    range=[-100, 250],
    gridcolor=GRID_COLOR,
    zeroline=False,
    ticksuffix="%"
)

fig_bootstrap.update_yaxes(
    showgrid=False
)


# =========================================================
# VISUAL 2 — TOPIC CONSTELLATION
# Each bubble is one topic
# =========================================================

topic_viz = topic_results.copy()

podcast_order = (
    topic_viz.groupby("requested_channel")[
        "median_engagement_score"
    ]
    .max()
    .sort_values()
    .index
    .tolist()
)

podcast_positions = {
    podcast: position
    for position, podcast in enumerate(podcast_order)
}

topic_viz["podcast_position"] = (
    topic_viz["requested_channel"]
    .map(podcast_positions)
    .astype(float)
)

topic_viz["topic_index"] = (
    topic_viz.groupby("requested_channel")
    .cumcount()
)

topic_viz["topics_in_podcast"] = (
    topic_viz.groupby("requested_channel")[
        "topic"
    ]
    .transform("size")
)

# Give topics within each podcast a small vertical offset
topic_viz["vertical_offset"] = (
    topic_viz["topic_index"]
    - (topic_viz["topics_in_podcast"] - 1) / 2
) * 0.10

topic_viz["y_position"] = (
    topic_viz["podcast_position"]
    + topic_viz["vertical_offset"]
)

fig_topics = px.scatter(
    topic_viz,
    x="median_engagement_score",
    y="y_position",
    size="episode_count",
    color="median_engagement_score",
    color_continuous_scale=[
        [0.00, "#C84B5A"],
        [0.45, "#F1C6C9"],
        [0.50, "#F2F2F2"],
        [0.55, "#B9DFDA"],
        [1.00, "#148A80"]
    ],
    color_continuous_midpoint=0,
    size_max=34,
    custom_data=[
        "requested_channel",
        "topic",
        "episode_count",
        "median_views_per_day",
        "median_likes_per_1000",
        "median_comments_per_1000"
    ]
)

fig_topics.update_traces(
    marker=dict(
        opacity=0.88,
        line=dict(
            color="white",
            width=1.5
        )
    ),
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"
        "%{customdata[1]}<br><br>"
        "Relative engagement: %{x:.2f}<br>"
        "Episodes: %{customdata[2]}<br>"
        "Median views/day: %{customdata[3]:,.1f}<br>"
        "Median likes/1,000: %{customdata[4]:.1f}<br>"
        "Median comments/1,000: %{customdata[5]:.2f}"
        "<extra></extra>"
    )
)

fig_topics.add_vline(
    x=0,
    line_width=2,
    line_dash="dash",
    line_color=DARK_TEXT
)

fig_topics.update_layout(
    title=dict(
        text=(
            "<b>The Topic Constellation</b>"
            "<br><sup>Bubble size represents the number of episodes; "
            "position shows performance relative to the podcast’s norm</sup>"
        ),
        x=0.02
    ),
    xaxis_title=(
        "Median engagement score "
        "(relative to the same podcast)"
    ),
    yaxis_title="",
    yaxis=dict(
        tickmode="array",
        tickvals=list(podcast_positions.values()),
        ticktext=list(podcast_positions.keys()),
        range=[
            -0.6,
            len(podcast_positions) - 0.4
        ],
        showgrid=False
    ),
    coloraxis_colorbar=dict(
        title="Relative<br>engagement"
    ),
    template="plotly_white",
    paper_bgcolor=BACKGROUND,
    plot_bgcolor="white",
    height=710,
    margin=dict(
        l=225,
        r=80,
        t=100,
        b=70
    ),
    font=dict(
        family="Arial",
        color=DARK_TEXT
    )
)

fig_topics.update_xaxes(
    gridcolor=GRID_COLOR,
    zeroline=False
)


# =========================================================
# VISUAL 3 — EPISODE UNIVERSE
# Reach versus loyalty, with bubble size showing discussion
# =========================================================

episode_viz = analysis_ready.copy()

episode_viz = episode_viz[
    episode_viz["views_per_day"].gt(0)
    & episode_viz["likes_per_1000_views"].notna()
    & episode_viz["comments_per_1000_views"].notna()
].copy()

episode_viz["format_display"] = np.where(
    episode_viz["format_reliable"],
    episode_viz["format"],
    "Unclear"
)

# Increase small differences in comment rate enough to be visible
episode_viz["discussion_bubble"] = (
    np.sqrt(
        episode_viz["comments_per_1000_views"]
        .clip(lower=0)
        + 0.10
    )
)

views_median = episode_viz["views_per_day"].median()
likes_median = episode_viz[
    "likes_per_1000_views"
].median()

fig_universe = px.scatter(
    episode_viz,
    x="views_per_day",
    y="likes_per_1000_views",
    size="discussion_bubble",
    color="requested_channel",
    symbol="format_display",
    log_x=True,
    size_max=28,
    opacity=0.72,
    custom_data=[
        "title",
        "requested_channel",
        "topic",
        "format_display",
        "comments_per_1000_views",
        "duration_minutes",
        "upload_day"
    ]
)

fig_universe.update_traces(
    marker=dict(
        line=dict(
            color="white",
            width=0.7
        )
    ),
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"
        "%{customdata[1]}<br><br>"
        "Topic: %{customdata[2]}<br>"
        "Format: %{customdata[3]}<br>"
        "Views/day: %{x:,.1f}<br>"
        "Likes/1,000: %{y:.1f}<br>"
        "Comments/1,000: %{customdata[4]:.2f}<br>"
        "Duration: %{customdata[5]:.1f} minutes<br>"
        "Uploaded: %{customdata[6]}"
        "<extra></extra>"
    )
)

fig_universe.add_vline(
    x=views_median,
    line_width=1.5,
    line_dash="dot",
    line_color=MUTED_TEXT
)

fig_universe.add_hline(
    y=likes_median,
    line_width=1.5,
    line_dash="dot",
    line_color=MUTED_TEXT
)

fig_universe.add_annotation(
    x=0.98,
    y=0.98,
    xref="paper",
    yref="paper",
    text="<b>High reach + high loyalty</b>",
    showarrow=False,
    xanchor="right",
    bgcolor="rgba(255,255,255,0.8)",
    font=dict(
        size=12,
        color=ACCENT_TEAL
    )
)

fig_universe.add_annotation(
    x=0.02,
    y=0.98,
    xref="paper",
    yref="paper",
    text="<b>Lower reach + high loyalty</b>",
    showarrow=False,
    xanchor="left",
    bgcolor="rgba(255,255,255,0.8)",
    font=dict(
        size=12,
        color=ACCENT_PINK
    )
)

fig_universe.update_layout(
    title=dict(
        text=(
            "<b>The Episode Universe: Reach versus Audience Loyalty</b>"
            "<br><sup>Bubble size represents comments per 1,000 views; "
            "hover over an episode for details</sup>"
        ),
        x=0.02
    ),
    xaxis_title="Views per day — logarithmic scale",
    yaxis_title="Likes per 1,000 views",
    legend_title_text="Podcast / format",
    template="plotly_white",
    paper_bgcolor=BACKGROUND,
    plot_bgcolor="white",
    height=720,
    margin=dict(
        l=80,
        r=40,
        t=105,
        b=80
    ),
    font=dict(
        family="Arial",
        color=DARK_TEXT
    )
)

fig_universe.update_xaxes(
    gridcolor=GRID_COLOR,
    zeroline=False
)

fig_universe.update_yaxes(
    gridcolor=GRID_COLOR,
    zeroline=False
)


# =========================================================
# SUMMARY VALUES FOR REPORT CARDS
# =========================================================

views_difference = bootstrap_plot_df.loc[
    bootstrap_plot_df["metric"] == "Views per day",
    "percent_difference"
].iloc[0]

likes_difference = bootstrap_plot_df.loc[
    bootstrap_plot_df["metric"] == "Likes per 1,000 views",
    "percent_difference"
].iloc[0]

episode_total = len(analysis_ready)
podcast_total = analysis_ready[
    "requested_channel"
].nunique()


# =========================================================
# CREATE ONE SELF-CONTAINED INTERACTIVE HTML REPORT
# =========================================================

plot_config = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True
}

bootstrap_html = pio.to_html(
    fig_bootstrap,
    full_html=False,
    include_plotlyjs=True,
    config=plot_config
)

topics_html = pio.to_html(
    fig_topics,
    full_html=False,
    include_plotlyjs=False,
    config=plot_config
)

universe_html = pio.to_html(
    fig_universe,
    full_html=False,
    include_plotlyjs=False,
    config=plot_config
)


# In[32]:


report_html = f"""
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>Women-Led Podcast Performance Report</title>

<style>
    body {{
        margin: 0;
        background: {BACKGROUND};
        color: {DARK_TEXT};
        font-family: Arial, Helvetica, sans-serif;
    }}

    .page {{
        max-width: 1280px;
        margin: auto;
        padding: 48px 30px 80px;
    }}

    .hero {{
        background:
            linear-gradient(
                120deg,
                #152238 0%,
                #293A67 55%,
                #A63E75 100%
            );
        color: white;
        padding: 52px;
        border-radius: 26px;
        box-shadow: 0 20px 50px rgba(31, 38, 58, 0.16);
    }}

    .hero h1 {{
        margin: 0 0 12px;
        font-size: 44px;
        line-height: 1.08;
    }}

    .hero p {{
        max-width: 760px;
        margin: 0;
        color: rgba(255, 255, 255, 0.84);
        font-size: 18px;
        line-height: 1.6;
    }}

    .cards {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 18px;
        margin: 26px 0 38px;
    }}

    .card {{
        background: white;
        border: 1px solid #ECEEF3;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 8px 28px rgba(31, 38, 58, 0.08);
    }}

    .card .number {{
        margin-bottom: 8px;
        font-size: 31px;
        font-weight: 750;
    }}

    .card .label {{
        color: {MUTED_TEXT};
        line-height: 1.35;
    }}

    .findings {{
        margin: 4px 0 38px;
    }}

    .findings h2 {{
        margin: 0 0 16px;
        font-size: 28px;
    }}

    .finding-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
    }}

    .finding {{
        background: white;
        border: 1px solid #ECEEF3;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 8px 28px rgba(31, 38, 58, 0.08);
    }}

    .finding-title {{
        margin-bottom: 10px;
        color: #1B9AAA;
        font-size: 21px;
        font-weight: 750;
    }}

    .finding p {{
        margin: 0;
        color: #4B5565;
        line-height: 1.55;
    }}

    .section {{
        background: white;
        border: 1px solid #ECEEF3;
        border-radius: 22px;
        margin-top: 26px;
        padding: 18px 18px 6px;
        box-shadow: 0 10px 34px rgba(31, 38, 58, 0.08);
    }}

    .chart-explainer {{
        margin: 8px 28px 4px;
        padding: 14px 18px;
        background: #F4F6FA;
        border-left: 4px solid #1B9AAA;
        border-radius: 8px;
        color: #4B5565;
        font-size: 15px;
        line-height: 1.55;
    }}

    .chart-explainer b {{
        color: {DARK_TEXT};
    }}

    .note {{
        margin: 35px 8px 0;
        color: {MUTED_TEXT};
        font-size: 14px;
        line-height: 1.6;
    }}

    @media (max-width: 900px) {{
        .cards {{
            grid-template-columns: repeat(2, 1fr);
        }}

        .finding-grid {{
            grid-template-columns: 1fr;
        }}

        .hero h1 {{
            font-size: 34px;
        }}
    }}

    @media (max-width: 560px) {{
        .page {{
            padding: 24px 12px 50px;
        }}

        .cards {{
            grid-template-columns: 1fr;
        }}

        .hero {{
            padding: 32px 24px;
        }}

        .chart-explainer {{
            margin: 8px 8px 4px;
        }}
    }}
</style>
</head>

<body>

<div class="page">

    <div class="hero">
        <h1>What Drives Engagement in Woman-Led Podcasts?</h1>

        <p>
            An exploratory analysis of women-led YouTube podcasts,
            examining reach, audience interaction, episode format,
            topic, duration, titles, and publishing patterns.
        </p>
    </div>

    <div class="cards">

        <div class="card">
            <div class="number">{episode_total}</div>

            <div class="label">
                full podcast episodes analyzed
            </div>
        </div>

        <div class="card">
            <div class="number">{podcast_total}</div>

            <div class="label">
                women-led podcast channels
            </div>
        </div>

        <div class="card">
            <div class="number">{views_difference:+.0f}%</div>

            <div class="label">
                guest-versus-solo difference in median views per day
            </div>
        </div>

        <div class="card">
            <div class="number">{abs(likes_difference):.0f}%</div>

            <div class="label">
                higher median likes per 1,000 views for solo episodes
            </div>
        </div>

    </div>

    <div class="findings">

        <h2>Key Findings</h2>

        <div class="finding-grid">

            <div class="finding">
                <div class="finding-title">
                    Guest episodes expand reach
                </div>

                <p>
                    Episodes including guests received substantially more median
                    views per day, while solo episodes earned more likes
                    per 1,000 views. This suggests guests may attract a
                    broader audience, while solo episodes may deepen
                    engagement among existing viewers.
                </p>
            </div>

            <div class="finding">
                <div class="finding-title">
                    Reach and loyalty differ
                </div>

                <p>
                    Episodes attracting the most viewers were not always
                    those receiving the most interactions per viewer.
                    Views, likes, and comments therefore represent
                    different dimensions of audience performance.
                </p>
            </div>

            <div class="finding">
                <div class="finding-title">
                    Topic effects varied by show
                </div>

                <p>
                    Several topics appeared stronger within individual
                    podcasts, but no topic pattern remained consistently
                    reliable across all eight channels after accounting
                    for multiple comparisons.
                </p>
            </div>

        </div>

    </div>

    <div class="section">

        <div class="chart-explainer">
            <b>How to read this:</b>
            Each dot shows the median difference between guest and solo
            episodes. The horizontal line is the 95% confidence
            interval. Results to the right favor guest episodes, while
            results to the left favor solo episodes; intervals that cross
            zero remain uncertain.
        </div>

        {bootstrap_html}

    </div>

    <div class="section">

        <div class="chart-explainer">
            <b>How to read this:</b>
            Each bubble represents one topic within a podcast. Topics to
            the right of zero performed above that podcast’s typical
            episode, while topics to the left performed below it. Larger
            bubbles represent topics supported by more episodes.
        </div>

        {topics_html}

    </div>

    <div class="section">

        <div class="chart-explainer">
            <b>How to read this:</b>
            Each point represents one episode. Moving right indicates
            greater reach, moving upward indicates more likes per viewer,
            and larger bubbles indicate more comments per viewer. The
            dotted lines divide episodes using the overall median values.
        </div>

        {universe_html}

    </div>

    <div class="note">
        Results are observational and exploratory. Associations should
        not be interpreted as causal effects. Topic performance is
        normalized within each podcast because audience size and content
        context differ substantially across channels.
    </div>

</div>

</body>
</html>
"""

output_path = Path(
    "podcast_visual_report.html"
)

output_path.write_text(
    report_html,
    encoding="utf-8"
)

print(
    "Created:",
    output_path.resolve()
)


# In[33]:


from IPython.display import FileLink, display

display(FileLink("podcast_visual_report.html"))


# In[ ]:





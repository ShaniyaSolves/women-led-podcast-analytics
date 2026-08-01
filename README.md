# women-led-podcast-analytics
An exploratory analysis of women-led YouTube podcasts, examining reach, audience interaction, episode format, topic, duration, titles, and publishing patterns.

# Women-Led Podcast Analytics

## What Drives Engagement in Women-Led Podcasts?

This project analyzes public YouTube performance data from women-led podcasts to explore how episode format, topic, duration, titles, and publishing patterns relate to audience reach and interaction.

The project was designed as a small monitoring, evaluation, and impact analysis exercise focused on translating digital engagement data into understandable findings and practical recommendations.

## Interactive Report

View the interactive report here:

**[Women-Led Podcast Performance Report](https://shaniyasolves.github.io/women-led-podcast-analytics/)**

The report includes interactive charts that allow viewers to explore individual episodes, topic performance, and guest-versus-solo differences.

## Research Questions

This analysis focused on three main questions:

1. Do guest and solo episodes perform differently?
2. Are certain topics associated with stronger engagement within individual podcasts?
3. Do episodes with greater reach also receive more interaction per viewer?

## Dataset

The final dataset contains:

- **325 full podcast episodes**
- **8 women-led YouTube podcast channels**
- Up to 50 recent full episodes per channel
- Publicly available video metadata and engagement statistics

Videos under 10 minutes and obvious clips, Shorts, trailers, teasers, or promotional uploads were excluded.

### Podcasts Included

- A Better You Podcast
- A Really Good Cry
- Alex & Annie
- Baby, This Is Keke Palmer
- Good Hang with Amy Poehler
- Rotten Mango
- Sis, Be Real
- The Chic Code

## Metrics

The analysis separates reach from interaction using:

- **Views per day:** adjusts total views for the age of the video
- **Likes per 1,000 views:** measures positive interaction relative to audience size
- **Comments per 1,000 views:** measures discussion relative to audience size
- **Engagement score:** combines standardized reach, likes, and comments within each podcast

Comparisons were primarily made within podcasts because the channels have substantially different audience sizes and publishing histories.

## Methods

Episode data were collected using the YouTube Data API.

Episode topics and formats were classified from titles and public video descriptions. Supporting evidence was checked against the original title or description, and guest-versus-solo comparisons used only labels supported by direct, non-repeated evidence.

The analysis included:

- Descriptive statistics
- Within-podcast topic comparisons
- Mann–Whitney U tests
- Kruskal–Wallis tests
- Holm corrections for multiple comparisons
- Robust linear regression
- Bootstrap confidence intervals
- Interactive Plotly visualizations

An exploratory prediction model was also tested, but it was not emphasized because it did not consistently outperform a podcast-specific baseline.

## Key Findings

### Guest episodes may expand reach

Within **A Really Good Cry**, guest episodes received substantially more median views per day than solo episodes.

This suggests that recognizable guests or guest networks may help introduce a podcast to a broader audience.

### Solo episodes may deepen interaction

Solo episodes received more likes per 1,000 views than guest episodes within the same podcast.

This may indicate that existing viewers feel a stronger connection to episodes centered on the regular host.

### Reach and audience loyalty are different

Episodes with the highest views were not always the episodes receiving the most likes or comments per viewer.

For performance monitoring, reach and interaction should therefore be tracked separately rather than reduced to a single metric.

### Topic performance varied by podcast

Some topics appeared stronger within individual podcasts, but no topic pattern remained statistically reliable across the full set of shows after correcting for multiple comparisons.

Topic rankings should therefore be treated as descriptive signals for future testing rather than universal conclusions.

## Practical Recommendations

Podcast teams could use these findings to support a mixed content strategy:

- Use selected guest episodes to attract new viewers and increase reach.
- Maintain solo or host-focused episodes to support audience connection and loyalty.
- Track views, likes, and comments separately because they represent different forms of performance.
- Compare topics within the same podcast rather than directly comparing channels with different audiences.
- Test content decisions prospectively using consistent measurement periods and predefined performance indicators.

## Limitations

This is an observational and exploratory analysis, so the findings do not establish causation.

Additional limitations include:

- Public YouTube statistics reflect lifetime performance rather than a fixed post-publication window.
- Views per day is an approximate adjustment for video age.
- Podcast audiences and publishing strategies differ substantially.
- Topic and format labels were derived from public titles and descriptions rather than complete transcripts.
- Only one podcast had enough reliable guest and solo labels for a balanced format comparison.
- Some topic categories contained relatively small numbers of episodes.
- External factors such as guest popularity, promotion, current events, and recommendation algorithms were not directly measured.

## Repository Contents

```text
women-led-podcast-analytics/
│
├── index.html
│   └── Interactive GitHub Pages report
│
├── podcast_analysis.py
│   └── Data collection, cleaning, analysis, and visualization code
│
├── README.md
│   └── Project overview and findings
│
└── requirements.txt
    └── Python package requirements

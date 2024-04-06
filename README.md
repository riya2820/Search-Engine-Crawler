# Search-Engine-Crawler

# Overview

This search engine crawler is designed specifically for the Informatics and Computer Science (ICS) domain. It navigates through websites within this domain, extracting and indexing content to facilitate quick and relevant searches. Uses Beautiful Soup library for parsing XML files. The crawler efficiently processes academic papers, articles, and other course content. By calculating TF-IDF scores and cosine similarity, it retireves the most relevant search results.

# Features

1. XML Parsing: Utilizes Beautiful Soup to parse XML files, enabling efficient extraction of structured data like academic papers and documentation.

2. Domain-Specific Crawling: Targets websites and content within the specified domain.

3. TF-IDF Scoring: Implements TF-IDF scoring to evaluate the importance of words within documents relative to a collection, improving search result relevance.

4. Cosine Similarity: Uses cosine similarity measures to compare document similarity, enhancing the crawler's ability to find related documents based on content.


# Prerequisites:
Ensure you have Python 3.x installed on your system

pip install beautifulsoup4 requests numpy scipy
python crawler.py

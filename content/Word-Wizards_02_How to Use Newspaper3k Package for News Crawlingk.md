---
Title: How to Use Newspaper3k Package for News Crawlingk (by Group "Word Wizards")
Date: 2024-03-16 11:13
Category: Progress Report
Tags: Group Word Wizards
---

# How to Use Newspaper3k Package for News Crawling

By Group "Word Wizards"

After we obtain the data set, we need to download the text data of the news from a list of URL links. This blog introduces a simple method to achieve this goal.

## 1 Introduction to the newspaper package

The Newspaper framework is one of the most liked crawler framework on GitHub in Python crawler frameworks, suitable for scraping news web pages. Its operation is very simple and easy to learn, even for beginners who have no understanding of crawlers at all, it is very friendly. It can be easily mastered with simple learning because it does not require consideration of headers, IP proxies, webpage parsing, webpage source code architecture, and other issues. This is its advantage, but also its disadvantage. Not considering these factors may lead to the possibility of being directly rejected when accessing web pages.

However, the Newspaper framework also has limitations, for example the framework is sometimes unstable, and there may be various bugs during the crawling process, such as failure to obtain URLs, news information, etc. However, for those who want to obtain some news corpus, it is worth a try. It is simple, convenient, easy to use, and does not require a deep understanding of professional knowledge about crawlers.

**Functionalities of newspaper:**

- Multi-threaded article download framework
- News URL recognition
- Extracting text from HTML
- Extracting top images from HTML
- Extracting all images from HTML
- Extracting keywords from text
- Extracting summaries from text
- Extracting authors from text
- Extracting Google Trends terms
- Supporting over 10 languages (English, Chinese, German, Arabic, etc.)
- ...



## 2 Example Tutorial of the newspaper package



### to install:

```bash
pip install newspaper3k
```

It is important to note that you should install newspaper3k instead of newspaper because newspaper is the installation package for Python 2, and pip install newspaper will not install correctly. Please use pip install newspaper3k for Python 3 to install it correctly.



### Example of Crawling a Single News Article

To crawl a single news article, you will use the Article package within newspaper. This involves scraping the content of a specific webpage, i.e., when you want to scrape the content of a particular news article, you first need to obtain its URL and then use this single URL as the target to crawl the content.



```python
from newspaper import Article

# Target news URL
url = 'https://www.dw.com/en/chatgpt-sparks-ai-investment-bonanza/a-65368393'

# Create an Article object with the URL and specify the language
news = Article(url, language='en')

# Download the webpage content
news.download()

# Parse the webpage
news.parse()

# Print the news details
print('Title:', news.title)       # News title
print('Text:\n', news.text)        # Main content
print('Authors:', news.authors)   # News authors
print('Keywords:', news.keywords) # News keywords
print('Summary:', news.summary)   # News summary

# Additional information that can be extracted
# print('Top Image:', news.top_image)      # Image URL
# print('Movies:', news.movies)            # Video URLs
# print('Publish Date:', news.publish_date) # Publication date
# print('HTML:', news.html)                 # Webpage source code

```



### result:




```
Title: ChatGPT sparks AI investment bonanza – DW – 04
Text:
The launch of a new branch of artificial intelligence has reenergized the global tech sector. As investors pour billions into AI startups, will tough regulations stop humans from losing control?

The artificial intelligence (AI) gold rush is truly underway. After the release last November of ChatGPT — a game-changing content-generating platform — by research and development company OpenAI, several other tech giants, including Google and Alibaba have raced to release their own versions.

Investors from Shanghai to Silicon Valley are now pouring tens of billions of dollars into startups specializing in so-called generative AI in what some analysts think could become a new dot-com bubble.

The speed at which algorithms rather than humans have been utilized to create high-quality text, software code, music, video and images has sparked concerns that millions of jobs globally could be replaced and the technology may even start controlling humans.

But even Tesla boss Elon Musk, who has repeatedly warned of the dangers of AI, has announced plans to launch a rival to ChatGPT.

Elon Musk told FOX News that his AI version called TruthGPT would be less of a threat than the others Image: FOX News via AP/picture alliance

ChatGPT quickly adopted

Businesses and organizations have quickly discovered ways to easily integrate generative AI into functions like customer services, marketing, and software development. Analysts say the enthusiasm of early adopters will likely have a massive snowball effect.

"The next two to three years will define so much about generative AI," David Foster, cofounder of Applied Data Science Partners, a London-based AI and data consultancy, told DW. "We will talk about it in the same way as the internet itself — how it changes everything that we do as a human species."

......
```





In addition to the commonly used title and main content, you can also retrieve the author, publication time, summary, keywords, image links, video links, and more from a news article using the newspaper package. However, it's important to note that the extraction is not always 100% accurate. In many cases, the author, keywords, and article summary may not be recognized accurately. On the other hand, the publication time, image links, and video links are usually recognizable.



## 3 Crawling Multiple News Articles from the Same Website

Crawling a single news article is inefficient, especially when you need to first find the detailed URLs of the news articles. When you need to crawl a large number of news articles from a website or multiple websites, this approach is clearly not sufficient. The newspaper package allows you to build a news source, encompassing all the news articles from an entire news website, and then index and crawl the website using this news source. Below is an example to illustrate how to crawl multiple news articles using the newspaper library.



1. ### Building the News Source

```python
import newspaper
url = 'https://www.dw.com/en/top-stories/s-9097'    
dw_paper = newspaper.build(url, language='en')  

dw_paper.size()   # To check how many links are available
```



2. ### Article Caching

By default, newspaper caches all previously extracted articles and removes any articles it has already extracted. This feature is useful for preventing duplicate articles and improving extraction speed.

```python
# After some time with the default article caching
url = 'https://www.dw.com/en/top-stories/s-9097'    
dw_paper = newspaper.build(url, language='en') 
print(dw_paper.size())
# Output: 1

# When rebuilding the news source for the same website after some time, only 18 new articles are found, indicating that 18 new/updated articles were added during this period. If you do not want to use this feature, you can use the memoize_articles parameter to opt out.

dw_paper = newspaper.build(url, language='en', memoize_articles=False)  # Build the news source without article caching
```



## 4 Extracting Source Categories

You can extract all the other news website links under the source website using the `category_urls()` method. By extracting these website links, you can establish more news sources and obtain more news articles.

```python
# Extracting Source Categories
for category in dw_paper.category_urls():
    print(category)
```

result:

```
https://learngerman.dw.com
https://corporate.dw.com
https://akademie.dw.com
https://www.dw.com/en/top-stories/s-9097
```



## 5 Extracting Brand and Description of the News Source

Once you have constructed the news source, you can directly view its brand name and the description of the news website.

```python
# Extracting Brand and Description of the News Source
print('Brand:', dw_paper.brand)  # Brand
print('Description:', dw_paper.description)  # Description
```

result:

```
Brand: dw 
Description: News, off-beat stories and analysis of German and international affairs. Dive deeper with our features from Europe and beyond. Watch our 24/7 TV stream.
```



## 6 Viewing News Links

After constructing the news source, you can also examine all the news links under the entire news source along with their quantity.

```python
# Viewing all news links under the news source
for article in dw_paper.articles:
    print(article.url)

print(len(dw_paper.articles))  # Viewing the quantity of news links, which is consistent with dw_paper.size()
```



## 7 Extracting all news content from a news source

By using a for loop, you can load and parse news articles one by one, extracting their content. Because the newspaper library is more of a brute-force web scraping tool, there is a high chance of encountering access denial, so you should incorporate error handling using try-except blocks.

```python
import pandas as pd         # Import the pandas library
news_title = []
news_text = []
news = dw_paper.articles
for i in range(len(news)):    # Loop through the length of news links
    paper = news[i]
    try:
        paper.download()
        paper.parse()
        news_title.append(paper.title)     # Store the news title in a list
        news_text.append(paper.text)       # Store the news content in a list
    except:
        news_title.append('NULL')          # Replace with 'NULL' if unable to access
        news_text.append('NULL')          
        continue

# Create a data table to store the scraped news information
dw_paper_data = pd.DataFrame({'title': news_title, 'text': news_text})
dw_paper_data
```

Output:

> 
>
> | title                                             | text                                              |
> | ------------------------------------------------- | ------------------------------------------------- |
> | Press                                             | New series sheds light on Germany's role in Af... |
> | Traineeship 2024/2025: Introducing the new DW ... | Johan Brockschmidt\n\nIf you hadn't chosen to ... |

From the results, it seems that the newspaper scraping was very successful this time. There were no instances of 404 errors or access denial, and all the news articles were successfully scraped.
---
Title: Scrape News with GoogleNews and Newspaper3k (by Group "Word Wizards")
Date: 2024-03-3 10:12
Category: Progress Report
Tags: Group Word Wizards
---

This demonstration repository illustrates how to use Python to fetch news articles from Google based on given keywords. Subsequently, the fetched articles are processed by the GPT-3.5 model to generate a concise summary of the key points in the articles.


This demonstration utilizes two Python libraries to fetch the latest news articles from Google and retrieve their full content. The first library is **[GoogleNews]**, which enables us to search for news articles based on keywords and retrieve their titles and URLs. The second library is **[Newspaper3k]**, which allows us to download the HTML pages of articles and parse them to extract their textual content.


### Import necessary libraries

```python

from GoogleNews import GoogleNews
import pandas as pd
import requests
from fake_useragent import UserAgent
import newspaper
from newspaper import fulltext
import re

# Define the keyword to search.
keyword = 'Sora'


```





### Get news link from Google News

We search for news related to the Sora topic. The language is English, the region is the United States, and the time is set to one day. We obtain two pages of news data from Google News and save it into a dataframe.

```python
# Perform news scraping from Google and extract the result into Pandas dataframe. 
googlenews = GoogleNews(lang='en', region='US', period='1d', encode='utf-8')
googlenews.clear()
googlenews.search(keyword)
googlenews.get_page(2)
news_result = googlenews.result(sort=True)
news_data_df = pd.DataFrame.from_dict(news_result)
```



But I got an error during the operation:

```
<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1129)>
<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1129)>
```





The error typically occurs when there is an issue with SSL certificate verification. This can happen when the Python environment is unable to verify the SSL certificate of the website you are trying to access.

To resolve this issue, I use the following solutions:

```bash
pip install --upgrade certifi
```

> ps:
>
> If updating `certifi` doesn't solve the issue, you can try setting SSL verification to `False` when making the request. However, this is not recommended for production code as it bypasses SSL certificate verification.
>
> Here's how you can do it:
> ```python
> import ssl
> import certifi
> 
> ssl._create_default_https_context = ssl._create_unverified_context
> 
> # Your scraping code here
> googlenews = GoogleNews(lang='en', region='US', period='1d', encode='utf-8')
> googlenews.clear()
> googlenews.search(keyword)
> googlenews.get_page(2)
> news_result = googlenews.result(sort=True)
> news_data_df = pd.DataFrame.from_dict(news_result)
> ```
>

new version of code:

```python
import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context


googlenews = GoogleNews(lang='en', region='US', period='1d', encode='utf-8')
googlenews.clear()
googlenews.search(keyword)
googlenews.get_page(2)
news_result = googlenews.result(sort=True)
news_data_df = pd.DataFrame.from_dict(news_result)


# Display information of dataframe.
news_data_df.info()
```



here is the output:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20 entries, 0 to 19
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype         
---  ------    --------------  -----         
 0   title     20 non-null     object        
 1   media     20 non-null     object        
 2   date      20 non-null     object        
 3   datetime  20 non-null     datetime64[ns]
 4   desc      20 non-null     object        
 5   link      20 non-null     object        
 6   img       20 non-null     object        
dtypes: datetime64[ns](1), object(6)
memory usage: 1.2+ KB
```



and we can see the dataframe here:

| title                                             | media            | date        | datetime                   | desc | link                                                         | img                                                         |
| ------------------------------------------------- | ---------------- | ----------- | -------------------------- | ---- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| OpenAI's Sora isn't the end of the world... yet   | XDA Developers   | 0 hours ago | 2024-02-27 10:48:56.943676 |      | [Link](https://www.xda-developers.com/sora-isnt-the-end-of-the-world-yet) | ![Image](data:image/gif;base64,R0lGODlhAQABAIAAAP//////...) |
| Japan Moon lander revives                         | The Manila Times | 0 hours ago | 2024-02-27 10:48:56.942748 |      | [Link](https://www.manilatimes.net/2024/02/27/news/nation/japan-moon-lander-revives) | ![Image](data:image/gif;base64,R0lGODlhAQABAIAAAP//////...) |
| A Sora Rival Raises Millions from NEA             | The Information  | 1 hours ago | 2024-02-27 09:48:56.947537 |      | [Link](https://www.theinformation.com/articles/a-sora-rival-raises-millions-from-nea) | ![Image](data:image/gif;base64,R0lGODlhAQABAIAAAP//////...) |
| Dog Adopted After 900 Days in Shelter Returned... | Yahoo            | 1 hours ago | 2024-02-27 09:48:56.946703 |      | [Link](https://www.yahoo.com/lifestyle/dog-adopted-900-days-shelter-returned-owners-184629114.html) | ![Image](data:image/gif;base64,R0lGODlhAQABAIAAAP//////...) |
| Sora's Leap into AI-Driven Video Generation Sp... | BNN Breaking     | 1 hours ago | 2024-02-27 09:48:56.945744 |      | [Link](https://bnnbreaking.com/tech/soras-leap-into-ai-driven-video-generation-sparks-controversy) | ![Image](data:image/gif;base64,R0lGODlhAQABAIAAAP//////...) |

as you can see, there are only some links to the news, but what if we want the full text of the paper?



### Get the full text of the news from the link



To achieve this goal, we use newspaper3k package, one of the most liked crawler framework on GitHub in Python crawler frameworks, suitable for scraping news web pages.



```python
ua = UserAgent()
news_data_df_with_text = []
for index, headers in news_data_df.iterrows():
    news_title = str(headers['title'])
    news_media = str(headers['media'])
    news_update = str(headers['date'])
    news_timestamp = str(headers['datetime'])
    news_description = str(headers['desc'])
    news_link = str(headers['link'])
    print(news_link)
    news_img = str(headers['img'])
    try:
        # html = requests.get(news_link).text
        html = requests.get(news_link, headers={'User-Agent':ua.chrome}, timeout=5).text
        text = fulltext(html)
        print('Text Content Scraped')
    except:
        print('Text Content Scraped Error, Skipped')
        pass
    news_data_df_with_text.append([news_title, news_media, news_update, news_timestamp, 
                                         news_description, news_link, news_img, text])

news_data_with_text_df = pd.DataFrame(news_data_df_with_text, columns=['Title', 'Media', 'Update', 'Timestamp',
                                                                    'Description', 'Link', 'Image', 'Text'])
```



and here is the example result:



| Title                                           | Media            | Update      | Timestamp                  | Description | Link                                                         | Image                                                       | Text                                              |
| ----------------------------------------------- | ---------------- | ----------- | -------------------------- | ----------- | ------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------- |
| OpenAI's Sora isn't the end of the world... yet | XDA Developers   | 0 hours ago | 2024-02-27 10:48:56.943676 |             | [Link](https://www.xda-developers.com/sora-isnt-the-end-of-the-world-yet) | ![Image](data:image/gif;base64,R0lGODlhAQABAIAAAP//////...) | This page is not available!                       |
| Japan Moon lander revives                       | The Manila Times | 0 hours ago | 2024-02-27 10:48:56.942748 |             | [Link](https://www.manilatimes.net/2024/02/27/news/nation/japan-moon-lander-revives) | ![Image](data:image/gif;base64,R0lGODlhAQABAIAAAP//////...) | ERROR 404\n\npage not found\n\nClick here to g... |
| A Sora Rival Raises Millions from NEA           | The Information  | 1 hours ago | 2024-02-27 09:48:56.947537 |             | [Link](https://www.theinformation.com/articles/a-sora-rival-raises-millions-from-nea) | ![Image](data:image/gif;base64,R0lGODlhAQABAIAAAP//////...) | Sorry, we weren't able to find that.\n\nIf you... |










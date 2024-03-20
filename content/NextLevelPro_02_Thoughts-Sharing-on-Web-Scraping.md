---
Title: Thoughts Sharing on Web Scraping (by Group "Next Level Pro") 
Date: 2024-03-19
Category: Progress Report 2
Tags: Next Level Pro
---


When it comes to text analysis, web scraping plays a crucial role in gathering a substantial amount of data from various online sources. Each website has its unique structure and requires us to employ a combination of foundational methods and flexible approaches to extract the desired information. In our previous blog post [Web Scraping for Text Data & Progress Update (by Group "Next Level Pro")](https://buehlmaier.github.io/MFIN7036-student-blog-2024-02/web-scraping-for-text-data-progress-update-by-group-next-level-pro.html), we discussed some of these methods. However, as we delved deeper into our text analysis journey, we encountered new challenges that demanded innovative solutions. In this blog post, we aim to share our experiences, highlighting the problems we faced and the strategies we devised to overcome them. We hope that our insights will inspire and guide others who embark on similar endeavors.

# Code Structure for Extracting Data by Selenium

*Note: Before using the code provided in this section, make sure to import the required packages.*
```python
import time, random
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from bs4 import BeautifulSoup
import re
```
For some websites, the Requests package may not be useful because of JavaScript. Selenium is good to get the same information as we inspect the page in our browser. Here we show an illustration about how to extract the final HTML version of the website using Selenium.

When searching for Nvidia-related news on the CNN website, the manually opened page's URL is as follows : *https://edition.cnn.com/search?q=nvidia&from=0&size=10&page=1&sort=relevance&types=all&section=*. Based on this structure, by changing the "from" and "page" values, we can generate multiple page links and perform batch scraping using a for loop. The sample code is shown below:
```python
HTMLs = []

start = time.time()

for i in range(100):
    from_ = str(10*i)
    page = str(i + 1)
    
    service = Service(executable_path="C:\Program Files\Google\Chrome\Application\chromedriver-win64\chromedriver.exe")
    options = webdriver.ChromeOptions()

    driver = webdriver.Chrome(service=service, options=options)
    url = "https://edition.cnn.com/search?q=nvidia&from=" + from_ + "&size=10&page=" + page +\
        "&sort=relevance&types=all&section="
    driver.get(url)
    time.sleep(20)
    innerHTML = driver.execute_script("return document.body.innerHTML")
    HTMLs.append(innerHTML)
    
    if i%5 == 0:
        print(i/100, 'completed')
    
    time.sleep(random.random())
    
end = time.time()
print((end-start)/60)
```
Then, with the 100 HTML content, by inspecting common structure and feature on every page, we can apply BeautifulSoup and regular expression to get the information we want with another loop.
```python
links = []
titles = []
dates = []
descriptions = []

for html in HTMLs:
    linkpattern = r'headline.*?data-zjs-href="([^"]+)".*?data-zjs-component_position='
    links += re.findall(linkpattern, html)
    
    soup = BeautifulSoup(html, 'html.parser')
    titles += [title.text for title in soup.select('div>span.container__headline-text')]
    
    datepattern = r">\s+(\w{3} \d{2}, \d{4})\s+<"
    dates += re.findall(datepattern, html)
    
    descriptions += [description.text for description in 
                     soup.find_all('div', class_ = 'container__description container_list-images-with-description__description inline-placeholder')]
```

# A Troubleshooting Guide for Scraping Yahoo Finance

In the previous blog, we explored a Python code that scrapes finance news articles from Yahoo Finance and performs sentiment analysis on them. However, we have encountered some issues while trying to collect new data and noticed that the output format for the data is incorrect. In this troubleshooting guide, we will try to address these issues and provide solutions to ensure accurate data retrieval and proper formatting.

## Issue 1: Incorrect Date Format
The Problem is that the output data is displayed in the format of "x days ago" instead of the actual date. To ensure the correct date format, we followed these steps:

1. Extract the publication date of each article. Yahoo Finance typically provides the publication date within the article or as metadata in the HTML.

2. Parse the date string using the appropriate date parsing library, such as datetime.strptime, to convert it into a datetime object.

3. Format the datetime object into "YYYY-MM-DD" by using the strftime method. 

4. Replace the "x days ago" format in the output with the formatted date string.

By implementing these steps, we retrieve accurate and up-to-date data while ensuring the correct date format for your analysis.

## Issue 2: Skipping Articles
When attempting to collect new data, the code skips all articles instead of retrieving them. 
The output displayed “
Skipping S&P Futures5,114.50-14.50(-0.28%): article text not found
Skipping Dow Futures38,643.00-113.00(-0.29%): article text not found
Skipping Nasdaq Futures17,983.00-63.75(-0.35%): article text not found…”

This discrepancy may be attributed to potential changes in the source code of Yahoo Finance. However, it is worth mentioning that we have successfully gathered over a thousand data from the last three months using alternative financial data platforms. For the sake of saving time, we have opted not to utilize data from Yahoo Finance.


# Alternative API to retrieve data from Reuters

Further to our previous blog, we were able to find another API in *https://rapidapi.com/hub* to facilitate our web scraping task. The API can be called easily using `requests`. A simple code snippet is as below:

```python
import requests

url = "https://reuters-business-and-financial-news.p.rapidapi.com/get-articles-by-keyword-name/Nvidia/1/20" #Search results for Nvidia's News

headers = {
	"X-RapidAPI-Key": "8df536404dmshfb46d4239ecbbb9p16f69ejsn8fb103eaab51", #Generated from the API's website
	"X-RapidAPI-Host": "reuters-business-and-financial-news.p.rapidapi.com"
}

response = requests.get(url, headers=headers)

print(response.json())
```

The collected response is a json. We can then parse it using normal string manipulations.

```python
J=response.json()
articlesName=[]
articlesShortDescription=[]
articlesDescription=[]
publishedAt=[]
for i in J['articles']:
    articlesName.append(i['articlesName'])
    articlesShortDescription.append(i['articlesShortDescription'])
    articlesDescription.append(i['articlesDescription'])
    publishedAt.append(i['publishedAt'])
reuters1=pd.DataFrame({'title':articlesName, 'Short':articlesShortDescription, 'Text':articlesDescription, 'Date':publishedAt})
reuters1.head()
```

![Figure 1]({static}/images/NextLevelPro_02_01DFbefore.png)

Here we noticed the `Text` and `Date` column consists of json string instead of the raw information. Therefore we need to further process the columns to obtain the required data.

```python
textdata=[]
for i in range(len(reuters1)):
    sss=J['articles'][i]['articlesDescription'][1:-1]
    ssss=sss.split("},{")
    content=""
    for i in range(len(ssss)):
        if i ==0:
            t=json.loads(ssss[i]+"}")
            if 'content' in t:
                if not t['content'].isupper():
                    content+=t['content']+" "
        elif i ==len(ssss)-1:
            t=json.loads("{"+ssss[i])
            if 'content' in t:
                if not t['content'].isupper():
                    content+=t['content']+" "
        else: 
            t=json.loads("{"+ssss[i]+"}")
            if 'content' in t:
                if not t['content'].isupper():
                    content+=t['content']+" "
    textdata.append(content)
reuters1['@text']=textdata
datetimes=[]
for i in range(len(reuters1)):
    datetimes.append(reuters1.Date[i]['date'])
reuters1['datetime']=datetimes
reuters1['@date']=pd.to_datetime(reuters1['datetime']).dt.date
reuters1=reuters1[['title', 'Short', '@text', '@date']]
reuters1.head()
```

![Figure 2]({static}/images/NextLevelPro_02_02DFafter.png)

With the successful extraction of a substantial amount of data through web scraping, we are now well-equipped to proceed with sentiment analysis. By employing advanced techniques and algorithms, we can uncover patterns and trends that will contribute to our prediction of NVIDIA's volatility. We hope that these experience sharing on web scraping methods prove to be helpful for your endeavors and inspire new insights in your text analysis journey.

---
Title: Literature Review of Potential Project Ideas (by Group "Financial Language Insights")
Date: 2024-03-04 21:01
Category: Blog Report
Tags: Group Financial Language Insights
---

This blog aims to elaborate on the existing literature and GitHub projects that inspire our topic. The topic we choose to investigate is the use of Twitter sentiment analysis to predict stock price movement for the semiconductor industry. The process of narrowing down our final thesis from a large scope of ideas and variables is illustrated as follows:

First, each member put forward their idea. 

1.	Our first idea was inspired by Twitter sentiment analysis: a case study in the automotive industry, (Shukri, 2015) and Twitter Sentiment Analysis and Influence on Stock Performance Using Transfer Entropy and EGARCH Methods (Papana, 2022). The former utilizes sentiment analysis yet aims to predict the sales volume data for different car manufacturers, like Mercedes, BMW, and Audi; the latter employs the same method to predict the stock market index as a whole instead of a specific segment or industry. Combining these two ideas, one group member came up with the idea to adopt sentiment analysis on tweets to predict price returns in a certain industry. Certain merits are encompassed in this idea, including that 1) we can practice the text scraping technique that we learned in class, although we later came to the realization that Twitter offered a built-in API to access their tweets, and that 2) there were relatively more existing literature and GitHub projects in this area.

2.	Another member proposed a more novel idea that delves deep into the realm of corporate finance and micro-economics. He recommended that we use deep learning methods to study the relationship between public opinion and the stock risk premium. A cross-disciplinary essay, published in the Journal of Procedia Computer Science, Public opinion risk on major emergencies: a textual analysis, authored by Zhenghan Xu, served as a source of enlightenment. In his enriching essay, Xu explores the evolution of the “public opinion risk” during the ongoing battle against the COVID-19 epidemic. If the public opinion risk is perceptible and quantifiable, why can’t we use public opinion to gauge, if not explain and predict, the previously mysterious and elusive equity risk premium? 

3.	While the previous two share the same characteristic of focusing on the sentiment analysis function within the Natural Language Processing domain, from a new perspective, a member suggested a new feature of NLP that we can leverage - information extraction. Drawing inspiration from his previous internships where he witnessed analysts manually key in financial data from annual reports into Excel models, he wondered if it is possible to automate this process using NLP to increase efficiency and enhance productivity. Upon researching online, he discovered FinBERT: A Large Language Model for Extracting Information from Financial Text, written by Allen Huang. Although this paper investigates information retrieval from sell-side analyst reports utilizing large language models, whose organizational structures vary from firm to firm and from person to person, we hope to apply NLP to annual reports filed by corporates, which are more structured and easier to fetch. 

After the three members illustrated their ideas in full, our group carefully read through all the literature covered above to gain a comprehensive picture of the strengths, weaknesses, and feasibility of the research topics discussed. For our first idea, we endorsed the idea because it is popular in different variations. Specifically, we talked about two implementation methods. First, we could adopt this method in all industries and select the one industry in which Twitter sentiment data has the highest predicting power; second, we could select an industry that we are interested in. After intense discussion, we decided to choose the semiconductor sector since the advent of the LLM model and its various applications had put the semiconductor industry under the spotlight on Twitter, sparking discussions from not only financial analysts but also tech enthusiasts. This circumstance guarantees ample data that we can analyze and train our model with.  

Although the first topic was deemed suitable by most members, we still covered the second and third ideas. Upon further scrutiny, we considered that it is hard to determine the data sources from which we could calculate the “public opinion” in the second idea. As for the third idea, we found it difficult to navigate the technique of information extraction in NLP since it was not covered in the content of this course. To ensure the accountability and interpretability of the method we used, we decided to focus on sentiment analysis, a realm we feel more confident working in.

## Reference
Huang, A., Wang, H., & Yang, Y. (2023). FinBERT: A Large Language Model for Extracting Information from Financial Text*. Contemporary Accounting Research, 40(2), 806–841. https://doi.org/10.1111/1911-3846.12832 

Mendoza-Urdiales, R. A., Núñez-Mora, J. A., Santillán‐Salgado, R. J., & Herrera, H. V. (2022). Twitter sentiment analysis and influence on stock performance using transfer entropy and EGARCH methods. Entropy, 24(7), 874. https://doi.org/10.3390/e24070874 Shukri, S. E., Yaghi, R. I., 

Aljarah, I., & Alsawalqah, H. (n.d.). Twitter sentiment analysis: A case study in the automotive industry. IEEE. https://doi.org/10.1109/aeect.2015.7360594 

Xu, Z., Zhan, B., & Wang, S. (2023). Public opinion risk on major emergencies: A textual analysis. Procedia Computer Science, 221, 833–838. https://doi.org/10.1016/j.procs.2023.08.058

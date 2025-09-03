# Card Sorting with Fewer Cards and the Same Mental Models? A Re-examination of an Established Practice

## About

This is the official repository for the research paper *"Card Sorting with Fewer Cards and the Same Mental Models? A Re-examination of an Established Practice"*. Paper examines the validity of using randomized subsets of cards in card sorting. It shows how subsets affect data quality, sample size needs, and how personality and cognitive factors shape participants’ mental models.

### Paper citation

Not published yet.

### Contents

* [Dataset](#dataset)
* [Scripts](#scripts)
* [Extended results](#extended-results)
* [Experiment](#experiment)
* [Authors](#authors)
* [License](#license)

## Dataset

Data from 160 participants was collected via [UXtweak's user panel](https://www.uxtweak.com/user-participant-recruitment). [Raw data](./analysis/data/raw/) was processed - final utilized data files include information about respondents and card sorting results:
- [Respondents](/analysis/data/respondents.csv)
  - demographics
  - Big-5 questionnaire items
  - CRT items
  - feedback
  - attention check questions
  - time and created categories
- [Results](/analysis/data/results.csv)
  - cards sorted into categories
  - standardized categories
- [Interactions (Results)](/analysis/data/raw/raw_interactions.csv)
  - card moves, renames and deletions


## Scripts

See below for guidelines on configuring the virtual environment used for data analysis and explanation of key scripts found in the file structure.


### Environment 

All of the scripts located in the analysis folder are written using Python (version 3.12) and other external libraries installed using pip (version 23.2.1). Scripts were executed using jupyter notebooks in VS Code. A [requirements file](./analysis/requirements.txt) is provided for installing dependencies. After installing Python, the below commands can be used in the [analysis](./analysis/) directory to install the environment:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS devices, this process could be slightly different:
```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

This will create your virtual environment and install the dependencies. After the installation, notebooks could be executed in any available IDE (IDEs such as VS Code, in which the environment could be even installed automatically) or using the built-in web environment using this command in the command line (while having activated the virtual environment using `.venv\Scripts\activate` or `source .venv/bin/activate`):
```
jupyter notebook
```


### Files

Following files are present in the [scripts directory](/analysis/scripts):

- [Data preparation](/analysis/scripts/1_data-preparation.ipynb) - reformatting raw export files, calculating scores (Big5, CRT,...)
- Exploratory data analysis of the dataset:
  - [Respondents data](/analysis/scripts/2-1_respondents-analysis.ipynb)
  - [Results data](/analysis/scripts/2-2_results-analysis.ipynb)
  - [Results data - similarity](/analysis/scripts/2-3_result-analysis-similarity.ipynb)
  - [Results data - sample size](/analysis/scripts/2-4_result-analysis-sample-size.ipynb)
  - [Results data - interactions](/analysis/scripts/2-5_result-analysis-interactions.ipynb)
  - [Results data - clustering](/analysis/scripts/2-6_result-analysis-clustering.ipynb)
- [Hypothesis testing](/analysis/scripts/3_statistical-tests.ipynb) - usage of statistical tests
- [Utility methods](/analysis/scripts/utils.py) - imports and many utility functions used during the analysis

## Extended results

Apart from the extended results in the chapter Scripts, [similarity matrices](./analysis/data/matrices/) of several types and [list of cards](./analysis/data/cards/) are provided.


## Experiment

See below for additional experiment information not present in the article.

### Experiment preview
Experiment previews are publicly available for all study variants:
- [Variant E50](https://study.uxtweak.com/cardsort/preview/DGakRNOmP8lKeVnIZuPyX/hmF5nxL2DmZ0PIAb4QpQ1)
- [Variant E30](https://study.uxtweak.com/cardsort/preview/Q6951sQoeNkRRarctlgIO/tMFUy9GSte3WI0djbZFmf)
- [Variant B50](https://study.uxtweak.com/cardsort/preview/B3IbyKq8eaxIHN0fFAiwx/UrEtQ6D1rnYdIyZbCRPuL)
- [Variant B30](https://study.uxtweak.com/cardsort/preview/5Sajz79xUEJ4Q5aNnms78/KT2pKM6piQbeLaMvRIJti)

### Cards

Utilized cards are also available in [CSV format](./analysis/data/cards/).


|Variants E|||
|---|---|---|
| Air Conditioners | Gaming Consoles | Printers |
| Air Purifiers | Graphic Cards | Processors |
| Cameras | Hair Dryers | Radiators |
| Chargers | Headphones | Radios |
| Coffee Makers | Hobs | Scanners |
| Computer Mice | Irons | Shavers |
| Cookers | Juicers | Smart Watches |
| Deep Fryers | Keyboards | Smartphones |
| Desktop Computers | Kitchen Scales | Speakers |
| Dishwashers | Laptops | Stand Mixers |
| Drones | Microphones | Tablets |
| E-Readers | Microwave Ovens | Televisions |
| Electric Kettles | Monitors | Trimmers |
| Electric Toothbrushes | Musical Instruments | Vacuum Cleaners |
| Fans | Ovens | VR Headsets |
| Freezers | PC Games | Washing Machines |
| Fridges | Phone Cases |  |


|Variants B|||
|---|---|---|
| Account Statements | Financial Education Webinars | Online Purchases |
| Balance Transfer | Fixed Deposit | Other Borrowing Options |
| Bereavement Support | Fixed Term Savings | Overdraft Protection |
| Bonds | Fixed-rate Credit Card | Personal Loan Calculator |
| Budgeting Tools and Resources | Fraud Alerts and Monitoring | Personal Pension |
| Car Insurance Renewal | Fund Performance | Personalised Financial Advice |
| Cardless Cash Withdrawal | Google/Apple Pay | Remortgage Options |
| Cheque Payments | Insurance Coverage Calculator | Retirement Planning Calculator |
| Claims | Interest Rates | Rewards Programme |
| Credit Score | International Payments | Savings Options |
| Current Accounts | International Trading | Self-Invested Personal Pension (SIPP) |
| Daily Savings | Investment Accounts | Stock Trading and Shares |
| Data and Privacy Control | ISA Accounts | Student account |
| Debit Cards | Life Insurance Coverage | Student Loan Repayment |
| Debt Management Services | Mobile Banking App | Travel Insurance |
| Digital Confidence | Mortgage Calculator | Trustee Banking |
| Financial Assistance | Mortgage Rates |  |

### Initial questionnaire

#### Demographics

|Question|Question type|Options|
|---|---|---|
|How old are you?|Single choice|18 - 24<br>25 - 34<br>35 - 44<br>45 - 54<br>55|
|Which gender do you identify as?|Single choice|Man<br>Woman<br>Nonbinary|
|What is the highest education level you have completed?|Single choice|None completed<br>Secondary education<br>High school diploma<br>Technical/community college<br>Undergraduate degree<br>Graduate degree<br>Doctoral degree|
|What is your personal income per year, after tax?|Single choice|No income<br>£0 - £9,999<br>£10,000 - £19,999<br>£20,000 - £29,999<br>£30,000 - £39,999<br>£40,000 - £49,999<br>£50,000 - £74,999<br>£75,000 - £100,000<br>More than £100,000|
|How often do you visit e-commerce websites selling electronics? / How often do you visit your bank’s website or other banks’ websites for personal or business use?|Single choice|Multiple times a day<br>At least once a day<br>At least once a week<br>At least once a month<br>Less often|

#### CRT test items

|Question|Question type|Options|
|---|---|---|
|A bat and a ball cost £1.10 in total. The bat costs £1.00 more than the ball. How much does the ball cost?|Single choice|5 pence<br>10 pence<br>9 pence<br>1 pence
|If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?|Single choice|5 minutes<br>100 minutes<br>20 minutes<br>500 minutes
|In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?|Single choice|47 days<br>24 days<br>12 days<br>36 days
|Have you encountered any of the previous three mathematical questions (MQ1 - MQ3) prior to taking this survey?|Single choice|Yes/No
|If you answered "Yes" to the previous question, elaborate on which specific questions you have encountered. Otherwise, feel free to ignore this question.|Free text||

#### Attention check

|Question|Question type|Options|
|---|---|---|
|You may love pizza or ice cream. But when we ask you what you would order in the restaurant, you need to choose a salad. Based on the text above, what is a meal you would order in a restaurant?|Single choice|Pizza<br>Ice cream<br>Spaghetti<br>Salad<br>Fish and chips<br>Hamburger<br>Lasagna<br>Other|

### Final questionnaire items

#### Feedback

|Question|Question type|Scale|
|---|---|---|
|How clear or unclear was it to understand the card labels? |Likert scale|Very unclear to Very clear|
|How easy or difficult was it to categorise the cards?|Likert scale|Very difficult to Very easy|
|How did you find it to maintain concentration while sorting cards?|Likert scale|Very challenging to Very easy|
|How would you rate the amount of time it took you to complete the card sorting activity?|Likert scale|Much too long to Much too short|
|What's your opinion of the number of cards that you were asked to sort?|Likert scale|Too many cards to Too few cards|

#### Big-5 inventory

|Question|Question type|Scale|
|---|---|---|
|I am someone who tends to be quiet.|Likert scale|Disagree strongly to Agree strongly|
|I am someone who is compassionate, has a soft heart.|Likert scale|Disagree strongly to Agree strongly|
|I am someone who tends to be disorganized.|Likert scale|Disagree strongly to Agree strongly|
|I am someone who worries a lot.|Likert scale|Disagree strongly to Agree strongly|
|I am someone who is fascinated by art, music, or literature.|Likert scale|Disagree strongly to Agree strongly|
|I am someone who is dominant, acts as a leader.|Likert scale|Disagree strongly to Agree strongly|
|I am someone who is sometimes rude to others.|Likert scale|Disagree strongly to Agree strongly|
|I am someone who has difficulty getting started on tasks.|Likert scale|Disagree strongly to Agree strongly|
|I am someone who tends to feel depressed, blue.|Likert scale|Disagree strongly to Agree strongly|
|I am someone who has little interest in abstract ideas.|Likert scale|Disagree strongly to Agree strongly|
|I am someone who is full of energy.|Likert scale|Disagree strongly to Agree strongly|
|I am someone who assumes the best about people.|Likert scale|Disagree strongly to Agree strongly|
|I am someone who is reliable, can always be counted on.|Likert scale|Disagree strongly to Agree strongly|
|I am someone who is emotionally stable, not easily upset.|Likert scale|Disagree strongly to Agree strongly|
|I am someone who is original, comes up with new ideas.|Likert scale|Disagree strongly to Agree strongly|


#### Attention check
|Question|Question type|Options|
|---|---|---|
|How many months are there in a year? Even though the correct answer is twelve, make sure to choose the option thirty.|Single choice|1<br>7<br>12<br>30<br>365<br>Other|

#### Other
|Question|Question type|Options|
|---|---|---|
|Is there anything else you’d like to tell us regarding this study?|Free text||


### Study messages

#### Welcome message
Welcome to this research study, and thank you for agreeing to participate. The whole study shouldn't take longer than 20 minutes to complete. Please pay close attention to the instructions to ensure that your participation is valid. If asked to, please answer the questions in as much detail as you can. The more effort you make, the more useful your answers will be to us. 

Please answer honestly, your opinion is what matters the most. Thank you for participating.


#### Instructions
You will be presented with a list of *fifty (50) [thirty (30)]* cards. Follow the instructions on the next screen to sort all cards into categories in a way that makes sense to you. If you change your mind about anything while sorting, you can adjust it to your liking at any time.

This is not a test of your ability, there are no right or wrong answers.
That's it, let's get started!

1. Step
  - Take note of the list of cards on the left.
  - Please, split these cards into categories that feel "right" to you.
  - There's no actual right or wrong way to do it. Simply sort them by intuition.
2. Step
  - Drag a card from the left and drop it here. This will create a new category.
3. Step
  - To rename a new category, click its title and type.
4. Step
  - Drop more cards into a category to add more than one card into the group.
  - To create more categories, drag cards over empty space and drop them there.
  - Once you're finished, click "Finish sort" on the right. Enjoy!


## Authors

### General contact 

Email: 
cardsort.research[AT]gmail.com


**Eduard Kuric**\
He is a researcher and lecturer at [Faculty of Informatics and Information Technologies](https://www.fiit.stuba.sk/), [Slovak University of Technology in Bratislava](https://www.stuba.sk/). His research interests include human-computer interaction analysis, user modeling, personalized web-based systems, and machine learning. Eduard is also the head of the UX Research Department and the founder of [UXtweak](https://www.uxtweak.com/).
- [LinkedIn](https://www.linkedin.com/in/eduard-kuric-b7141280/)
- [Google Scholar](https://scholar.google.com/citations?user=MwjpNoAAAAAJ&hl=en&oi=ao)
- Email: eduard.kuric([AT])stuba.sk

**Peter Demcak**\
Researcher with background in software engineering, whose current topics of interest involve user behavior, human-computer interaction, UX research methods and design practices, and machine learning. Currently occupies the position of a scientific and user experience researcher at [UXtweak](https://www.uxtweak.com/), with focus on research that supports work of UX professionals.
- Email: peter.demcak([AT])uxtweak.com

**Matus Krajcovic**\
User experience researcher at [UXtweak](https://www.uxtweak.com/) and PhD student at [Faculty of Informatics and Information Technologies](https://www.fiit.stuba.sk/), [Slovak University of Technology in Bratislava](https://www.stuba.sk/). Currently focuses on data analysis and research in machine learning use in the field of human-computer interaction.
- Email: matus.krajcovic([AT])uxtweak.com


## License
This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

[![Creative Commons License](https://i.creativecommons.org/l/by-nc/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc/4.0/)

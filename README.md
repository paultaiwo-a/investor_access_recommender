Project Walkthrough




Notebook #1 - Data Pre-processing

In this notebook, the transactions were loaded and cleaned. Duplicate transaction ids were dropped, and only buys were kept.

Buys by year and the assets bought by the most customers were checked. The plots are matplotlib. The cleaned buys were exported as a csv for the next notebook.



Notebook #2 - Modelling

In this notebook, the cleaned buys were used to suggest assets a customer had not bought yet. The two models were popularity and item-item.

Recall@10 and nDCG@10 were compared on buys from 30 May 2022 to 30 Nov 2022. Item-item scored higher on both.




The notebooks use pandas, numpy, and matplotlib. Notebook 1 downloads the FAR-Trans csv if it is not already in data/raw.

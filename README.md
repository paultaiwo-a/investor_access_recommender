Project Walkthrough




clean.py

The transactions are loaded and cleaned. Duplicate transaction ids are dropped, and only buys are kept.

Buys by year and the assets bought by the most customers are plotted with matplotlib. The charts are saved in figures/. The cleaned buys are written to data/processed/buys.csv.



model.py

The cleaned buys are used to suggest assets a customer has not bought yet. The two models are popularity and item-item.

Recall@10 and nDCG@10 are compared on buys from 30 May 2022 to 30 Nov 2022. Item-item scored higher on both.




The scripts use pandas, numpy, and matplotlib. clean.py downloads the FAR-Trans csv if it is not already in data/raw.

```bash
python clean.py
python model.py
```

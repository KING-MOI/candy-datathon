# The Crispy Cup

Datathon entry for the prompt "produce what you think will be the best-selling candy." We used FiveThirtyEight's Ultimate Halloween Candy Power Ranking from Kaggle.

Live site: https://crispy-cup.vercel.app

## The answer

Make a milk chocolate peanut butter cup with crisped rice in the filling. Sell it one cup per wrapper, make it sweet and soft, and price it in the middle of the market.

Our regression forecasts it at a 77% win rate. A random forest, which only trusts trait combinations real candies have tried, forecasts 68%. Either figure puts it in the top 12 of 83 candies.

## What the data is

- 85 rows: 83 candies, plus a dime and a quarter that FiveThirtyEight added as controls.
- Nine yes/no traits per candy: chocolate, fruit, caramel, peanut or almond, nougat, crisped rice or wafer, hard, bar, many per pack.
- Sugar and price, given as percentiles within the set.
- Win rate: the share of about 269,000 random one-on-one matchups the candy won, voted on by 8,371 IP addresses in 2017.

The data has no sales figures, so we use win rate as the stand-in for demand.

## Findings

1. Chocolate adds about 21 points of win rate with the other traits held fixed (p < 0.0001). All 4,000 bootstrap refits agree it helps.
2. Peanut or almond adds about 11 points (p = 0.048, 99% of refits). The top two candies are both Reese's peanut butter cups.
3. Crisped rice adds about 9 points in the regression, with a wide interval because only 7 candies have it. 94% of refits say it helps. The forest scores it as neutral. This is the one bet in our recipe.
4. Chocolate with fruit, the host's example, looks good to an additive model, but only one candy has both (Tootsie Pop). The model forecast it at 66% and it won 49%.
5. Stacking add-ins stops helping after two. The everything-bar recipe's closest real candy, Snickers Crisper, was forecast at 87% and won 60%, the biggest miss of any candy.
6. Price has no clear effect once ingredients are held fixed. Pricier candy wins because it is usually chocolate. Sugar helps a little (+10 points from least to most sugary, p = 0.04).
7. 36 of 83 candies lose to a plain quarter.

## Limits

- Votes are not sales.
- 83 candies is a small sample, so intervals are wide.
- Brand is mixed in. Reese's and Twix beat their recipe forecasts by about 20 points, and the model can't separate peanut butter from the Reese's brand.
- The recipe model explains about a third of the variation on candies it hasn't seen (leave-one-out R² 0.35). It cuts forecast error from 12.4 to 9.4 points.

## Files

- `data/candy-data.csv` is the Kaggle file, unchanged.
- `analysis.py` runs every test and model, writes `build/data.json`, and builds `site/index.html`.
- `template.html` holds the page layout and charts (D3).
- `site/index.html` is the built page. It is one file and works offline.

## Run it

```
python -m venv .venv && .venv/bin/pip install pandas numpy scipy statsmodels scikit-learn
.venv/bin/python analysis.py
open site/index.html
```

## Deploy

```
cd site && vercel deploy --prod
```

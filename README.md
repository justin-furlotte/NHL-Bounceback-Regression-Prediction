# NHL-Bounceback-Regression-Prediction
Using machine learning to predict players who are due to have bounce-back or regression seasons in terms of goals scored. I've also been turning this into a Plotly Dash application to better visualize the data and the results. It is currently deployed here using Heroku (best viewed on desktop!): https://outlierdetectionnhl.herokuapp.com (somewhat computationally heavy so sometimes takes a minute to load).

The idea is to perform supervised outlier detection. The user selects an NHL season (e.g. 2020), and the model then trains on all available player data from 2010 up to the season preceding the selected season (e.g. the 2010 season up to and including the 2019 season are used as training data). A prediction is for goals is then made for each player on the user's selected season. Note that the actual number of goals the player scored is already known -  instead, intuitively, the model is trying to learn how many goals a player "deserved" to score, i.e. a fancier expected goals model. The cross validation mean absolute error is roughly 1.6 goals (depending on the season). 

For each player, the model predicts of the number of goals scored during the user's selected season. We compare this to the actual number of goals they scored. To perform outlier detection, we look for the players who differed most from their expected goals. This is what the `Find` method in `utils.py` does. Players who scored significantly more goals than the model predicted are said to have "overperformed", while players who scored significantly less than the model predicted are said to have "underperformed". The idea is that each player has a baseline number of goals they "deserved" to score, plus some random noise (intuitively, they got lucky/unlucky). A plot like the one below is then generated for each season, showing the top overperformers and underperformers; the plot compares their goal totals in the selected season to their goal totals the following season to see if regression to the mean occured the next year.

![alt text](https://github.com/justin-furlotte/NHL-Bounceback-Regression-Prediction/blob/main/underperformers.png)

![alt text](https://github.com/justin-furlotte/NHL-Bounceback-Regression-Prediction/blob/main/overperformers.png)


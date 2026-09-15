# Kaggle Recommendation Engine Project Write-Up

## 1. Task Definition

The goal of this project was to build a recommendation engine from raw interaction data and produce a ranked list of items for each user. The dataset contains 150,000 interactions between 5,000 users and 2,000 items, with ratings ranging from 1.0 to 5.0 and timestamps covering roughly two years.

I treated the task as a top-N recommendation problem rather than a rating prediction problem. The final system recommends the top 10 items for each user. I chose this setup because the practical goal of a recommendation system is usually not to predict the exact rating a user would give to every item, but to rank a relatively small number of unseen items that are most likely to be relevant. A top-10 list also makes the different models easier to compare using ranking metrics such as Precision@10, Recall@10, MAP@10, and NDCG@10.

An important detail of the dataset is that some columns that might initially look like user or item attributes are not actually stable. For example, `Category` and `Price` can change for the same item, while `Platform` and `Location` are better interpreted as interaction context. For that reason, I avoided treating these fields as permanent user or item properties without evidence.

## 2. Train-Test Split

Because the dataset contains timestamps, I used a temporal split rather than randomly splitting individual interaction rows. The cutoff date was October 1, 2024. Interactions before this date were used for training, while later interactions were used for evaluation.

This was important because a random split could allow information from later interactions to influence a model that is supposed to predict earlier behavior. In a realistic recommendation setting, the model only has access to historical activity at the time recommendations are produced. A temporal split therefore gives a more honest simulation of how the model would behave after being deployed.

I also checked for cold-start cases and repeated user-item interactions across the train and test periods. There were no completely unseen users or items in the test period. However, some user-item pairs in the test set had already appeared in training. Since all recommendation methods exclude items that a user has already seen, I removed these repeated pairs from the final evaluation set. Otherwise, the evaluation could penalize a recommender for correctly following the "do not recommend seen items" rule.

## 3. Models

I started with two simple baselines: Random and Most Popular. The Random baseline recommends randomly selected unseen items, while the Most Popular baseline ranks unseen items according to their total number of training interactions. These baselines were useful because they provided a reference point for deciding whether more complicated models were actually learning anything useful.

The first personalized model was an item-item k-nearest neighbors model. I created a sparse user-item interaction matrix and used cosine similarity to find items with similar interaction patterns. In this setting, a high cosine similarity means that two items tend to be interacted with by similar groups of users. For a specific user, candidate items receive scores based on their similarity to items that the user has already interacted with. Previously seen items are removed from the final ranking.

The second model was a LightGBM classifier. Instead of using only collaborative interaction structure, this model uses engineered user- and item-level features. These include user interaction count, item interaction count, average ratings, user tenure, average prices, and differences between user-level and item-level values. All aggregate features were calculated using training data only so that information from the future did not leak into model training.

For LightGBM, observed user-item interactions were treated as positive examples. Negative examples were created by sampling items that a user had not interacted with. I compared negative sampling ratios of 1:1, 1:3, 1:5, and 1:10. The 1:10 model was eventually selected for the final pipeline because it gave the strongest Recall@10 and Hit Rate@10 among the tested models while maintaining much higher catalogue coverage than the popularity baseline.

I used `LGBMClassifier` instead of `LGBMRanker`. A ranker would also have been a reasonable option, but it requires query groups that identify how many candidate examples belong to each user. The classifier provided a simpler first implementation while still allowing the predicted interaction probabilities to be used directly as ranking scores.

## 4. Results

The final results were:

| Model | Precision@10 | Recall@10 | Hit Rate@10 | MAP@10 | NDCG@10 | Catalogue Coverage | Avg. Popularity Rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.001807 | 0.004559 | 0.017861 | 0.001226 | 0.003048 | 1.0000 | 1001.70 |
| Most Popular | 0.001745 | 0.005026 | 0.017445 | 0.001576 | 0.003427 | 0.0065 | 5.69 |
| Item-item kNN | 0.001620 | 0.004148 | 0.015992 | 0.001138 | 0.002759 | 0.9970 | 1028.28 |
| LightGBM 1:10 | 0.001807 | 0.005122 | 0.018069 | 0.001349 | 0.003233 | 0.6255 | 151.94 |

These results were interesting because the more complex models did not simply dominate the baselines. The Most Popular model achieved the best MAP@10 and NDCG@10. This suggests that global popularity is a strong signal in this dataset. However, it achieved only 0.65% catalogue coverage, which means that it repeatedly recommends a very small group of popular items.

The LightGBM 1:10 model achieved the highest Recall@10 and Hit Rate@10, while covering 62.55% of the catalogue. I therefore selected it as the final model. It does not win every metric, but it gives a better balance between finding relevant items and avoiding an extremely narrow recommendation set.

The item-item kNN model produced very high catalogue coverage, but its ranking accuracy was weaker than both LightGBM and the Most Popular baseline. This shows that producing diverse recommendations by itself is not enough if the items are not ranked effectively for each user.

## 5. What Surprised Me

The most surprising result was how competitive the Most Popular baseline was. Before running the evaluation, I expected the personalized models to clearly outperform a global popularity ranking. Instead, Most Popular achieved the strongest MAP and NDCG.

This result was useful because it showed why simple baselines are important. Without them, it would have been easy to assume that a more complicated recommendation model was automatically better.

Another limitation that became clearer during the project is how the rating signal is used by the LightGBM model. For binary interaction training, both a low-rated and a high-rated observed item are considered positive interactions. For example, an item rated 1.0 and an item rated 5.0 both indicate that an interaction occurred. This loses some information about whether the user actually liked the item. The positive-versus-unseen formulation makes this a reasonable implementation for this project, but it is still a limitation of the final model.

## 6. What I Would Do With Two More Weeks

With two more weeks, I would first separate model selection from final evaluation. In the current project, I compared different LightGBM negative-sampling ratios on the same evaluation period and then selected the 1:10 model based on those results. A cleaner setup would use training data for fitting, a separate validation period for choosing the sampling ratio and other hyperparameters, and a final test period that is only used once to report the final performance.

I would also like to add a simple feedback loop so that the recommendation system can learn from how users react to its previous recommendations. For example, if a recommended item is clicked, rated highly, or the user otherwise shows interest in it, that feedback could be stored as a new interaction and included in the next training cycle. I would not try to build a full real-time online learning system in two weeks, but a small prototype that collects feedback and periodically retrains the model would be realistic. This would make the system more dynamic instead of keeping the model fixed after the initial training stage.

Another improvement would be to use the rating signal more carefully. In the current binary LightGBM formulation, all observed interactions are treated as positive examples, even though a rating of 1.0 clearly represents a different preference from a rating of 5.0. I would experiment with weighting interactions according to rating strength or reformulating the target so that stronger preferences have more influence on the ranking.

I would also test a matrix-factorization method such as implicit ALS and compare it with the item-item kNN model. This could capture latent user and item preferences that are not represented well by local item similarities.

Finally, I would experiment with richer features and a more scalable recommendation architecture. The current LightGBM model can score all unseen items because the catalogue contains only 2,000 items, but this approach would become expensive for a much larger catalogue. A more scalable version could first generate a smaller candidate set using collaborative filtering and then use LightGBM only to rank those candidates.

## 7. What Is Still Wrong

The final system works end to end, but there are still several limitations that would need to be addressed before treating it as a real production recommendation system.

One limitation is how the LightGBM model uses the rating signal. In the current binary formulation, every observed interaction is treated as a positive example. This means that a rating of 1.0 and a rating of 5.0 both indicate a positive interaction, even though they clearly represent very different levels of user preference. Because of this, the model loses some information about whether a user actually liked an item or simply interacted with it.

Another limitation is the cold-start strategy. For a completely new user with no interaction history, the system falls back to Most Popular recommendations. This is a reasonable default, but it is not personalized. A better system could use reliable user or item metadata, onboarding preferences, or contextual information to make more useful recommendations before enough interaction history has been collected.

The current serving function is also designed more as a clean demonstration of inference than as a production service. It loads the stored model and training data from disk when recommendations are requested. This is fine for the scope of the project, but in a real online system these resources would normally stay loaded in memory so that recommendations could be returned much faster.

There are also limitations in the way the final model was selected. Since different LightGBM negative-sampling ratios were compared using the same evaluation period, the reported performance is not based on a completely untouched final test set. This does not prevent the current comparison from being useful, but a separate validation and test period would make the final result more reliable.

Finally, the relatively low ranking metrics show that there is still a lot of room for improving the recommendation quality. The fact that the Most Popular baseline remains very competitive also suggests that the available features and interaction patterns do not provide a very strong personalization signal. The current system is therefore a useful working pipeline, but not a finished recommendation solution.

# Projet Machine Learning El Karoui 2025

Ce projet correspond à mon travail pour la [compétition Kaggle](https://www.kaggle.com/c/m2-proba-finance-2025) proposée dans le cadre du [cours Machine Learning](https://finance.math.upmc.fr/enseignements/1_4_machine_learning/) du [M2 Probabilité Finance (SU-X)](https://finance.math.upmc.fr/). 

"The goal of this competition is to predict "engagement" of tweets from X (Twitter). Engagement is defined as the sum of the number of "retweets" and "likes" which a tweet receives." (*[Kaggle](https://www.kaggle.com/c/m2-proba-finance-2025)*)

## Evaluation
The evaluation metric for this competition is RMSE.

## Approche choisie
Après une analyse statistique (I) des données du dataset (train+test, train uniquement pour la target=engagement), mise en place d'une pipeline de transformation des données (II-1) qui donnera lieu au choix d'un modèle de régression (II-2) et l'optimisation de ses hyperparamètres (II-3) suivi enfin d'une pipeline d'execution directe pour la génération de l'output (III). Les sections de ce readme résument les choix faits pour la partie III.


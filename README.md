# MASTER PROJECT M1 - Cross-asset machine-learning-based performance simulator
## Introduction

Ce projet utilise l'apprentissage automatique pour prédire des indicateurs financiers à partir de données historiques. L'objectif est d'utiliser ces prédictions pour informer des décisions d'investissement à moyen terme, en se concentrant sur des insights quotidiens.

## Summary

1. [Aperçu du Projet](#aperçu-du-projet)
2. [Indicateurs Utilisés](#indicateurs-utilisés)
3. [Collecte de Données](#collecte-de-données)
4. [Prétraitement des Données](#prétraitement-des-données)
5. [Modélisation et Entraînement](#modélisation-et-entrainement)
6. [Évaluation du Modèle](#évaluation-du-modèle)
7. [Stratégie d'Investissement](#stratégie-dinvestissement)
8. [Déploiement](#déploiement)
9. [Gestion des Risques](#gestion-des-risques)
10. [Conclusion](#conclusion)

## Project overview

Ce projet a pour but de développer un modèle prédictif pour des indicateurs financiers clés, en utilisant des techniques d'apprentissage automatique. Les prédictions obtenues seront utilisées pour élaborer une stratégie d'investissement informée.

## Asset classes, financial assets & financial indicators 

Pour ce projet, nous utilisons les indicateurs financiers suivants :
- **Prix de Clôture des Actions** : Suivi quotidien du prix de clôture des actions.
- **Volumes d'Échange** : Nombre de titres échangés quotidiennement.
- **Volatilité (ATR)** : Mesure de la volatilité basée sur l'Average True Range.
- **Bollinger Bands** : Indicateurs de volatilité utilisant des moyennes mobiles et des écarts-types.
- **Indice de Force Relative (RSI)** : Mesure de la vitesse et du changement des mouvements de prix.
- **MACD (Moving Average Convergence Divergence)** : Indicateur de momentum basé sur la différence entre deux moyennes mobiles exponentielles.

## Data retrievement and storage

Les données historiques pour les indicateurs mentionnés ci-dessus sont collectées à partir de sources fiables comme Alpha Vantage, Yahoo Finance, et Quandl.

## Data cleaning and processing

Le prétraitement des données comprend les étapes suivantes :
- Nettoyage des données : Gestion des valeurs manquantes et des outliers.
- Normalisation des données : Mise à l'échelle des données pour uniformiser les échelles.
- Ingénierie des caractéristiques : Création de nouvelles caractéristiques telles que les moyennes mobiles et les indicateurs techniques.

## Modeling and training

Pour la modélisation, nous utilisons plusieurs algorithmes d'apprentissage automatique, notamment :
- Régression Linéaire
- Arbres de Décision
- Forêts Aléatoires
- Réseaux de Neurones à Long Court Terme (LSTM) pour les séries temporelles

Le modèle est entraîné sur un jeu de données de formation et évalué sur un jeu de données de test.

## Model evaluation

L'évaluation du modèle est réalisée à l'aide de métriques telles que :
- Erreur Absolue Moyenne (MAE)
- Erreur Quadratique Moyenne (MSE)
- Coefficient de Détermination (R²)

## Investment strategies

Les prédictions générées par le modèle sont utilisées pour élaborer une stratégie d'investissement basée sur des signaux d'achat/vente.

## Release

Un système automatisé est mis en place pour :
- Mettre à jour régulièrement les données
- Réentraîner le modèle avec de nouvelles données
- Générer des prédictions quotidiennes

## Risk monitoring

La gestion des risques est assurée par :
- La diversification des investissements
- La mise en place de stops-loss
- L'évaluation continue des performances du modèle

## Conclusion

Ce projet démontre l'utilisation de l'apprentissage automatique pour la prédiction d'indicateurs financiers et l'élaboration de stratégies d'investissement informées. Des améliorations futures incluent l'intégration de nouveaux indicateurs et l'optimisation continue des modèles.

## Authors

- **Matthieu VICHET**
- **Elsa PAYA**
- **Mattéo LO RE**
- **Antoine GUILLAUME**
- **Clément OLLIVIER**

## License

Ce projet est sous licence [Nom de la Licence].

## Thanks

Nous remercions [Noms ou Organisations] pour leur soutien et leurs contributions.


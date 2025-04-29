# Projets- Alghoritheme ml R
# Charger les librairies nécessaires
library(tidymodels)
library(modeldata)

# Charger la base de données cells
data("cells")

# Explorer les données
glimpse(cells)
summary(cells)

# Diviser les données en ensembles d'entraînement et de test
set.seed(123)
data_split <- initial_split(cells, prop = 0.75, strata = class)
TRAIN <- training(data_split)
TEST <- testing(data_split)

# Créer une recette de prétraitement
recette <- recipe(class ~ ., data = TRAIN) %>%
  step_dummy(all_nominal_predictors()) %>%  # Encoder les variables catégorielles
  step_zv(all_predictors()) %>%            # Supprimer les variables à variance nulle         
  step_normalize(all_predictors())         # Normaliser les variable

#A. Régression Logistique
# Définir le modèle
modele_logistic <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Créer un workflow
workflow_logistic <- workflow() %>%
  add_recipe(recette) %>%
  add_model(modele_logistic)

# Entraîner le modèle
fit_logistic <- fit(workflow_logistic, data = TRAIN)

# Prédire sur l'ensemble de test
predictions_logistic <- predict(fit_logistic, TEST) %>%
  bind_cols(predict(fit_logistic, TEST, type = "prob")) %>%
  bind_cols(TEST %>% select(class))

# Évaluer le modèle
metrics_logistic <- predictions_logistic %>%
  metrics(truth = class, estimate = .pred_class, .pred_PS)

# Afficher les métriques
print(metrics_logistic)

#B. Arbre de Décision
# Définir le modèle
modele_tree <- decision_tree() %>%
  set_engine("rpart") %>%
  set_mode("classification")

# Créer un workflow
workflow_tree <- workflow() %>%
  add_recipe(recette) %>%
  add_model(modele_tree)

# Entraîner le modèle
fit_tree <- fit(workflow_tree, data = TRAIN)

# Prédire sur l'ensemble de test
predictions_tree <- predict(fit_tree, TEST) %>%
  bind_cols(predict(fit_tree, TEST, type = "prob")) %>%
  bind_cols(TEST %>% select(class))

# Évaluer le modèle
metrics_tree <- predictions_tree %>%
  metrics(truth = class, estimate = .pred_class, .pred_PS)

# Afficher les métriques
print(metrics_tree)

#C. Forêt Aléatoire
# Définir le modèle

library(tidymodels)
library(ranger)  # Charger le package ranger
modele_rf <- rand_forest() %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Créer un workflow
workflow_rf <- workflow() %>%
  add_recipe(recette) %>%
  add_model(modele_rf)

# Entraîner le modèle
fit_rf <- fit(workflow_rf, data = TRAIN)

# Prédire sur l'ensemble de test
predictions_rf <- predict(fit_rf, TEST) %>%
  bind_cols(predict(fit_rf, TEST, type = "prob")) %>%
  bind_cols(TEST %>% select(class))

# Évaluer le modèle
metrics_rf <- predictions_rf %>%
  metrics(truth = class, estimate = .pred_class, .pred_PS)

# Afficher les métriques
print(metrics_rf)

#D. SVM (Machine à Vecteurs de Support)
# Définir le modèle
library(kernlab)
modele_svm <- svm_rbf() %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# Créer un workflow
workflow_svm <- workflow() %>%
  add_recipe(recette) %>%
  add_model(modele_svm)

# Entraîner le modèle
fit_svm <- fit(workflow_svm, data = TRAIN)

# Prédire sur l'ensemble de test
predictions_svm <- predict(fit_svm, TEST) %>%
  bind_cols(predict(fit_svm, TEST, type = "prob")) %>%
  bind_cols(TEST %>% select(class))

# Évaluer le modèle
metrics_svm <- predictions_svm %>%
  metrics(truth = class, estimate = .pred_class, .pred_PS)

# Afficher les métriques
print(metrics_svm)


# Comparer les métriques des modèles
comparaison_metriques <- bind_rows(
  logistic = metrics_logistic,
  tree = metrics_tree,
  rf = metrics_rf,
  svm = metrics_svm,
  .id = "modele"
)

# Afficher la comparaison
print(comparaison_metriques)

# Trouver le modèle avec la meilleure AUC-ROC
meilleur_modele <- comparaison_metriques %>%
  filter(.metric == "roc_auc") %>%  # Filtrer pour la métrique AUC-ROC
  arrange(desc(.estimate)) %>%      # Trier par .estimate (AUC-ROC) en ordre décroissant
  slice(1) %>%                      # Sélectionner la première ligne (meilleur modèle)
  pull(modele)                      # Extraire le nom du modèle

# Afficher le meilleur modèle
print(paste("Le meilleur modèle est :", meilleur_modele))


# Tracer la courbe ROC pour chaque modèle
roc_curves <- bind_rows(
  logistic = predictions_logistic,
  tree = predictions_tree,
  rf = predictions_rf,
  svm = predictions_svm,
  .id = "modele"
) %>%
  group_by(modele) %>%
  roc_curve(truth = class, .pred_PS)

# Afficher les courbes ROC
autoplot(roc_curves)


# Prédire sur l'ensemble de test avec le meilleur modèle
predictions_test <- predict(fit_rf, TEST) %>%
  bind_cols(predict(fit_rf, TEST, type = "prob")) %>%
  bind_cols(TEST %>% select(class))

# Calculer les métriques de performance
metrics_test <- predictions_test %>%
  metrics(truth = class, estimate = .pred_class, .pred_PS)

# Afficher les métriques
print(metrics_test)

# Tracer la courbe ROC
roc_curve_test <- predictions_test %>%
  roc_curve(truth = class, .pred_PS)

autoplot(roc_curve_test)

# Model Card - Customer Journey Intelligence

## Usage prevu
Estimer la probabilite qu'une session web aboutisse a un achat, pour prioriser les visiteurs a
fort potentiel (ciblage marketing, personnalisation) et comprendre les facteurs de conversion.
Usage d'aide a la decision, pas de decision automatique irreversible.

## Donnees
Google Analytics Sample (Google Merchandise Store), `bigquery-public-data.google_analytics_sample`
(aout 2016 - juillet 2017). Un generateur synthetique reproduit la structure (sessions + hits +
clients multi-visites) pour le developpement et les tests sans authentification.

## Modeles
LightGBM regularise (early stopping), probabilites recalibrees par regression isotonic.
Deux variantes :
- `insession` : features des premiers hits de la visite (pre-achat).
- `customer` : features d'historique construites uniquement a partir des sessions anterieures.

Evaluation par **split temporel** (train sur le passe, test sur le futur) et metriques adaptees au
desequilibre : **PR-AUC**, ROC-AUC, Brier, lift par decile.

## Prevention de la fuite de donnees
Les totaux de session (pageviews, hits, time_on_site) sont **exclus** : ils sont mecaniquement
gonfles par l'achat. Les features d'historique sont strictement anterieures a la session notee.
Les pages panier/checkout/achat sont exclues de la fenetre de debut de parcours.

## Limites
- Les parcours GA sample viennent d'un seul site (Merchandise Store, tres US-centre) ; les facteurs
  (ex. pays dominant) ne se transposent pas tels quels a un autre site.
- L'attribution Markov premier ordre ignore les effets d'ordre superieur et hors-ligne.
- Probabilites a re-calibrer et seuil a fixer selon un cout/gain metier reel avant tout usage.

## Ethique
Donnees anonymisees, pas d'attribut sensible utilise. Un score de propension ne doit pas servir a
exclure ou desavantager un utilisateur.

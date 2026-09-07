-- Clickstream ordonne (un hit PAGE par ligne).
-- page_type : coarse (pour les features tabulaires) ; page_token : riche (pagePathLevel1, pour le GRU).
-- Parametres : {date_min} {date_max}.
SELECT
  CONCAT(fullVisitorId, '-', CAST(visitId AS STRING)) AS session_id,
  fullVisitorId,
  h.hitNumber AS hit_index,
  CASE h.eCommerceAction.action_type
    WHEN '2' THEN 'product' WHEN '3' THEN 'cart'
    WHEN '5' THEN 'checkout' WHEN '6' THEN 'purchase'
    ELSE 'browse'
  END AS page_type,
  h.eCommerceAction.action_type AS action_type,
  IFNULL(NULLIF(h.page.pagePathLevel1, ''), 'other') AS page_token,
  IFNULL(h.time, 0) / 1000 AS seconds
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`, UNNEST(hits) AS h
WHERE _TABLE_SUFFIX BETWEEN '{date_min}' AND '{date_max}'
  AND h.type = 'PAGE'

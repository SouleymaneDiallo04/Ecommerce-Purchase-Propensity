-- Une ligne par session (features de contexte + totaux + label converted).
-- Parametres : {date_min} {date_max} (format YYYYMMDD).
SELECT
  fullVisitorId,
  CONCAT(fullVisitorId, '-', CAST(visitId AS STRING)) AS session_id,
  _TABLE_SUFFIX AS date,
  channelGrouping,
  trafficSource.source     AS source,
  trafficSource.medium     AS medium,
  device.deviceCategory    AS device_category,
  device.operatingSystem   AS os,
  CAST(device.isMobile AS INT64) AS is_mobile,
  geoNetwork.country       AS country,
  geoNetwork.subContinent  AS sub_continent,
  IFNULL(visitNumber, 1)        AS visit_number,
  IFNULL(totals.newVisits, 0)   AS new_visit,
  IFNULL(totals.pageviews, 0)   AS pageviews,
  IFNULL(totals.hits, 0)        AS hits,
  IFNULL(totals.timeOnSite, 0)  AS time_on_site,
  IFNULL(totals.bounces, 0)     AS bounces,
  EXTRACT(HOUR FROM TIMESTAMP_SECONDS(visitStartTime)) AS hour,
  IF(IFNULL(totals.transactions, 0) >= 1, 1, 0) AS converted
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '{date_min}' AND '{date_max}'

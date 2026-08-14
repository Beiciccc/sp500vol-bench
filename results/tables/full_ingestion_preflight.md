# Full S&P 500 Ingestion Preflight

- Generated UTC: 2026-06-06T18:01:39.535350+00:00
- EDGAR ingestion: not run
- CRSP market availability: checked

## Summary

| metric | value |
| --- | --- |
| config | configs/data/full.yaml |
| membership_table | data/universe/sp500_membership.parquet |
| date_range | 2010-01-01 to 2025-12-31 |
| unique_ticker_count | 886 |
| unique_ticker_cik_count | 901 |
| cik_missing_count | 0 |
| membership_interval_count | 911 |
| checked_interval_count | 911 |
| crsp_failed_or_partial_count | 19 |
| risky_historical_renamed_delisted_count | 384 |

## Active Universe Size by Year

| year | as_of | active_tickers |
| --- | --- | --- |
| 2010 | 2010-12-31 | 500 |
| 2011 | 2011-12-31 | 499 |
| 2012 | 2012-12-31 | 500 |
| 2013 | 2013-12-31 | 500 |
| 2014 | 2014-12-31 | 502 |
| 2015 | 2015-12-31 | 504 |
| 2016 | 2016-12-31 | 505 |
| 2017 | 2017-12-31 | 504 |
| 2018 | 2018-12-31 | 505 |
| 2019 | 2019-12-31 | 505 |
| 2020 | 2020-12-31 | 505 |
| 2021 | 2021-12-31 | 505 |
| 2022 | 2022-12-31 | 503 |
| 2023 | 2023-12-31 | 504 |
| 2024 | 2024-12-31 | 504 |
| 2025 | 2025-12-31 | 503 |

## Failed or Partial Market-Data Tickers

| ticker | cik | member_from | member_to | interval_count | status | rows | expected_business_days | coverage_ratio | first_date | last_date | issue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AA | 0001675149 | 2016-11-01 | 2016-11-01 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| ADNT | 0001670541 | 2016-10-31 | 2016-10-31 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| APY | 0001723089 | 2018-05-09 | 2018-05-09 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| ASIX | 0001673985 | 2016-10-03 | 2016-10-03 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| BIVV | 0001681689 | 2017-02-02 | 2017-02-02 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| CVET | 0001752836 | 2019-02-08 | 2019-02-08 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| FCPT | 0001650132 | 2015-11-10 | 2015-11-10 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| FTRE | 0001965040 | 2023-07-03 | 2023-07-05 | 1 | partial | 1 | 3 | 0.333 | 2023-07-05 | 2023-07-05 | low business-day coverage (33.3%) |
| GTX | 0001735707 | 2018-10-01 | 2018-10-01 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| KLG | 0001959348 | 2023-10-02 | 2023-10-02 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| KTB | 0001760965 | 2019-05-23 | 2019-05-23 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| LW | 0001679273 | 2016-11-10 | 2016-11-10 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| MBC | 0001941365 | 2022-12-15 | 2022-12-16 | 1 | partial | 1 | 2 | 0.500 | 2022-12-16 | 2022-12-16 | low business-day coverage (50.0%) |
| NGVT | 0001653477 | 2016-05-16 | 2016-05-16 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| PHIN | 0001968915 | 2023-07-05 | 2023-07-05 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| QCP | 0001677203 | 2016-11-01 | 2016-11-01 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| REZI | 0001740332 | 2018-10-29 | 2018-10-29 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| VSM | 0001660690 | 2016-10-03 | 2016-10-03 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| YUMC | 0001673358 | 2016-11-01 | 2016-11-01 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |

## Risky Historical / Renamed / Delisted Tickers

| ticker | risk_reasons |
| --- | --- |
| AA | membership_ended_before_config_end, crsp_failed |
| AAL | membership_ended_before_config_end |
| AAP | membership_ended_before_config_end |
| ABC | membership_ended_before_config_end |
| ABMD | membership_ended_before_config_end |
| ACE | membership_ended_before_config_end |
| ACS | membership_ended_before_config_end |
| ACT | membership_ended_before_config_end |
| ADNT | membership_ended_before_config_end, crsp_failed |
| ADS | membership_ended_before_config_end |
| ADT | membership_ended_before_config_end |
| AET | membership_ended_before_config_end |
| AGL | membership_ended_before_config_end |
| AGN | membership_ended_before_config_end |
| AIRC | membership_ended_before_config_end |
| AIV | membership_ended_before_config_end |
| AKS | membership_ended_before_config_end |
| ALK | membership_ended_before_config_end |
| ALTR | membership_ended_before_config_end |
| ALXN | membership_ended_before_config_end |
| AMG | membership_ended_before_config_end |
| AMTM | membership_ended_before_config_end |
| AN | membership_ended_before_config_end |
| ANDV | membership_ended_before_config_end |
| ANF | membership_ended_before_config_end |
| ANR | membership_ended_before_config_end |
| ANSS | membership_ended_before_config_end |
| ANTM | membership_ended_before_config_end |
| APC | membership_ended_before_config_end |
| APOL | membership_ended_before_config_end |
| APY | membership_ended_before_config_end, crsp_failed |
| ARG | membership_ended_before_config_end |
| ARNC | membership_ended_before_config_end |
| ASIX | membership_ended_before_config_end, crsp_failed |
| ATI | membership_ended_before_config_end |
| ATVI | membership_ended_before_config_end |
| AVP | membership_ended_before_config_end |
| AYE | membership_ended_before_config_end |
| AYI | membership_ended_before_config_end |
| BBBY | membership_ended_before_config_end |
| BBT | membership_ended_before_config_end |
| BBWI | membership_ended_before_config_end |
| BCR | membership_ended_before_config_end |
| BDK | membership_ended_before_config_end |
| BEAM | membership_ended_before_config_end |
| BHF | membership_ended_before_config_end |
| BHGE | membership_ended_before_config_end |
| BHI | membership_ended_before_config_end |
| BIG | membership_ended_before_config_end |
| BIO | membership_ended_before_config_end |
| BIVV | membership_ended_before_config_end, crsp_failed |
| BJS | membership_ended_before_config_end |
| BLL | membership_ended_before_config_end |
| BMC | membership_ended_before_config_end |
| BMS | membership_ended_before_config_end |
| BNI | membership_ended_before_config_end |
| BRCM | membership_ended_before_config_end |
| BTU | membership_ended_before_config_end |
| BWA | membership_ended_before_config_end |
| BXLT | membership_ended_before_config_end |
| CA | membership_ended_before_config_end |
| CAM | membership_ended_before_config_end |
| CBE | membership_ended_before_config_end |
| CBG | membership_ended_before_config_end |
| CBS | membership_ended_before_config_end |
| CCE | membership_ended_before_config_end |
| CDAY | membership_ended_before_config_end |
| CE | membership_ended_before_config_end |
| CELG | membership_ended_before_config_end |
| CEPH | membership_ended_before_config_end |
| CERN | membership_ended_before_config_end |
| CFN | membership_ended_before_config_end |
| CHK | membership_ended_before_config_end |
| CLF | membership_ended_before_config_end |
| CMA | membership_ended_before_config_end |
| CMCSK | membership_ended_before_config_end |
| CNX | membership_ended_before_config_end |
| COG | membership_ended_before_config_end |
| COH | membership_ended_before_config_end |
| COL | membership_ended_before_config_end |
| COTY | membership_ended_before_config_end |
| COV | membership_ended_before_config_end |
| CPGX | membership_ended_before_config_end |
| CPRI | membership_ended_before_config_end |
| CPWR | membership_ended_before_config_end |
| CSC | membership_ended_before_config_end |
| CSRA | membership_ended_before_config_end |
| CTL | membership_ended_before_config_end |
| CTLT | membership_ended_before_config_end |
| CTXS | membership_ended_before_config_end |
| CVC | membership_ended_before_config_end |
| CVET | membership_ended_before_config_end, crsp_failed |
| CVH | membership_ended_before_config_end |
| CXO | membership_ended_before_config_end |
| CZR | membership_ended_before_config_end |
| DF | membership_ended_before_config_end |
| DFS | membership_ended_before_config_end |
| DISCA | membership_ended_before_config_end |
| DISCK | membership_ended_before_config_end |
| DISH | membership_ended_before_config_end |
| DLPH | membership_ended_before_config_end |
| DNB | membership_ended_before_config_end |
| DNR | membership_ended_before_config_end |
| DO | membership_ended_before_config_end |
| DPS | membership_ended_before_config_end |
| DRE | membership_ended_before_config_end |
| DTV | membership_ended_before_config_end |
| DV | membership_ended_before_config_end |
| DWDP | membership_ended_before_config_end |
| DXC | membership_ended_before_config_end |
| EK | membership_ended_before_config_end |
| EMC | membership_ended_before_config_end |
| EMN | membership_ended_before_config_end |
| ENDP | membership_ended_before_config_end |
| ENPH | membership_ended_before_config_end |
| EP | membership_ended_before_config_end |
| ERTS | membership_ended_before_config_end |
| ESRX | membership_ended_before_config_end |
| ESV | membership_ended_before_config_end |
| ETFC | membership_ended_before_config_end |
| ETSY | membership_ended_before_config_end |
| EVHC | membership_ended_before_config_end |
| FB | membership_ended_before_config_end |
| FBHS | membership_ended_before_config_end |
| FBIN | membership_ended_before_config_end |
| FCPT | membership_ended_before_config_end, crsp_failed |
| FDO | membership_ended_before_config_end |
| FHN | membership_ended_before_config_end |
| FI | membership_ended_before_config_end |
| FII | membership_ended_before_config_end |
| FL | membership_ended_before_config_end |
| FLIR | membership_ended_before_config_end |
| FLR | membership_ended_before_config_end |
| FLS | membership_ended_before_config_end |
| FLT | membership_ended_before_config_end |
| FMC | membership_ended_before_config_end |
| FO | membership_ended_before_config_end |
| FOSL | membership_ended_before_config_end |
| FPL | membership_ended_before_config_end |
| FRC | membership_ended_before_config_end |
| FRX | membership_ended_before_config_end |
| FTI | membership_ended_before_config_end |
| FTR | membership_ended_before_config_end |
| FTRE | membership_ended_before_config_end, crsp_partial |
| GAS | membership_ended_before_config_end |
| GCI | membership_ended_before_config_end |
| GENZ | membership_ended_before_config_end |
| GGP | membership_ended_before_config_end |
| GHC | membership_ended_before_config_end |
| GMCR | membership_ended_before_config_end |
| GME | membership_ended_before_config_end |
| GNW | membership_ended_before_config_end |
| GPS | membership_ended_before_config_end |
| GR | membership_ended_before_config_end |
| GT | membership_ended_before_config_end |
| GTX | membership_ended_before_config_end, crsp_failed |
| HAR | membership_ended_before_config_end |
| HBI | membership_ended_before_config_end |
| HCBK | membership_ended_before_config_end |
| HCN | membership_ended_before_config_end |
| HCP | membership_ended_before_config_end |
| HES | membership_ended_before_config_end |
| HFC | membership_ended_before_config_end |
| HNZ | membership_ended_before_config_end |
| HOG | membership_ended_before_config_end |
| HOT | membership_ended_before_config_end |
| HP | membership_ended_before_config_end |
| HRB | membership_ended_before_config_end |
| HRS | membership_ended_before_config_end |
| HSP | membership_ended_before_config_end |
| IGT | membership_ended_before_config_end |
| IILG | membership_ended_before_config_end |
| ILMN | membership_ended_before_config_end |
| INFO | membership_ended_before_config_end |
| IPG | membership_ended_before_config_end |
| IPGP | membership_ended_before_config_end |
| ITT | membership_ended_before_config_end |
| JAVA | membership_ended_before_config_end |
| JCP | membership_ended_before_config_end |
| JDSU | membership_ended_before_config_end |
| JEC | membership_ended_before_config_end |
| JEF | membership_ended_before_config_end |
| JNPR | membership_ended_before_config_end |
| JNS | membership_ended_before_config_end |
| JOY | membership_ended_before_config_end |
| JOYG | membership_ended_before_config_end |
| JWN | membership_ended_before_config_end |
| K | membership_ended_before_config_end |
| KFT | membership_ended_before_config_end |
| KG | membership_ended_before_config_end |
| KLG | membership_ended_before_config_end, crsp_failed |
| KMX | membership_ended_before_config_end |
| KORS | membership_ended_before_config_end |
| KRFT | membership_ended_before_config_end |
| KSS | membership_ended_before_config_end |
| KSU | membership_ended_before_config_end |
| KTB | membership_ended_before_config_end, crsp_failed |
| LB | membership_ended_before_config_end |
| LEG | membership_ended_before_config_end |
| LIFE | membership_ended_before_config_end |
| LKQ | membership_ended_before_config_end |
| LLL | membership_ended_before_config_end |
| LLTC | membership_ended_before_config_end |
| LM | membership_ended_before_config_end |
| LNC | membership_ended_before_config_end |
| LO | membership_ended_before_config_end |
| LSI | membership_ended_before_config_end |
| LTD | membership_ended_before_config_end |
| LUK | membership_ended_before_config_end |
| LUMN | membership_ended_before_config_end |
| LVLT | membership_ended_before_config_end |
| LW | crsp_failed |
| LXK | membership_ended_before_config_end |
| M | membership_ended_before_config_end |
| MAC | membership_ended_before_config_end |
| MAT | membership_ended_before_config_end |
| MBC | membership_ended_before_config_end, crsp_partial |
| MDP | membership_ended_before_config_end |
| MEE | membership_ended_before_config_end |
| MFE | membership_ended_before_config_end |
| MHFI | membership_ended_before_config_end |
| MHK | membership_ended_before_config_end |
| MHP | membership_ended_before_config_end |
| MHS | membership_ended_before_config_end |
| MI | membership_ended_before_config_end |
| MIL | membership_ended_before_config_end |
| MJN | membership_ended_before_config_end |
| MKTX | membership_ended_before_config_end |
| MMI | membership_ended_before_config_end |
| MNK | membership_ended_before_config_end |
| MOLX | membership_ended_before_config_end |
| MON | membership_ended_before_config_end |
| MOT | membership_ended_before_config_end |
| MRO | membership_ended_before_config_end |
| MUR | membership_ended_before_config_end |
| MWV | membership_ended_before_config_end |
| MWW | membership_ended_before_config_end |
| MXIM | membership_ended_before_config_end |
| MYL | membership_ended_before_config_end |
| NAVI | membership_ended_before_config_end |
| NBL | membership_ended_before_config_end |
| NBR | membership_ended_before_config_end |
| NE | membership_ended_before_config_end |
| NFX | membership_ended_before_config_end |
| NGVT | membership_ended_before_config_end, crsp_failed |
| NKTR | membership_ended_before_config_end |
| NLOK | membership_ended_before_config_end |
| NLSN | membership_ended_before_config_end |
| NOV | membership_ended_before_config_end |
| NOVL | membership_ended_before_config_end |
| NSM | membership_ended_before_config_end |
| NU | membership_ended_before_config_end |
| NVLS | membership_ended_before_config_end |
| NWL | membership_ended_before_config_end |
| NYT | membership_ended_before_config_end |
| NYX | membership_ended_before_config_end |
| ODP | membership_ended_before_config_end |
| OGN | membership_ended_before_config_end |
| OI | membership_ended_before_config_end |
| PARA | membership_ended_before_config_end |
| PBCT | membership_ended_before_config_end |
| PBG | membership_ended_before_config_end |
| PBI | membership_ended_before_config_end |
| PCL | membership_ended_before_config_end |
| PCLN | membership_ended_before_config_end |
| PCP | membership_ended_before_config_end |
| PCS | membership_ended_before_config_end |
| PDCO | membership_ended_before_config_end |
| PEAK | membership_ended_before_config_end |
| PENN | membership_ended_before_config_end |
| PETM | membership_ended_before_config_end |
| PGN | membership_ended_before_config_end |
| PHIN | membership_ended_before_config_end, crsp_failed |
| PKI | membership_ended_before_config_end |
| PLL | membership_ended_before_config_end |
| POM | membership_ended_before_config_end |
| PRGO | membership_ended_before_config_end |
| PTV | membership_ended_before_config_end |
| PVH | membership_ended_before_config_end |
| PX | membership_ended_before_config_end |
| PXD | membership_ended_before_config_end |
| QCP | membership_ended_before_config_end, crsp_failed |
| QEP | membership_ended_before_config_end |
| QLGC | membership_ended_before_config_end |
| QRVO | membership_ended_before_config_end |
| R | membership_ended_before_config_end |
| RAI | membership_ended_before_config_end |
| RDC | membership_ended_before_config_end |
| RE | membership_ended_before_config_end |
| REZI | membership_ended_before_config_end, crsp_failed |
| RHI | membership_ended_before_config_end |
| RHT | membership_ended_before_config_end |
| RIG | membership_ended_before_config_end |
| RRC | membership_ended_before_config_end |
| RRD | membership_ended_before_config_end |
| RSH | membership_ended_before_config_end |
| RTN | membership_ended_before_config_end |
| RX | membership_ended_before_config_end |
| S | membership_ended_before_config_end |
| SAI | membership_ended_before_config_end |
| SBNY | membership_ended_before_config_end |
| SCG | membership_ended_before_config_end |
| SE | membership_ended_before_config_end |
| SEDG | membership_ended_before_config_end |
| SEE | membership_ended_before_config_end |
| SHLD | membership_ended_before_config_end |
| SIAL | membership_ended_before_config_end |
| SIG | membership_ended_before_config_end |
| SII | membership_ended_before_config_end |
| SIVB | membership_ended_before_config_end |
| SLE | membership_ended_before_config_end |
| SLG | membership_ended_before_config_end |
| SLM | membership_ended_before_config_end |
| SNI | membership_ended_before_config_end |
| SOLS | membership_ended_before_config_end |
| SPLS | membership_ended_before_config_end |
| SRCL | membership_ended_before_config_end |
| STI | membership_ended_before_config_end |
| STJ | membership_ended_before_config_end |
| STR | membership_ended_before_config_end |
| SUN | membership_ended_before_config_end |
| SVU | membership_ended_before_config_end |
| SWN | membership_ended_before_config_end |
| SWY | membership_ended_before_config_end |
| SYMC | membership_ended_before_config_end |
| TDC | membership_ended_before_config_end |
| TE | membership_ended_before_config_end |
| TEG | membership_ended_before_config_end |
| TFCF | membership_ended_before_config_end |
| TFCFA | membership_ended_before_config_end |
| TFX | membership_ended_before_config_end |
| TGNA | membership_ended_before_config_end |
| THC | membership_ended_before_config_end |
| TIE | membership_ended_before_config_end |
| TIF | membership_ended_before_config_end |
| TLAB | membership_ended_before_config_end |
| TMK | membership_ended_before_config_end |
| TRIP | membership_ended_before_config_end |
| TSO | membership_ended_before_config_end |
| TSS | membership_ended_before_config_end |
| TWC | membership_ended_before_config_end |
| TWTR | membership_ended_before_config_end |
| TWX | membership_ended_before_config_end |
| TYC | membership_ended_before_config_end |
| UA | membership_ended_before_config_end |
| UAA | membership_ended_before_config_end |
| UNM | membership_ended_before_config_end |
| URBN | membership_ended_before_config_end |
| UTX | membership_ended_before_config_end |
| VAR | membership_ended_before_config_end |
| VFC | membership_ended_before_config_end |
| VIA | membership_ended_before_config_end |
| VIAB | membership_ended_before_config_end |
| VIAC | membership_ended_before_config_end |
| VNO | membership_ended_before_config_end |
| VNT | membership_ended_before_config_end |
| VSM | membership_ended_before_config_end, crsp_failed |
| WAG | membership_ended_before_config_end |
| WBA | membership_ended_before_config_end |
| WCG | membership_ended_before_config_end |
| WFM | membership_ended_before_config_end |
| WFMI | membership_ended_before_config_end |
| WFR | membership_ended_before_config_end |
| WHR | membership_ended_before_config_end |
| WIN | membership_ended_before_config_end |
| WLP | membership_ended_before_config_end |
| WLTW | membership_ended_before_config_end |
| WPI | membership_ended_before_config_end |
| WPO | membership_ended_before_config_end |
| WPX | membership_ended_before_config_end |
| WRK | membership_ended_before_config_end |
| WU | membership_ended_before_config_end |
| WYN | membership_ended_before_config_end |
| X | membership_ended_before_config_end |
| XEC | membership_ended_before_config_end |
| XL | membership_ended_before_config_end |
| XLNX | membership_ended_before_config_end |
| XRAY | membership_ended_before_config_end |
| XRX | membership_ended_before_config_end |
| XTO | membership_ended_before_config_end |
| YHOO | membership_ended_before_config_end |
| YUMC | membership_ended_before_config_end, crsp_failed |
| ZION | membership_ended_before_config_end |
| ZMH | membership_ended_before_config_end |

## CRSP Market-Data Availability by Ticker

| ticker | cik | member_from | member_to | interval_count | status | rows | expected_business_days | coverage_ratio | first_date | last_date | issue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0001090872 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AA | 0000004281 | 2010-01-04 | 2016-10-31 | 1 | ok | 1720 | 1781 | 0.966 | 2010-01-04 | 2016-10-31 |  |
| AA | 0001675149 | 2016-11-01 | 2016-11-01 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| AAL | 0000006201 | 2015-03-23 | 2024-09-20 | 1 | ok | 2392 | 2480 | 0.965 | 2015-03-23 | 2024-09-20 |  |
| AAP | 0001158449 | 2015-07-09 | 2023-08-24 | 1 | ok | 2047 | 2121 | 0.965 | 2015-07-09 | 2023-08-24 |  |
| AAPL | 0000320193 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ABBV | 0001551152 | 2013-01-02 | 2025-12-31 | 1 | ok | 3269 | 3391 | 0.964 | 2013-01-03 | 2025-12-31 |  |
| ABC | 0001140859 | 2010-01-04 | 2023-08-29 | 1 | ok | 3437 | 3562 | 0.965 | 2010-01-04 | 2023-08-29 |  |
| ABMD | 0000815094 | 2018-05-31 | 2022-12-21 | 1 | ok | 1150 | 1190 | 0.966 | 2018-05-31 | 2022-12-21 |  |
| ABNB | 0001559720 | 2023-09-18 | 2025-12-31 | 1 | ok | 575 | 598 | 0.962 | 2023-09-18 | 2025-12-31 |  |
| ABT | 0000001800 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ACE | 0000896159 | 2010-07-15 | 2016-01-14 | 1 | ok | 1386 | 1436 | 0.965 | 2010-07-15 | 2016-01-14 |  |
| ACGL | 0000947484 | 2022-11-01 | 2025-12-31 | 1 | ok | 794 | 827 | 0.960 | 2022-11-01 | 2025-12-31 |  |
| ACN | 0001467373 | 2011-07-06 | 2025-12-31 | 1 | ok | 3645 | 3781 | 0.964 | 2011-07-06 | 2025-12-31 |  |
| ACS | 0000002135 | 2010-01-04 | 2010-02-05 | 1 | ok | 24 | 25 | 0.960 | 2010-01-04 | 2010-02-05 |  |
| ACT | 0001578845 | 2013-01-24 | 2015-06-12 | 1 | ok | 601 | 622 | 0.966 | 2013-01-24 | 2015-06-12 |  |
| ADBE | 0000796343 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ADI | 0000006281 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ADM | 0000007084 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ADNT | 0001670541 | 2016-10-31 | 2016-10-31 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| ADP | 0000008670 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ADS | 0001101215 | 2013-12-23 | 2020-06-19 | 1 | ok | 1634 | 1695 | 0.964 | 2013-12-23 | 2020-06-19 |  |
| ADSK | 0000769397 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ADT | 0001546640 | 2012-10-01 | 2016-04-29 | 1 | ok | 899 | 935 | 0.961 | 2012-10-02 | 2016-04-29 |  |
| AEE | 0001002910 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AEP | 0000004904 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AES | 0000874761 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AET | 0001122304 | 2010-01-04 | 2018-11-28 | 1 | ok | 2243 | 2323 | 0.966 | 2010-01-04 | 2018-11-28 |  |
| AFL | 0000004977 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AGL | 0001004155 | 2011-12-13 | 2011-12-15 | 1 | ok | 3 | 3 | 1.000 | 2011-12-13 | 2011-12-15 |  |
| AGN | 0000850693 | 2010-01-04 | 2015-03-16 | 1 | ok | 1308 | 1356 | 0.965 | 2010-01-04 | 2015-03-16 |  |
| AGN | 0001578845 | 2015-06-15 | 2020-05-08 | 1 | ok | 1235 | 1280 | 0.965 | 2015-06-15 | 2020-05-08 |  |
| AIG | 0000005272 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AIRC | 0001820877 | 2020-12-15 | 2020-12-18 | 1 | ok | 3 | 4 | 0.750 | 2020-12-16 | 2020-12-18 |  |
| AIV | 0000922864 | 2010-01-04 | 2020-12-18 | 1 | ok | 2761 | 2860 | 0.965 | 2010-01-04 | 2020-12-18 |  |
| AIZ | 0001267238 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AJG | 0000354190 | 2016-05-31 | 2025-12-31 | 1 | ok | 2412 | 2502 | 0.964 | 2016-05-31 | 2025-12-31 |  |
| AKAM | 0001086222 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AKS | 0000918160 | 2010-01-04 | 2011-12-16 | 1 | ok | 495 | 510 | 0.971 | 2010-01-04 | 2011-12-16 |  |
| ALB | 0000915913 | 2016-07-01 | 2025-12-31 | 1 | ok | 2389 | 2479 | 0.964 | 2016-07-01 | 2025-12-31 |  |
| ALGN | 0001097149 | 2017-06-19 | 2025-12-31 | 1 | ok | 2147 | 2228 | 0.964 | 2017-06-19 | 2025-12-31 |  |
| ALK | 0000766421 | 2016-05-13 | 2023-12-15 | 1 | ok | 1912 | 1981 | 0.965 | 2016-05-13 | 2023-12-15 |  |
| ALL | 0000899051 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ALLE | 0001579241 | 2013-12-02 | 2025-12-31 | 1 | ok | 3038 | 3153 | 0.964 | 2013-12-03 | 2025-12-31 |  |
| ALTR | 0000768251 | 2010-01-04 | 2015-12-24 | 1 | ok | 1506 | 1559 | 0.966 | 2010-01-04 | 2015-12-24 |  |
| ALXN | 0000899866 | 2012-05-25 | 2021-07-20 | 1 | ok | 2302 | 2388 | 0.964 | 2012-05-25 | 2021-07-20 |  |
| AMAT | 0000006951 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AMCR | 0001748790 | 2019-06-11 | 2025-12-31 | 1 | ok | 1649 | 1712 | 0.963 | 2019-06-12 | 2025-12-31 |  |
| AMD | 0000002488 | 2010-01-04 | 2013-09-20 | 1 | ok | 936 | 970 | 0.965 | 2010-01-04 | 2013-09-20 |  |
| AMD | 0000002488 | 2017-03-20 | 2025-12-31 | 1 | ok | 2210 | 2293 | 0.964 | 2017-03-20 | 2025-12-31 |  |
| AME | 0001037868 | 2013-09-23 | 2025-12-31 | 1 | ok | 3088 | 3203 | 0.964 | 2013-09-23 | 2025-12-31 |  |
| AMG | 0001004434 | 2014-07-01 | 2019-12-20 | 1 | ok | 1380 | 1429 | 0.966 | 2014-07-01 | 2019-12-20 |  |
| AMGN | 0000318154 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AMP | 0000820027 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AMT | 0001053507 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AMTM | 0002011286 | 2024-09-30 | 2024-12-20 | 1 | ok | 58 | 60 | 0.967 | 2024-10-01 | 2024-12-20 |  |
| AMZN | 0001018724 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AN | 0000350698 | 2010-01-04 | 2017-08-07 | 1 | ok | 1912 | 1981 | 0.965 | 2010-01-04 | 2017-08-07 |  |
| ANDV | 0000050104 | 2017-08-01 | 2018-09-28 | 1 | ok | 294 | 304 | 0.967 | 2017-08-01 | 2018-09-28 |  |
| ANET | 0001596532 | 2018-08-28 | 2025-12-31 | 1 | ok | 1846 | 1917 | 0.963 | 2018-08-28 | 2025-12-31 |  |
| ANF | 0001018840 | 2010-01-04 | 2013-12-20 | 1 | ok | 1000 | 1035 | 0.966 | 2010-01-04 | 2013-12-20 |  |
| ANR | 0001301063 | 2011-06-02 | 2012-10-01 | 1 | ok | 337 | 348 | 0.968 | 2011-06-02 | 2012-10-01 |  |
| ANSS | 0001013462 | 2017-06-19 | 2025-07-16 | 1 | ok | 2030 | 2108 | 0.963 | 2017-06-19 | 2025-07-16 |  |
| ANTM | 0001156039 | 2014-12-03 | 2022-06-27 | 1 | ok | 1904 | 1974 | 0.965 | 2014-12-03 | 2022-06-27 |  |
| AON | 0000315293 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AOS | 0000091142 | 2017-07-26 | 2025-12-31 | 1 | ok | 2121 | 2201 | 0.964 | 2017-07-26 | 2025-12-31 |  |
| APA | 0001841666 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| APC | 0000773910 | 2010-01-04 | 2019-08-08 | 1 | ok | 2416 | 2504 | 0.965 | 2010-01-04 | 2019-08-08 |  |
| APD | 0000002969 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| APH | 0000820313 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| APO | 0001858681 | 2024-12-23 | 2025-12-31 | 1 | ok | 256 | 268 | 0.955 | 2024-12-23 | 2025-12-31 |  |
| APOL | 0000929887 | 2010-01-04 | 2013-06-28 | 1 | ok | 878 | 910 | 0.965 | 2010-01-04 | 2013-06-28 |  |
| APP | 0001751008 | 2025-09-22 | 2025-12-31 | 1 | ok | 71 | 73 | 0.973 | 2025-09-22 | 2025-12-31 |  |
| APTV | 0001521332 | 2017-12-05 | 2025-12-31 | 1 | ok | 2029 | 2107 | 0.963 | 2017-12-05 | 2025-12-31 |  |
| APY | 0001723089 | 2018-05-09 | 2018-05-09 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| ARE | 0001035443 | 2017-03-20 | 2025-12-31 | 1 | ok | 2210 | 2293 | 0.964 | 2017-03-20 | 2025-12-31 |  |
| ARES | 0001176948 | 2025-12-11 | 2025-12-31 | 1 | ok | 14 | 15 | 0.933 | 2025-12-11 | 2025-12-31 |  |
| ARG | 0000804212 | 2010-01-04 | 2016-05-20 | 1 | ok | 1607 | 1665 | 0.965 | 2010-01-04 | 2016-05-20 |  |
| ARNC | 0000004281 | 2016-11-01 | 2020-03-31 | 1 | ok | 858 | 891 | 0.963 | 2016-11-01 | 2020-03-31 |  |
| ASIX | 0001673985 | 2016-10-03 | 2016-10-03 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| ATI | 0001018963 | 2010-01-04 | 2015-07-01 | 1 | ok | 1383 | 1433 | 0.965 | 2010-01-04 | 2015-07-01 |  |
| ATO | 0000731802 | 2019-02-15 | 2025-12-31 | 1 | ok | 1729 | 1794 | 0.964 | 2019-02-15 | 2025-12-31 |  |
| ATVI | 0000718877 | 2015-08-31 | 2023-10-12 | 1 | ok | 2044 | 2119 | 0.965 | 2015-08-31 | 2023-10-12 |  |
| AVB | 0000915912 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AVGO | 0001730168 | 2014-05-08 | 2025-12-31 | 1 | ok | 2931 | 3040 | 0.964 | 2014-05-08 | 2025-12-31 |  |
| AVP | 0000008868 | 2010-01-04 | 2015-03-20 | 1 | ok | 1312 | 1360 | 0.965 | 2010-01-04 | 2015-03-20 |  |
| AVY | 0000008818 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AWK | 0001410636 | 2016-03-04 | 2025-12-31 | 1 | ok | 2472 | 2564 | 0.964 | 2016-03-04 | 2025-12-31 |  |
| AXON | 0001069183 | 2023-05-04 | 2025-12-31 | 1 | ok | 668 | 695 | 0.961 | 2023-05-04 | 2025-12-31 |  |
| AXP | 0000004962 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| AYE | 0000003673 | 2010-01-04 | 2011-02-25 | 1 | ok | 290 | 300 | 0.967 | 2010-01-04 | 2011-02-25 |  |
| AYI | 0001144215 | 2016-05-03 | 2018-06-15 | 1 | ok | 535 | 554 | 0.966 | 2016-05-03 | 2018-06-15 |  |
| AZO | 0000866787 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| BA | 0000012927 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| BAC | 0000070858 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| BALL | 0000009389 | 2022-05-10 | 2025-12-31 | 1 | ok | 915 | 952 | 0.961 | 2022-05-10 | 2025-12-31 |  |
| BAX | 0000010456 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| BBBY | 0000886158 | 2010-01-04 | 2017-07-25 | 1 | ok | 1903 | 1972 | 0.965 | 2010-01-04 | 2017-07-25 |  |
| BBT | 0000092230 | 2010-01-04 | 2019-12-06 | 1 | ok | 2500 | 2590 | 0.965 | 2010-01-04 | 2019-12-06 |  |
| BBWI | 0000701985 | 2021-08-03 | 2024-09-30 | 1 | ok | 795 | 825 | 0.964 | 2021-08-03 | 2024-09-30 |  |
| BBY | 0000764478 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| BCR | 0000009892 | 2010-01-04 | 2017-12-28 | 1 | ok | 2012 | 2084 | 0.965 | 2010-01-04 | 2017-12-28 |  |
| BDK | 0000012355 | 2010-01-04 | 2010-03-12 | 1 | ok | 48 | 50 | 0.960 | 2010-01-04 | 2010-03-12 |  |
| BDX | 0000010795 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| BEAM | 0000789073 | 2011-10-04 | 2014-04-30 | 1 | ok | 646 | 672 | 0.961 | 2011-10-04 | 2014-04-30 |  |
| BEN | 0000038777 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| BF | 0000014693 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| BG | 0001996862 | 2023-03-15 | 2025-12-31 | 1 | ok | 703 | 731 | 0.962 | 2023-03-15 | 2025-12-31 |  |
| BHF | 0001685040 | 2017-08-07 | 2019-04-02 | 1 | ok | 415 | 432 | 0.961 | 2017-08-08 | 2019-04-02 |  |
| BHGE | 0001701605 | 2017-07-05 | 2019-10-17 | 1 | ok | 577 | 597 | 0.966 | 2017-07-05 | 2019-10-17 |  |
| BHI | 0000808362 | 2010-01-04 | 2017-07-03 | 1 | ok | 1888 | 1956 | 0.965 | 2010-01-04 | 2017-07-03 |  |
| BIG | 0000768835 | 2010-01-04 | 2013-02-13 | 1 | ok | 784 | 813 | 0.964 | 2010-01-04 | 2013-02-13 |  |
| BIIB | 0000875045 | 2010-01-04 | 2025-12-31 | 1 | ok | 4023 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| BIO | 0000012208 | 2020-06-22 | 2024-09-20 | 1 | ok | 1070 | 1110 | 0.964 | 2020-06-22 | 2024-09-20 |  |
| BIVV | 0001681689 | 2017-02-02 | 2017-02-02 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| BJS | 0000864328 | 2010-01-04 | 2010-04-28 | 1 | ok | 80 | 83 | 0.964 | 2010-01-04 | 2010-04-28 |  |
| BK | 0001390777 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| BKNG | 0001075531 | 2018-02-27 | 2025-12-31 | 1 | ok | 1973 | 2047 | 0.964 | 2018-02-27 | 2025-12-31 |  |
| BKR | 0001701605 | 2019-10-18 | 2025-12-31 | 1 | ok | 1559 | 1619 | 0.963 | 2019-10-18 | 2025-12-31 |  |
| BLDR | 0001316835 | 2023-12-18 | 2025-12-31 | 1 | ok | 511 | 533 | 0.959 | 2023-12-18 | 2025-12-31 |  |
| BLK | 0002012383 | 2011-04-04 | 2025-12-31 | 1 | ok | 3709 | 3848 | 0.964 | 2011-04-04 | 2025-12-31 |  |
| BLL | 0000009389 | 2010-01-04 | 2022-05-09 | 1 | ok | 3109 | 3221 | 0.965 | 2010-01-04 | 2022-05-09 |  |
| BMC | 0000835729 | 2010-01-04 | 2013-09-10 | 1 | ok | 928 | 962 | 0.965 | 2010-01-04 | 2013-09-10 |  |
| BMS | 0000011199 | 2010-01-04 | 2014-12-04 | 1 | ok | 1240 | 1284 | 0.966 | 2010-01-04 | 2014-12-04 |  |
| BMS | 0000011199 | 2019-06-07 | 2019-06-10 | 1 | ok | 2 | 2 | 1.000 | 2019-06-07 | 2019-06-10 |  |
| BMY | 0000014272 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| BNI | 0000934612 | 2010-01-04 | 2010-02-12 | 1 | ok | 29 | 30 | 0.967 | 2010-01-04 | 2010-02-12 |  |
| BR | 0001383312 | 2018-06-18 | 2025-12-31 | 1 | ok | 1896 | 1968 | 0.963 | 2018-06-18 | 2025-12-31 |  |
| BRCM | 0001054374 | 2010-01-04 | 2016-01-29 | 1 | ok | 1529 | 1585 | 0.965 | 2010-01-04 | 2016-01-29 |  |
| BRK | 0001067983 | 2010-02-16 | 2025-12-31 | 1 | ok | 3995 | 4142 | 0.965 | 2010-02-16 | 2025-12-31 |  |
| BRO | 0000079282 | 2021-09-20 | 2025-12-31 | 1 | ok | 1076 | 1118 | 0.962 | 2021-09-20 | 2025-12-31 |  |
| BSX | 0000885725 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| BTU | 0001064728 | 2010-01-04 | 2014-09-19 | 1 | ok | 1187 | 1230 | 0.965 | 2010-01-04 | 2014-09-19 |  |
| BWA | 0000908255 | 2011-12-19 | 2025-03-21 | 1 | ok | 3333 | 3460 | 0.963 | 2011-12-19 | 2025-03-21 |  |
| BX | 0001393818 | 2023-09-18 | 2025-12-31 | 1 | ok | 575 | 598 | 0.962 | 2023-09-18 | 2025-12-31 |  |
| BXLT | 0001620546 | 2015-07-01 | 2016-06-02 | 1 | ok | 232 | 242 | 0.959 | 2015-07-02 | 2016-06-02 |  |
| BXP | 0001037540 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| C | 0000831001 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CA | 0000356028 | 2010-01-04 | 2018-11-02 | 1 | ok | 2226 | 2305 | 0.966 | 2010-01-04 | 2018-11-02 |  |
| CAG | 0000023217 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CAH | 0000721371 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CAM | 0000941548 | 2010-01-04 | 2016-04-01 | 1 | ok | 1572 | 1630 | 0.964 | 2010-01-04 | 2016-04-01 |  |
| CARR | 0001783180 | 2020-04-03 | 2025-12-31 | 1 | ok | 1443 | 1499 | 0.963 | 2020-04-06 | 2025-12-31 |  |
| CAT | 0000018230 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CB | 0000020171 | 2010-01-04 | 2016-01-14 | 1 | ok | 1519 | 1574 | 0.965 | 2010-01-04 | 2016-01-14 |  |
| CB | 0000896159 | 2016-01-15 | 2025-12-31 | 1 | ok | 2505 | 2599 | 0.964 | 2016-01-15 | 2025-12-31 |  |
| CBE | 0001141982 | 2011-11-23 | 2012-11-30 | 1 | ok | 256 | 268 | 0.955 | 2011-11-23 | 2012-11-30 |  |
| CBG | 0001138118 | 2010-01-04 | 2018-03-19 | 1 | ok | 2066 | 2141 | 0.965 | 2010-01-04 | 2018-03-19 |  |
| CBOE | 0001374310 | 2017-03-01 | 2025-12-31 | 1 | ok | 2223 | 2306 | 0.964 | 2017-03-01 | 2025-12-31 |  |
| CBRE | 0001138118 | 2018-03-20 | 2025-12-31 | 1 | ok | 1958 | 2032 | 0.964 | 2018-03-20 | 2025-12-31 |  |
| CBS | 0002041610 | 2010-01-04 | 2019-12-04 | 1 | ok | 2498 | 2588 | 0.965 | 2010-01-04 | 2019-12-04 |  |
| CCE | 0001650107 | 2010-01-04 | 2016-05-27 | 1 | ok | 1612 | 1670 | 0.965 | 2010-01-04 | 2016-05-27 |  |
| CCI | 0001051470 | 2012-03-14 | 2025-12-31 | 1 | ok | 3471 | 3601 | 0.964 | 2012-03-14 | 2025-12-31 |  |
| CCL | 0000815097 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CDAY | 0001725057 | 2021-09-20 | 2024-01-31 | 1 | ok | 595 | 618 | 0.963 | 2021-09-20 | 2024-01-31 |  |
| CDNS | 0000813672 | 2017-09-18 | 2025-12-31 | 1 | ok | 2084 | 2163 | 0.963 | 2017-09-18 | 2025-12-31 |  |
| CDW | 0001402057 | 2019-09-23 | 2025-12-31 | 1 | ok | 1578 | 1638 | 0.963 | 2019-09-23 | 2025-12-31 |  |
| CE | 0001306830 | 2018-12-24 | 2025-03-21 | 1 | ok | 1569 | 1630 | 0.963 | 2018-12-24 | 2025-03-21 |  |
| CEG | 0001004440 | 2010-01-04 | 2012-03-12 | 1 | ok | 552 | 571 | 0.967 | 2010-01-04 | 2012-03-12 |  |
| CEG | 0001868275 | 2022-02-02 | 2025-12-31 | 1 | ok | 981 | 1021 | 0.961 | 2022-02-03 | 2025-12-31 |  |
| CELG | 0000816284 | 2010-01-04 | 2019-11-20 | 1 | ok | 2489 | 2578 | 0.965 | 2010-01-04 | 2019-11-20 |  |
| CEPH | 0000873364 | 2010-01-04 | 2011-10-13 | 1 | ok | 450 | 464 | 0.970 | 2010-01-04 | 2011-10-13 |  |
| CERN | 0000804753 | 2010-04-30 | 2022-06-07 | 1 | ok | 3048 | 3158 | 0.965 | 2010-04-30 | 2022-06-07 |  |
| CF | 0001324404 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CFG | 0000759944 | 2016-02-01 | 2025-12-31 | 1 | ok | 2495 | 2588 | 0.964 | 2016-02-01 | 2025-12-31 |  |
| CFN | 0001457543 | 2010-01-04 | 2015-03-16 | 1 | ok | 1308 | 1356 | 0.965 | 2010-01-04 | 2015-03-16 |  |
| CHD | 0000313927 | 2015-12-29 | 2025-12-31 | 1 | ok | 2517 | 2612 | 0.964 | 2015-12-29 | 2025-12-31 |  |
| CHK | 0000895126 | 2010-01-04 | 2018-03-16 | 1 | ok | 2065 | 2140 | 0.965 | 2010-01-04 | 2018-03-16 |  |
| CHRW | 0001043277 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CHTR | 0001091667 | 2016-09-08 | 2025-12-31 | 1 | ok | 2342 | 2430 | 0.964 | 2016-09-08 | 2025-12-31 |  |
| CI | 0001739940 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CINF | 0000020286 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CL | 0000021665 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CLF | 0000764065 | 2010-01-04 | 2014-04-01 | 1 | ok | 1068 | 1107 | 0.965 | 2010-01-04 | 2014-04-01 |  |
| CLX | 0000021076 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CMA | 0000028412 | 2010-01-04 | 2024-06-21 | 1 | ok | 3641 | 3775 | 0.965 | 2010-01-04 | 2024-06-21 |  |
| CMCSA | 0001166691 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CMCSK | 0001166691 | 2015-09-21 | 2015-12-11 | 1 | ok | 59 | 60 | 0.983 | 2015-09-21 | 2015-12-11 |  |
| CME | 0001156375 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CMG | 0001058090 | 2011-04-28 | 2025-12-31 | 1 | ok | 3692 | 3830 | 0.964 | 2011-04-28 | 2025-12-31 |  |
| CMI | 0000026172 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CMS | 0000811156 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CNC | 0001071739 | 2016-03-30 | 2025-12-31 | 1 | ok | 2455 | 2546 | 0.964 | 2016-03-30 | 2025-12-31 |  |
| CNP | 0001130310 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CNX | 0001070412 | 2010-01-04 | 2016-03-03 | 1 | ok | 1552 | 1609 | 0.965 | 2010-01-04 | 2016-03-03 |  |
| COF | 0000927628 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| COG | 0000858470 | 2010-01-04 | 2021-10-01 | 1 | ok | 2958 | 3065 | 0.965 | 2010-01-04 | 2021-10-01 |  |
| COH | 0001116132 | 2010-01-04 | 2017-10-30 | 1 | ok | 1971 | 2041 | 0.966 | 2010-01-04 | 2017-10-30 |  |
| COIN | 0001679788 | 2025-05-19 | 2025-12-31 | 1 | ok | 157 | 163 | 0.963 | 2025-05-19 | 2025-12-31 |  |
| COL | 0001137411 | 2010-01-04 | 2018-11-26 | 1 | ok | 2241 | 2321 | 0.966 | 2010-01-04 | 2018-11-26 |  |
| COO | 0000711404 | 2016-09-23 | 2025-12-31 | 1 | ok | 2331 | 2419 | 0.964 | 2016-09-23 | 2025-12-31 |  |
| COP | 0001163165 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| COR | 0001140859 | 2023-08-30 | 2025-12-31 | 1 | ok | 587 | 611 | 0.961 | 2023-08-30 | 2025-12-31 |  |
| COST | 0000909832 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| COTY | 0001024305 | 2016-10-03 | 2020-09-18 | 1 | ok | 998 | 1035 | 0.964 | 2016-10-03 | 2020-09-18 |  |
| COV | 0001385187 | 2011-03-01 | 2015-01-26 | 1 | ok | 983 | 1020 | 0.964 | 2011-03-01 | 2015-01-26 |  |
| CPAY | 0001175454 | 2024-03-25 | 2025-12-31 | 1 | ok | 445 | 463 | 0.961 | 2024-03-25 | 2025-12-31 |  |
| CPB | 0000016732 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CPGX | 0001629995 | 2015-07-02 | 2016-06-30 | 1 | ok | 251 | 261 | 0.962 | 2015-07-06 | 2016-06-30 |  |
| CPRI | 0001530721 | 2019-01-02 | 2020-05-11 | 1 | ok | 342 | 354 | 0.966 | 2019-01-02 | 2020-05-11 |  |
| CPRT | 0000900075 | 2018-07-02 | 2025-12-31 | 1 | ok | 1886 | 1958 | 0.963 | 2018-07-02 | 2025-12-31 |  |
| CPT | 0000906345 | 2022-04-04 | 2025-12-31 | 1 | ok | 940 | 978 | 0.961 | 2022-04-04 | 2025-12-31 |  |
| CPWR | 0000859014 | 2010-01-04 | 2011-12-30 | 1 | ok | 504 | 520 | 0.969 | 2010-01-04 | 2011-12-30 |  |
| CRH | 0000849395 | 2025-12-22 | 2025-12-31 | 1 | ok | 7 | 8 | 0.875 | 2025-12-22 | 2025-12-31 |  |
| CRL | 0001100682 | 2021-05-14 | 2025-12-31 | 1 | ok | 1164 | 1209 | 0.963 | 2021-05-14 | 2025-12-31 |  |
| CRM | 0001108524 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CRWD | 0001535527 | 2024-06-24 | 2025-12-31 | 1 | ok | 383 | 398 | 0.962 | 2024-06-24 | 2025-12-31 |  |
| CSC | 0001688568 | 2010-01-04 | 2015-11-30 | 1 | ok | 1488 | 1541 | 0.966 | 2010-01-04 | 2015-11-30 |  |
| CSCO | 0000858877 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CSGP | 0001057352 | 2022-09-19 | 2025-12-31 | 1 | ok | 825 | 858 | 0.962 | 2022-09-19 | 2025-12-31 |  |
| CSRA | 0001646383 | 2015-11-30 | 2018-04-03 | 1 | ok | 588 | 612 | 0.961 | 2015-12-01 | 2018-04-03 |  |
| CSX | 0000277948 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CTAS | 0000723254 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CTL | 0000018926 | 2010-01-04 | 2020-09-17 | 1 | ok | 2696 | 2794 | 0.965 | 2010-01-04 | 2020-09-17 |  |
| CTLT | 0001596783 | 2020-09-21 | 2024-12-17 | 1 | ok | 1068 | 1107 | 0.965 | 2020-09-21 | 2024-12-17 |  |
| CTRA | 0000858470 | 2021-10-04 | 2025-12-31 | 1 | ok | 1066 | 1108 | 0.962 | 2021-10-04 | 2025-12-31 |  |
| CTSH | 0001058290 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CTVA | 0001755672 | 2019-06-03 | 2025-12-31 | 1 | ok | 1655 | 1718 | 0.963 | 2019-06-04 | 2025-12-31 |  |
| CTXS | 0000877890 | 2010-01-04 | 2022-09-29 | 1 | ok | 3208 | 3324 | 0.965 | 2010-01-04 | 2022-09-29 |  |
| CVC | 0001702780 | 2010-12-20 | 2016-06-21 | 1 | ok | 1385 | 1437 | 0.964 | 2010-12-20 | 2016-06-21 |  |
| CVET | 0001752836 | 2019-02-08 | 2019-02-08 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| CVH | 0001054833 | 2010-01-04 | 2013-05-06 | 1 | ok | 840 | 871 | 0.964 | 2010-01-04 | 2013-05-06 |  |
| CVNA | 0001690820 | 2025-12-22 | 2025-12-31 | 1 | ok | 7 | 8 | 0.875 | 2025-12-22 | 2025-12-31 |  |
| CVS | 0000064803 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CVX | 0000093410 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| CXO | 0001358071 | 2016-02-22 | 2021-01-15 | 1 | ok | 1236 | 1280 | 0.966 | 2016-02-22 | 2021-01-15 |  |
| CZR | 0001590895 | 2021-03-22 | 2025-09-19 | 1 | ok | 1131 | 1175 | 0.963 | 2021-03-22 | 2025-09-19 |  |
| D | 0000715957 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| DAL | 0000027904 | 2013-09-11 | 2025-12-31 | 1 | ok | 3096 | 3211 | 0.964 | 2013-09-11 | 2025-12-31 |  |
| DASH | 0001792789 | 2025-03-24 | 2025-12-31 | 1 | ok | 196 | 203 | 0.966 | 2025-03-24 | 2025-12-31 |  |
| DAY | 0001725057 | 2024-02-01 | 2025-12-31 | 1 | ok | 481 | 500 | 0.962 | 2024-02-01 | 2025-12-31 |  |
| DD | 0000030554 | 2010-01-04 | 2017-08-31 | 1 | ok | 1930 | 1999 | 0.965 | 2010-01-04 | 2017-08-31 |  |
| DD | 0001666700 | 2019-06-03 | 2025-12-31 | 1 | ok | 1656 | 1718 | 0.964 | 2019-06-03 | 2025-12-31 |  |
| DDOG | 0001561550 | 2025-07-09 | 2025-12-31 | 1 | ok | 123 | 126 | 0.976 | 2025-07-09 | 2025-12-31 |  |
| DE | 0000315189 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| DECK | 0000910521 | 2024-03-18 | 2025-12-31 | 1 | ok | 450 | 468 | 0.962 | 2024-03-18 | 2025-12-31 |  |
| DELL | 0001571996 | 2010-01-04 | 2013-10-28 | 1 | ok | 962 | 996 | 0.966 | 2010-01-04 | 2013-10-28 |  |
| DELL | 0001571996 | 2024-09-23 | 2025-12-31 | 1 | ok | 320 | 333 | 0.961 | 2024-09-23 | 2025-12-31 |  |
| DF | 0000931336 | 2010-01-04 | 2013-05-23 | 1 | ok | 853 | 884 | 0.965 | 2010-01-04 | 2013-05-23 |  |
| DFS | 0001393612 | 2010-01-04 | 2025-05-16 | 1 | ok | 3867 | 4010 | 0.964 | 2010-01-04 | 2025-05-16 |  |
| DG | 0000029534 | 2012-12-03 | 2025-12-31 | 1 | ok | 3290 | 3413 | 0.964 | 2012-12-03 | 2025-12-31 |  |
| DGX | 0001022079 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| DHI | 0000882184 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| DHR | 0000313616 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| DIS | 0001744489 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| DISCA | 0001437107 | 2010-03-01 | 2022-04-08 | 1 | ok | 3051 | 3160 | 0.966 | 2010-03-01 | 2022-04-08 |  |
| DISCK | 0001437107 | 2014-08-07 | 2022-04-08 | 1 | ok | 1933 | 2002 | 0.966 | 2014-08-07 | 2022-04-08 |  |
| DISH | 0001001082 | 2017-03-13 | 2023-06-16 | 1 | ok | 1578 | 1635 | 0.965 | 2017-03-13 | 2023-06-16 |  |
| DLPH | 0001521332 | 2012-12-24 | 2017-12-04 | 1 | ok | 1246 | 1291 | 0.965 | 2012-12-24 | 2017-12-04 |  |
| DLR | 0001297996 | 2016-05-18 | 2025-12-31 | 1 | ok | 2420 | 2511 | 0.964 | 2016-05-18 | 2025-12-31 |  |
| DLTR | 0000935703 | 2011-12-19 | 2025-12-31 | 1 | ok | 3529 | 3663 | 0.963 | 2011-12-19 | 2025-12-31 |  |
| DNB | 0001799208 | 2010-01-04 | 2017-04-04 | 1 | ok | 1826 | 1892 | 0.965 | 2010-01-04 | 2017-04-04 |  |
| DNR | 0000945764 | 2010-01-04 | 2015-03-20 | 1 | ok | 1312 | 1360 | 0.965 | 2010-01-04 | 2015-03-20 |  |
| DO | 0000949039 | 2010-01-04 | 2016-09-30 | 1 | ok | 1699 | 1760 | 0.965 | 2010-01-04 | 2016-09-30 |  |
| DOC | 0000765880 | 2024-03-04 | 2025-12-31 | 1 | ok | 460 | 478 | 0.962 | 2024-03-04 | 2025-12-31 |  |
| DOV | 0000029905 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| DOW | 0001666700 | 2010-01-04 | 2017-08-31 | 1 | ok | 1930 | 1999 | 0.965 | 2010-01-04 | 2017-08-31 |  |
| DOW | 0001751788 | 2019-04-02 | 2025-12-31 | 1 | ok | 1697 | 1762 | 0.963 | 2019-04-03 | 2025-12-31 |  |
| DPS | 0001418135 | 2010-01-04 | 2018-06-29 | 1 | ok | 2138 | 2215 | 0.965 | 2010-01-04 | 2018-06-29 |  |
| DPZ | 0001286681 | 2020-05-12 | 2025-12-31 | 1 | ok | 1418 | 1472 | 0.963 | 2020-05-12 | 2025-12-31 |  |
| DRE | 0000783280 | 2017-07-26 | 2022-09-30 | 1 | ok | 1306 | 1353 | 0.965 | 2017-07-26 | 2022-09-30 |  |
| DRI | 0000940944 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| DTE | 0000936340 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| DTV | 0001465112 | 2010-01-04 | 2015-07-24 | 1 | ok | 1399 | 1450 | 0.965 | 2010-01-04 | 2015-07-24 |  |
| DUK | 0001326160 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| DV | 0000730464 | 2010-01-04 | 2012-09-28 | 1 | ok | 692 | 715 | 0.968 | 2010-01-04 | 2012-09-28 |  |
| DVA | 0000927066 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| DVN | 0001090012 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| DWDP | 0001666700 | 2017-09-01 | 2019-05-31 | 1 | ok | 437 | 456 | 0.958 | 2017-09-05 | 2019-05-31 |  |
| DXC | 0001688568 | 2017-04-04 | 2023-10-02 | 1 | ok | 1635 | 1695 | 0.965 | 2017-04-04 | 2023-10-02 |  |
| DXCM | 0001093557 | 2020-05-12 | 2025-12-31 | 1 | ok | 1418 | 1472 | 0.963 | 2020-05-12 | 2025-12-31 |  |
| EA | 0000712515 | 2011-12-20 | 2025-12-31 | 1 | ok | 3528 | 3662 | 0.963 | 2011-12-20 | 2025-12-31 |  |
| EBAY | 0001065088 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ECL | 0000031462 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ED | 0001047862 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| EFX | 0000033185 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| EG | 0001095073 | 2023-07-10 | 2025-12-31 | 1 | ok | 624 | 648 | 0.963 | 2023-07-10 | 2025-12-31 |  |
| EIX | 0000827052 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| EK | 0000031235 | 2010-01-04 | 2010-12-17 | 1 | ok | 243 | 250 | 0.972 | 2010-01-04 | 2010-12-17 |  |
| EL | 0001001250 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ELV | 0001156039 | 2022-06-28 | 2025-12-31 | 1 | ok | 882 | 917 | 0.962 | 2022-06-28 | 2025-12-31 |  |
| EMC | 0000790070 | 2010-01-04 | 2016-09-06 | 1 | ok | 1681 | 1742 | 0.965 | 2010-01-04 | 2016-09-06 |  |
| EME | 0000105634 | 2025-09-22 | 2025-12-31 | 1 | ok | 71 | 73 | 0.973 | 2025-09-22 | 2025-12-31 |  |
| EMN | 0000915389 | 2010-01-04 | 2025-11-03 | 1 | ok | 3984 | 4131 | 0.964 | 2010-01-04 | 2025-11-03 |  |
| EMR | 0000032604 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ENDP | 0001593034 | 2015-01-27 | 2017-03-01 | 1 | ok | 528 | 547 | 0.965 | 2015-01-27 | 2017-03-01 |  |
| ENPH | 0001463101 | 2021-01-07 | 2025-09-19 | 1 | ok | 1181 | 1227 | 0.963 | 2021-01-07 | 2025-09-19 |  |
| EOG | 0000821189 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| EP | 0001066107 | 2010-01-04 | 2012-05-24 | 1 | ok | 604 | 624 | 0.968 | 2010-01-04 | 2012-05-24 |  |
| EPAM | 0001352010 | 2021-12-14 | 2025-12-31 | 1 | ok | 1016 | 1057 | 0.961 | 2021-12-14 | 2025-12-31 |  |
| EQIX | 0001101239 | 2015-03-23 | 2025-12-31 | 1 | ok | 2712 | 2813 | 0.964 | 2015-03-23 | 2025-12-31 |  |
| EQR | 0000906107 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| EQT | 0000033213 | 2010-01-04 | 2018-11-12 | 1 | ok | 2232 | 2311 | 0.966 | 2010-01-04 | 2018-11-12 |  |
| EQT | 0000033213 | 2022-10-03 | 2025-12-31 | 1 | ok | 815 | 848 | 0.961 | 2022-10-03 | 2025-12-31 |  |
| ERIE | 0000922621 | 2024-09-23 | 2025-12-31 | 1 | ok | 320 | 333 | 0.961 | 2024-09-23 | 2025-12-31 |  |
| ERTS | 0000712515 | 2010-01-04 | 2011-12-19 | 1 | ok | 496 | 511 | 0.971 | 2010-01-04 | 2011-12-19 |  |
| ES | 0000072741 | 2015-02-19 | 2025-12-31 | 1 | ok | 2734 | 2835 | 0.964 | 2015-02-19 | 2025-12-31 |  |
| ESRX | 0001532063 | 2010-01-04 | 2018-12-20 | 1 | ok | 2258 | 2339 | 0.965 | 2010-01-04 | 2018-12-20 |  |
| ESS | 0000920522 | 2014-04-02 | 2025-12-31 | 1 | ok | 2956 | 3066 | 0.964 | 2014-04-02 | 2025-12-31 |  |
| ESV | 0000314808 | 2012-07-31 | 2016-03-29 | 1 | ok | 920 | 956 | 0.962 | 2012-07-31 | 2016-03-29 |  |
| ETFC | 0001015780 | 2010-01-04 | 2020-10-01 | 1 | ok | 2706 | 2804 | 0.965 | 2010-01-04 | 2020-10-01 |  |
| ETN | 0001551182 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ETR | 0000065984 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ETSY | 0001370637 | 2020-09-21 | 2024-09-20 | 1 | ok | 1007 | 1045 | 0.964 | 2020-09-21 | 2024-09-20 |  |
| EVHC | 0001678531 | 2016-12-02 | 2018-10-10 | 1 | ok | 467 | 484 | 0.965 | 2016-12-02 | 2018-10-10 |  |
| EVRG | 0001711269 | 2018-06-05 | 2025-12-31 | 1 | ok | 1905 | 1977 | 0.964 | 2018-06-05 | 2025-12-31 |  |
| EW | 0001099800 | 2011-04-01 | 2025-12-31 | 1 | ok | 3710 | 3849 | 0.964 | 2011-04-01 | 2025-12-31 |  |
| EXC | 0001109357 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| EXE | 0000895126 | 2025-03-24 | 2025-12-31 | 1 | ok | 196 | 203 | 0.966 | 2025-03-24 | 2025-12-31 |  |
| EXPD | 0000746515 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| EXPE | 0001324424 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| EXR | 0001289490 | 2016-01-19 | 2025-12-31 | 1 | ok | 2504 | 2597 | 0.964 | 2016-01-19 | 2025-12-31 |  |
| F | 0000037996 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| FANG | 0001539838 | 2018-12-03 | 2025-12-31 | 1 | ok | 1779 | 1848 | 0.963 | 2018-12-03 | 2025-12-31 |  |
| FAST | 0000815556 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| FB | 0001326801 | 2013-12-23 | 2022-06-08 | 1 | ok | 2130 | 2208 | 0.965 | 2013-12-23 | 2022-06-08 |  |
| FBHS | 0001519751 | 2016-06-24 | 2022-12-14 | 1 | ok | 1631 | 1689 | 0.966 | 2016-06-24 | 2022-12-14 |  |
| FBIN | 0001519751 | 2022-12-15 | 2022-12-16 | 1 | ok | 2 | 2 | 1.000 | 2022-12-15 | 2022-12-16 |  |
| FCPT | 0001650132 | 2015-11-10 | 2015-11-10 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| FCX | 0000831259 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| FDO | 0000034408 | 2010-01-04 | 2015-07-06 | 1 | ok | 1385 | 1436 | 0.964 | 2010-01-04 | 2015-07-06 |  |
| FDS | 0001013237 | 2021-12-20 | 2025-12-31 | 1 | ok | 1012 | 1053 | 0.961 | 2021-12-20 | 2025-12-31 |  |
| FDX | 0001048911 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| FE | 0001031296 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| FFIV | 0001048695 | 2010-12-20 | 2025-12-31 | 1 | ok | 3781 | 3923 | 0.964 | 2010-12-20 | 2025-12-31 |  |
| FHN | 0000036966 | 2010-01-04 | 2013-06-21 | 1 | ok | 873 | 905 | 0.965 | 2010-01-04 | 2013-06-21 |  |
| FI | 0000798354 | 2023-06-07 | 2025-11-10 | 1 | ok | 610 | 634 | 0.962 | 2023-06-07 | 2025-11-10 |  |
| FICO | 0000814547 | 2023-03-20 | 2025-12-31 | 1 | ok | 700 | 728 | 0.962 | 2023-03-20 | 2025-12-31 |  |
| FII | 0001056288 | 2010-01-04 | 2012-12-31 | 1 | ok | 754 | 781 | 0.965 | 2010-01-04 | 2012-12-31 |  |
| FIS | 0001136893 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| FISV | 0000798354 | 2010-01-04 | 2025-12-31 | 1 | ok | 3414 | 4173 | 0.818 | 2010-01-04 | 2025-12-31 |  |
| FITB | 0000035527 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| FIX | 0001035983 | 2025-12-22 | 2025-12-31 | 1 | ok | 7 | 8 | 0.875 | 2025-12-22 | 2025-12-31 |  |
| FL | 0000850209 | 2016-04-04 | 2019-08-08 | 1 | ok | 844 | 874 | 0.966 | 2016-04-04 | 2019-08-08 |  |
| FLIR | 0000354908 | 2010-01-04 | 2021-05-13 | 1 | ok | 2860 | 2964 | 0.965 | 2010-01-04 | 2021-05-13 |  |
| FLR | 0001124198 | 2010-01-04 | 2019-06-03 | 1 | ok | 2369 | 2456 | 0.965 | 2010-01-04 | 2019-06-03 |  |
| FLS | 0000030625 | 2010-01-04 | 2021-03-19 | 1 | ok | 2822 | 2925 | 0.965 | 2010-01-04 | 2021-03-19 |  |
| FLT | 0001175454 | 2018-06-20 | 2024-03-22 | 1 | ok | 1449 | 1503 | 0.964 | 2018-06-20 | 2024-03-22 |  |
| FMC | 0000037785 | 2010-01-04 | 2025-03-21 | 1 | ok | 3828 | 3970 | 0.964 | 2010-01-04 | 2025-03-21 |  |
| FO | 0000789073 | 2010-01-04 | 2011-10-03 | 1 | ok | 442 | 456 | 0.969 | 2010-01-04 | 2011-10-03 |  |
| FOSL | 0000883569 | 2012-04-04 | 2016-01-04 | 1 | ok | 943 | 979 | 0.963 | 2012-04-04 | 2016-01-04 |  |
| FOX | 0001308161 | 2015-09-21 | 2019-03-18 | 1 | ok | 878 | 911 | 0.964 | 2015-09-21 | 2019-03-18 |  |
| FOX | 0001754301 | 2019-03-19 | 2025-12-31 | 1 | ok | 1707 | 1772 | 0.963 | 2019-03-20 | 2025-12-31 |  |
| FOXA | 0001308161 | 2013-07-01 | 2019-03-18 | 1 | ok | 1438 | 1491 | 0.964 | 2013-07-01 | 2019-03-18 |  |
| FOXA | 0001754301 | 2019-03-19 | 2025-12-31 | 1 | ok | 1707 | 1772 | 0.963 | 2019-03-20 | 2025-12-31 |  |
| FPL | 0000753308 | 2010-01-04 | 2010-06-22 | 1 | ok | 118 | 122 | 0.967 | 2010-01-04 | 2010-06-22 |  |
| FRC | 0001132979 | 2019-01-02 | 2023-04-28 | 1 | ok | 1089 | 1128 | 0.965 | 2019-01-02 | 2023-04-28 |  |
| FRT | 0000034903 | 2016-02-01 | 2025-12-31 | 1 | ok | 2495 | 2588 | 0.964 | 2016-02-01 | 2025-12-31 |  |
| FRX | 0000038074 | 2010-01-04 | 2014-06-30 | 1 | ok | 1130 | 1171 | 0.965 | 2010-01-04 | 2014-06-30 |  |
| FSLR | 0001274494 | 2010-01-04 | 2017-03-17 | 1 | ok | 1814 | 1880 | 0.965 | 2010-01-04 | 2017-03-17 |  |
| FSLR | 0001274494 | 2022-12-19 | 2025-12-31 | 1 | ok | 761 | 793 | 0.960 | 2022-12-19 | 2025-12-31 |  |
| FTI | 0001135152 | 2010-01-04 | 2017-01-13 | 1 | ok | 1771 | 1835 | 0.965 | 2010-01-04 | 2017-01-13 |  |
| FTI | 0001681459 | 2017-01-17 | 2021-02-11 | 1 | ok | 1025 | 1063 | 0.964 | 2017-01-18 | 2021-02-11 |  |
| FTNT | 0001262039 | 2018-10-11 | 2025-12-31 | 1 | ok | 1815 | 1885 | 0.963 | 2018-10-11 | 2025-12-31 |  |
| FTR | 0000020520 | 2010-01-04 | 2017-03-17 | 1 | ok | 1814 | 1880 | 0.965 | 2010-01-04 | 2017-03-17 |  |
| FTRE | 0001965040 | 2023-07-03 | 2023-07-05 | 1 | partial | 1 | 3 | 0.333 | 2023-07-05 | 2023-07-05 | low business-day coverage (33.3%) |
| FTV | 0001659166 | 2016-07-05 | 2025-12-31 | 1 | ok | 2387 | 2477 | 0.964 | 2016-07-06 | 2025-12-31 |  |
| GAS | 0000072020 | 2010-01-04 | 2011-12-09 | 1 | ok | 490 | 505 | 0.970 | 2010-01-04 | 2011-12-09 |  |
| GAS | 0001004155 | 2011-12-16 | 2016-06-30 | 1 | ok | 1141 | 1185 | 0.963 | 2011-12-16 | 2016-06-30 |  |
| GCI | 0000039899 | 2010-01-04 | 2015-06-26 | 1 | ok | 1380 | 1430 | 0.965 | 2010-01-04 | 2015-06-26 |  |
| GD | 0000040533 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| GDDY | 0001609711 | 2024-06-24 | 2025-12-31 | 1 | ok | 383 | 398 | 0.962 | 2024-06-24 | 2025-12-31 |  |
| GE | 0000040545 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| GEHC | 0001932393 | 2023-01-04 | 2025-12-31 | 1 | ok | 750 | 781 | 0.960 | 2023-01-05 | 2025-12-31 |  |
| GEN | 0000849399 | 2022-11-08 | 2025-12-31 | 1 | ok | 789 | 822 | 0.960 | 2022-11-08 | 2025-12-31 |  |
| GENZ | 0000732485 | 2010-01-04 | 2011-04-01 | 1 | ok | 315 | 325 | 0.969 | 2010-01-04 | 2011-04-01 |  |
| GEV | 0001996810 | 2024-04-02 | 2025-12-31 | 1 | ok | 439 | 457 | 0.961 | 2024-04-03 | 2025-12-31 |  |
| GGP | 0001496048 | 2013-12-10 | 2018-08-27 | 1 | ok | 1187 | 1230 | 0.965 | 2013-12-10 | 2018-08-27 |  |
| GHC | 0000104889 | 2013-11-29 | 2014-09-19 | 1 | ok | 203 | 211 | 0.962 | 2013-11-29 | 2014-09-19 |  |
| GILD | 0000882095 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| GIS | 0000040704 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| GL | 0000320335 | 2019-08-09 | 2025-12-31 | 1 | ok | 1608 | 1669 | 0.963 | 2019-08-09 | 2025-12-31 |  |
| GLW | 0000024741 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| GM | 0001467858 | 2013-06-07 | 2025-12-31 | 1 | ok | 3162 | 3279 | 0.964 | 2013-06-07 | 2025-12-31 |  |
| GMCR | 0001418135 | 2014-03-24 | 2016-03-02 | 1 | ok | 490 | 508 | 0.965 | 2014-03-24 | 2016-03-02 |  |
| GME | 0001326380 | 2010-01-04 | 2016-04-22 | 1 | ok | 1587 | 1645 | 0.965 | 2010-01-04 | 2016-04-22 |  |
| GNRC | 0001474735 | 2021-03-22 | 2025-12-31 | 1 | ok | 1202 | 1248 | 0.963 | 2021-03-22 | 2025-12-31 |  |
| GNW | 0001276520 | 2010-01-04 | 2015-11-17 | 1 | ok | 1480 | 1532 | 0.966 | 2010-01-04 | 2015-11-17 |  |
| GOOG | 0001652044 | 2010-01-04 | 2025-12-31 | 1 | ok | 4023 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| GOOGL | 0001652044 | 2014-04-03 | 2025-12-31 | 1 | ok | 2955 | 3065 | 0.964 | 2014-04-03 | 2025-12-31 |  |
| GPC | 0000040987 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| GPN | 0001123360 | 2016-04-25 | 2025-12-31 | 1 | ok | 2437 | 2528 | 0.964 | 2016-04-25 | 2025-12-31 |  |
| GPS | 0000039911 | 2010-01-04 | 2022-02-02 | 1 | ok | 3043 | 3153 | 0.965 | 2010-01-04 | 2022-02-02 |  |
| GR | 0000042542 | 2010-01-04 | 2012-07-26 | 1 | ok | 647 | 669 | 0.967 | 2010-01-04 | 2012-07-26 |  |
| GRMN | 0001121788 | 2012-12-12 | 2025-12-31 | 1 | ok | 3283 | 3406 | 0.964 | 2012-12-12 | 2025-12-31 |  |
| GS | 0000886982 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| GT | 0000042582 | 2010-01-04 | 2019-02-26 | 1 | ok | 2302 | 2387 | 0.964 | 2010-01-04 | 2019-02-26 |  |
| GTX | 0001735707 | 2018-10-01 | 2018-10-01 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| GWW | 0000277135 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| HAL | 0000045012 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| HAR | 0000800459 | 2010-01-04 | 2017-03-10 | 1 | ok | 1809 | 1875 | 0.965 | 2010-01-04 | 2017-03-10 |  |
| HAS | 0000046080 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| HBAN | 0000049196 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| HBI | 0001359841 | 2015-03-23 | 2021-12-17 | 1 | ok | 1700 | 1760 | 0.966 | 2015-03-23 | 2021-12-17 |  |
| HCA | 0000860730 | 2015-01-27 | 2025-12-31 | 1 | ok | 2750 | 2852 | 0.964 | 2015-01-27 | 2025-12-31 |  |
| HCBK | 0000921847 | 2010-01-04 | 2015-10-30 | 1 | ok | 1468 | 1520 | 0.966 | 2010-01-04 | 2015-10-30 |  |
| HCN | 0000766704 | 2010-01-04 | 2018-02-27 | 1 | ok | 2052 | 2127 | 0.965 | 2010-01-04 | 2018-02-27 |  |
| HCP | 0000765880 | 2010-01-04 | 2019-11-04 | 1 | ok | 2477 | 2566 | 0.965 | 2010-01-04 | 2019-11-04 |  |
| HD | 0000354950 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| HES | 0000004447 | 2010-01-04 | 2025-07-17 | 1 | ok | 3908 | 4054 | 0.964 | 2010-01-04 | 2025-07-17 |  |
| HFC | 0001915657 | 2018-06-18 | 2021-06-03 | 1 | ok | 746 | 774 | 0.964 | 2018-06-18 | 2021-06-03 |  |
| HIG | 0000874766 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| HII | 0001501585 | 2018-01-03 | 2025-12-31 | 1 | ok | 2010 | 2086 | 0.964 | 2018-01-03 | 2025-12-31 |  |
| HLT | 0001585689 | 2017-06-19 | 2025-12-31 | 1 | ok | 2147 | 2228 | 0.964 | 2017-06-19 | 2025-12-31 |  |
| HNZ | 0001637459 | 2010-01-04 | 2013-06-06 | 1 | ok | 862 | 894 | 0.964 | 2010-01-04 | 2013-06-06 |  |
| HOG | 0000793952 | 2010-01-04 | 2020-06-19 | 1 | ok | 2634 | 2730 | 0.965 | 2010-01-04 | 2020-06-19 |  |
| HOLX | 0000859737 | 2016-03-30 | 2025-12-31 | 1 | ok | 2455 | 2546 | 0.964 | 2016-03-30 | 2025-12-31 |  |
| HON | 0000773840 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| HOOD | 0001783879 | 2025-09-22 | 2025-12-31 | 1 | ok | 71 | 73 | 0.973 | 2025-09-22 | 2025-12-31 |  |
| HOT | 0000316206 | 2010-01-04 | 2016-09-22 | 1 | ok | 1693 | 1754 | 0.965 | 2010-01-04 | 2016-09-22 |  |
| HP | 0000046765 | 2010-03-01 | 2020-05-21 | 1 | ok | 2576 | 2669 | 0.965 | 2010-03-01 | 2020-05-21 |  |
| HPE | 0001645590 | 2015-11-02 | 2025-12-31 | 1 | ok | 2555 | 2653 | 0.963 | 2015-11-03 | 2025-12-31 |  |
| HPQ | 0000047217 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| HRB | 0000012659 | 2010-01-04 | 2020-09-18 | 1 | ok | 2697 | 2795 | 0.965 | 2010-01-04 | 2020-09-18 |  |
| HRL | 0000048465 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| HRS | 0000202058 | 2010-01-04 | 2019-06-28 | 1 | ok | 2388 | 2475 | 0.965 | 2010-01-04 | 2019-06-28 |  |
| HSIC | 0001000228 | 2015-03-18 | 2025-12-31 | 1 | ok | 2715 | 2816 | 0.964 | 2015-03-18 | 2025-12-31 |  |
| HSP | 0001274057 | 2010-01-04 | 2015-09-02 | 1 | ok | 1427 | 1478 | 0.965 | 2010-01-04 | 2015-09-02 |  |
| HST | 0001070750 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| HSY | 0000047111 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| HUBB | 0000048898 | 2023-10-18 | 2025-12-31 | 1 | ok | 553 | 576 | 0.960 | 2023-10-18 | 2025-12-31 |  |
| HUM | 0000049071 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| HWM | 0000004281 | 2020-04-01 | 2025-12-31 | 1 | ok | 1446 | 1501 | 0.963 | 2020-04-01 | 2025-12-31 |  |
| IBKR | 0001381197 | 2025-08-28 | 2025-12-31 | 1 | ok | 87 | 90 | 0.967 | 2025-08-28 | 2025-12-31 |  |
| IBM | 0000051143 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ICE | 0001571949 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| IDXX | 0000874716 | 2017-01-05 | 2025-12-31 | 1 | ok | 2260 | 2345 | 0.964 | 2017-01-05 | 2025-12-31 |  |
| IEX | 0000832101 | 2019-08-09 | 2025-12-31 | 1 | ok | 1608 | 1669 | 0.963 | 2019-08-09 | 2025-12-31 |  |
| IFF | 0000051253 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| IGT | 0000353944 | 2010-01-04 | 2014-06-20 | 1 | ok | 1124 | 1165 | 0.965 | 2010-01-04 | 2014-06-20 |  |
| IILG | 0001434620 | 2016-05-13 | 2016-05-13 | 1 | ok | 1 | 1 | 1.000 | 2016-05-13 | 2016-05-13 |  |
| ILMN | 0001110803 | 2015-11-19 | 2024-06-21 | 1 | ok | 2160 | 2242 | 0.963 | 2015-11-19 | 2024-06-21 |  |
| INCY | 0000879169 | 2017-02-28 | 2025-12-31 | 1 | ok | 2224 | 2307 | 0.964 | 2017-02-28 | 2025-12-31 |  |
| INFO | 0001598014 | 2017-06-02 | 2022-02-25 | 1 | ok | 1193 | 1236 | 0.965 | 2017-06-02 | 2022-02-25 |  |
| INTC | 0000050863 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| INTU | 0000896878 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| INVH | 0001687229 | 2022-09-19 | 2025-12-31 | 1 | ok | 825 | 858 | 0.962 | 2022-09-19 | 2025-12-31 |  |
| IP | 0000051434 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| IPG | 0000051644 | 2010-01-04 | 2025-11-26 | 1 | ok | 4001 | 4148 | 0.965 | 2010-01-04 | 2025-11-26 |  |
| IPGP | 0001111928 | 2018-03-07 | 2022-06-17 | 1 | ok | 1080 | 1118 | 0.966 | 2018-03-07 | 2022-06-17 |  |
| IQV | 0001478242 | 2017-11-15 | 2025-12-31 | 1 | ok | 2042 | 2121 | 0.963 | 2017-11-15 | 2025-12-31 |  |
| IR | 0001466258 | 2010-11-17 | 2020-02-28 | 1 | ok | 2335 | 2423 | 0.964 | 2010-11-17 | 2020-02-28 |  |
| IR | 0001699150 | 2020-03-03 | 2025-12-31 | 1 | ok | 1467 | 1522 | 0.964 | 2020-03-03 | 2025-12-31 |  |
| IRM | 0001020569 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ISRG | 0001035267 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| IT | 0000749251 | 2017-04-05 | 2025-12-31 | 1 | ok | 2198 | 2281 | 0.964 | 2017-04-05 | 2025-12-31 |  |
| ITT | 0000216228 | 2010-01-04 | 2011-10-31 | 1 | ok | 462 | 476 | 0.971 | 2010-01-04 | 2011-10-31 |  |
| ITW | 0000049826 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| IVZ | 0000914208 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| J | 0000052988 | 2019-12-10 | 2025-12-31 | 1 | ok | 1523 | 1582 | 0.963 | 2019-12-10 | 2025-12-31 |  |
| JAVA | 0000709519 | 2010-01-04 | 2010-01-26 | 1 | ok | 16 | 17 | 0.941 | 2010-01-04 | 2010-01-26 |  |
| JBHT | 0000728535 | 2015-07-01 | 2025-12-31 | 1 | ok | 2642 | 2741 | 0.964 | 2015-07-01 | 2025-12-31 |  |
| JBL | 0000898293 | 2010-01-04 | 2014-11-04 | 1 | ok | 1219 | 1262 | 0.966 | 2010-01-04 | 2014-11-04 |  |
| JBL | 0000898293 | 2023-12-18 | 2025-12-31 | 1 | ok | 511 | 533 | 0.959 | 2023-12-18 | 2025-12-31 |  |
| JCI | 0000833444 | 2010-01-04 | 2016-09-02 | 1 | ok | 1680 | 1740 | 0.966 | 2010-01-04 | 2016-09-02 |  |
| JCI | 0000833444 | 2016-09-06 | 2025-12-31 | 1 | ok | 2344 | 2432 | 0.964 | 2016-09-06 | 2025-12-31 |  |
| JCP | 0001166126 | 2010-01-04 | 2013-11-29 | 1 | ok | 985 | 1020 | 0.966 | 2010-01-04 | 2013-11-29 |  |
| JDSU | 0000912093 | 2010-01-04 | 2013-12-20 | 1 | ok | 1000 | 1035 | 0.966 | 2010-01-04 | 2013-12-20 |  |
| JEC | 0000052988 | 2010-01-04 | 2019-12-09 | 1 | ok | 2501 | 2591 | 0.965 | 2010-01-04 | 2019-12-09 |  |
| JEF | 0000096223 | 2018-05-24 | 2019-09-25 | 1 | ok | 337 | 350 | 0.963 | 2018-05-24 | 2019-09-25 |  |
| JKHY | 0000779152 | 2018-11-13 | 2025-12-31 | 1 | ok | 1792 | 1862 | 0.962 | 2018-11-13 | 2025-12-31 |  |
| JNJ | 0000200406 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| JNPR | 0001043604 | 2010-01-04 | 2025-07-01 | 1 | ok | 3897 | 4042 | 0.964 | 2010-01-04 | 2025-07-01 |  |
| JNS | 0002043380 | 2010-01-04 | 2011-11-22 | 1 | ok | 478 | 492 | 0.972 | 2010-01-04 | 2011-11-22 |  |
| JOY | 0000801898 | 2011-12-06 | 2015-10-07 | 1 | ok | 965 | 1002 | 0.963 | 2011-12-06 | 2015-10-07 |  |
| JOYG | 0000801898 | 2011-02-28 | 2011-12-05 | 1 | ok | 196 | 201 | 0.975 | 2011-02-28 | 2011-12-05 |  |
| JPM | 0000019617 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| JWN | 0000072333 | 2010-01-04 | 2020-06-19 | 1 | ok | 2634 | 2730 | 0.965 | 2010-01-04 | 2020-06-19 |  |
| K | 0000055067 | 2010-01-04 | 2025-12-10 | 1 | ok | 4010 | 4158 | 0.964 | 2010-01-04 | 2025-12-10 |  |
| KDP | 0001418135 | 2022-06-21 | 2025-12-31 | 1 | ok | 887 | 922 | 0.962 | 2022-06-21 | 2025-12-31 |  |
| KEY | 0000091576 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| KEYS | 0001601046 | 2018-11-06 | 2025-12-31 | 1 | ok | 1797 | 1867 | 0.963 | 2018-11-06 | 2025-12-31 |  |
| KFT | 0001103982 | 2010-01-04 | 2012-10-01 | 1 | ok | 693 | 716 | 0.968 | 2010-01-04 | 2012-10-01 |  |
| KG | 0001047699 | 2010-01-04 | 2010-12-17 | 1 | ok | 243 | 250 | 0.972 | 2010-01-04 | 2010-12-17 |  |
| KHC | 0001637459 | 2015-07-06 | 2025-12-31 | 1 | ok | 2639 | 2738 | 0.964 | 2015-07-07 | 2025-12-31 |  |
| KIM | 0000879101 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| KKR | 0001404912 | 2024-06-24 | 2025-12-31 | 1 | ok | 383 | 398 | 0.962 | 2024-06-24 | 2025-12-31 |  |
| KLAC | 0000319201 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| KLG | 0001959348 | 2023-10-02 | 2023-10-02 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| KMB | 0000055785 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| KMI | 0001506307 | 2012-05-25 | 2025-12-31 | 1 | ok | 3420 | 3549 | 0.964 | 2012-05-25 | 2025-12-31 |  |
| KMX | 0001170010 | 2010-06-28 | 2025-10-30 | 1 | ok | 3861 | 4004 | 0.964 | 2010-06-28 | 2025-10-30 |  |
| KO | 0000021344 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| KORS | 0001530721 | 2013-11-13 | 2018-12-31 | 1 | ok | 1291 | 1339 | 0.964 | 2013-11-13 | 2018-12-31 |  |
| KR | 0000056873 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| KRFT | 0001545158 | 2012-10-02 | 2015-07-02 | 1 | ok | 690 | 718 | 0.961 | 2012-10-03 | 2015-07-02 |  |
| KSS | 0000885639 | 2010-01-04 | 2020-09-18 | 1 | ok | 2697 | 2795 | 0.965 | 2010-01-04 | 2020-09-18 |  |
| KSU | 0000054480 | 2013-05-24 | 2021-12-13 | 1 | ok | 2155 | 2232 | 0.966 | 2013-05-24 | 2021-12-13 |  |
| KTB | 0001760965 | 2019-05-23 | 2019-05-23 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| KVUE | 0001944048 | 2023-08-25 | 2025-12-31 | 1 | ok | 590 | 614 | 0.961 | 2023-08-25 | 2025-12-31 |  |
| L | 0000060086 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| LB | 0000701985 | 2013-12-02 | 2021-08-02 | 1 | ok | 1930 | 2001 | 0.965 | 2013-12-02 | 2021-08-02 |  |
| LDOS | 0001336920 | 2019-08-09 | 2025-12-31 | 1 | ok | 1608 | 1669 | 0.963 | 2019-08-09 | 2025-12-31 |  |
| LEG | 0000058492 | 2010-01-04 | 2021-12-17 | 1 | ok | 3012 | 3120 | 0.965 | 2010-01-04 | 2021-12-17 |  |
| LEN | 0000920760 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| LH | 0000920148 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| LHX | 0000202058 | 2019-07-01 | 2025-12-31 | 1 | ok | 1636 | 1698 | 0.963 | 2019-07-01 | 2025-12-31 |  |
| LIFE | 0001073431 | 2010-01-04 | 2014-01-23 | 1 | ok | 1021 | 1059 | 0.964 | 2010-01-04 | 2014-01-23 |  |
| LII | 0001069202 | 2024-12-23 | 2025-12-31 | 1 | ok | 256 | 268 | 0.955 | 2024-12-23 | 2025-12-31 |  |
| LIN | 0001707925 | 2018-10-31 | 2025-12-31 | 1 | ok | 1800 | 1871 | 0.962 | 2018-11-01 | 2025-12-31 |  |
| LKQ | 0001065696 | 2016-05-23 | 2025-12-19 | 1 | ok | 2410 | 2500 | 0.964 | 2016-05-23 | 2025-12-19 |  |
| LLL | 0001039101 | 2010-01-04 | 2019-06-28 | 1 | ok | 2388 | 2475 | 0.965 | 2010-01-04 | 2019-06-28 |  |
| LLTC | 0000791907 | 2010-01-04 | 2017-03-10 | 1 | ok | 1809 | 1875 | 0.965 | 2010-01-04 | 2017-03-10 |  |
| LLY | 0000059478 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| LM | 0000704051 | 2010-01-04 | 2016-12-01 | 1 | ok | 1742 | 1804 | 0.966 | 2010-01-04 | 2016-12-01 |  |
| LMT | 0000936468 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| LNC | 0000059558 | 2010-01-04 | 2023-09-15 | 1 | ok | 3449 | 3575 | 0.965 | 2010-01-04 | 2023-09-15 |  |
| LNT | 0000352541 | 2016-07-01 | 2025-12-31 | 1 | ok | 2389 | 2479 | 0.964 | 2016-07-01 | 2025-12-31 |  |
| LO | 0001424847 | 2010-01-04 | 2015-06-11 | 1 | ok | 1369 | 1419 | 0.965 | 2010-01-04 | 2015-06-11 |  |
| LOW | 0000060667 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| LRCX | 0000707549 | 2012-06-05 | 2025-12-31 | 1 | ok | 3414 | 3542 | 0.964 | 2012-06-05 | 2025-12-31 |  |
| LSI | 0000703360 | 2010-01-04 | 2014-05-06 | 1 | ok | 1092 | 1132 | 0.965 | 2010-01-04 | 2014-05-06 |  |
| LTD | 0000701985 | 2010-01-04 | 2013-11-29 | 1 | ok | 985 | 1020 | 0.966 | 2010-01-04 | 2013-11-29 |  |
| LUK | 0000096223 | 2010-01-04 | 2018-05-23 | 1 | ok | 2112 | 2188 | 0.965 | 2010-01-04 | 2018-05-23 |  |
| LULU | 0001397187 | 2023-10-18 | 2025-12-31 | 1 | ok | 553 | 576 | 0.960 | 2023-10-18 | 2025-12-31 |  |
| LUMN | 0000018926 | 2020-09-18 | 2023-03-17 | 1 | ok | 628 | 651 | 0.965 | 2020-09-18 | 2023-03-17 |  |
| LUV | 0000092380 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| LVLT | 0000794323 | 2014-11-05 | 2017-10-12 | 1 | ok | 740 | 767 | 0.965 | 2014-11-05 | 2017-10-12 |  |
| LVS | 0001300514 | 2019-10-03 | 2025-12-31 | 1 | ok | 1570 | 1630 | 0.963 | 2019-10-03 | 2025-12-31 |  |
| LW | 0001679273 | 2016-11-10 | 2016-11-10 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| LW | 0001679273 | 2018-12-03 | 2025-12-31 | 1 | ok | 1779 | 1848 | 0.963 | 2018-12-03 | 2025-12-31 |  |
| LXK | 0001001288 | 2010-01-04 | 2012-09-28 | 1 | ok | 692 | 715 | 0.968 | 2010-01-04 | 2012-09-28 |  |
| LYB | 0001489393 | 2012-09-05 | 2025-12-31 | 1 | ok | 3350 | 3476 | 0.964 | 2012-09-05 | 2025-12-31 |  |
| LYV | 0001335258 | 2019-12-23 | 2025-12-31 | 1 | ok | 1514 | 1573 | 0.962 | 2019-12-23 | 2025-12-31 |  |
| M | 0000794367 | 2010-01-04 | 2020-04-03 | 1 | ok | 2581 | 2675 | 0.965 | 2010-01-04 | 2020-04-03 |  |
| MA | 0001141391 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MAA | 0000912595 | 2016-12-02 | 2025-12-31 | 1 | ok | 2282 | 2369 | 0.963 | 2016-12-02 | 2025-12-31 |  |
| MAC | 0000912242 | 2013-05-09 | 2019-12-20 | 1 | ok | 1668 | 1727 | 0.966 | 2013-05-09 | 2019-12-20 |  |
| MAR | 0001048286 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MAS | 0000062996 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MAT | 0000063276 | 2010-01-04 | 2019-06-06 | 1 | ok | 2372 | 2459 | 0.965 | 2010-01-04 | 2019-06-06 |  |
| MBC | 0001941365 | 2022-12-15 | 2022-12-16 | 1 | partial | 1 | 2 | 0.500 | 2022-12-16 | 2022-12-16 | low business-day coverage (50.0%) |
| MCD | 0000063908 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MCHP | 0000827054 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MCK | 0000927653 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MCO | 0001059556 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MDLZ | 0001103982 | 2012-10-02 | 2025-12-31 | 1 | ok | 3331 | 3457 | 0.964 | 2012-10-02 | 2025-12-31 |  |
| MDP | 0000065011 | 2010-01-04 | 2011-01-03 | 1 | ok | 253 | 261 | 0.969 | 2010-01-04 | 2011-01-03 |  |
| MDT | 0001613103 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MEE | 0000037748 | 2010-01-04 | 2011-06-01 | 1 | ok | 356 | 368 | 0.967 | 2010-01-04 | 2011-06-01 |  |
| MET | 0001099219 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| META | 0001326801 | 2022-06-09 | 2025-12-31 | 1 | ok | 894 | 930 | 0.961 | 2022-06-09 | 2025-12-31 |  |
| MFE | 0000890801 | 2010-01-04 | 2011-02-28 | 1 | ok | 291 | 301 | 0.967 | 2010-01-04 | 2011-02-28 |  |
| MGM | 0000789570 | 2017-07-26 | 2025-12-31 | 1 | ok | 2121 | 2201 | 0.964 | 2017-07-26 | 2025-12-31 |  |
| MHFI | 0000064040 | 2013-05-14 | 2016-04-27 | 1 | ok | 745 | 772 | 0.965 | 2013-05-14 | 2016-04-27 |  |
| MHK | 0000851968 | 2013-12-23 | 2025-12-19 | 1 | ok | 3017 | 3130 | 0.964 | 2013-12-23 | 2025-12-19 |  |
| MHP | 0000064040 | 2010-01-04 | 2013-05-13 | 1 | ok | 845 | 876 | 0.965 | 2010-01-04 | 2013-05-13 |  |
| MHS | 0001170650 | 2010-01-04 | 2012-03-30 | 1 | ok | 566 | 585 | 0.968 | 2010-01-04 | 2012-03-30 |  |
| MI | 0001399315 | 2010-01-04 | 2011-07-05 | 1 | ok | 379 | 392 | 0.967 | 2010-01-04 | 2011-07-05 |  |
| MIL | 0000066479 | 2010-01-04 | 2010-07-14 | 1 | ok | 133 | 138 | 0.964 | 2010-01-04 | 2010-07-14 |  |
| MJN | 0001452575 | 2010-01-04 | 2017-06-14 | 1 | ok | 1875 | 1943 | 0.965 | 2010-01-04 | 2017-06-14 |  |
| MKC | 0000063754 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MKTX | 0001278021 | 2019-07-01 | 2025-09-19 | 1 | ok | 1565 | 1625 | 0.963 | 2019-07-01 | 2025-09-19 |  |
| MLM | 0000916076 | 2014-07-02 | 2025-12-31 | 1 | ok | 2893 | 3001 | 0.964 | 2014-07-02 | 2025-12-31 |  |
| MMC | 0000062709 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MMI | 0001495569 | 2011-01-04 | 2012-05-21 | 1 | ok | 347 | 360 | 0.964 | 2011-01-05 | 2012-05-21 |  |
| MMM | 0000066740 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MNK | 0001567892 | 2014-08-19 | 2017-07-25 | 1 | ok | 739 | 766 | 0.965 | 2014-08-19 | 2017-07-25 |  |
| MNST | 0000865752 | 2012-06-29 | 2025-12-31 | 1 | ok | 3396 | 3524 | 0.964 | 2012-06-29 | 2025-12-31 |  |
| MO | 0000764180 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MOH | 0001179929 | 2022-03-02 | 2025-12-31 | 1 | ok | 963 | 1001 | 0.962 | 2022-03-02 | 2025-12-31 |  |
| MOLX | 0000067472 | 2010-01-04 | 2013-12-06 | 1 | ok | 990 | 1025 | 0.966 | 2010-01-04 | 2013-12-06 |  |
| MON | 0001110783 | 2010-01-04 | 2018-06-06 | 1 | ok | 2121 | 2198 | 0.965 | 2010-01-04 | 2018-06-06 |  |
| MOS | 0001285785 | 2011-09-26 | 2025-12-31 | 1 | ok | 3588 | 3723 | 0.964 | 2011-09-26 | 2025-12-31 |  |
| MOT | 0000068505 | 2010-01-04 | 2011-01-03 | 1 | ok | 253 | 261 | 0.969 | 2010-01-04 | 2011-01-03 |  |
| MPC | 0001510295 | 2011-07-01 | 2025-12-31 | 1 | ok | 3646 | 3784 | 0.964 | 2011-07-05 | 2025-12-31 |  |
| MPWR | 0001280452 | 2021-02-12 | 2025-12-31 | 1 | ok | 1227 | 1274 | 0.963 | 2021-02-12 | 2025-12-31 |  |
| MRK | 0000310158 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MRNA | 0001682852 | 2021-07-21 | 2025-12-31 | 1 | ok | 1118 | 1161 | 0.963 | 2021-07-21 | 2025-12-31 |  |
| MRO | 0000101778 | 2010-01-04 | 2024-11-21 | 1 | ok | 3748 | 3884 | 0.965 | 2010-01-04 | 2024-11-21 |  |
| MS | 0000895421 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MSCI | 0001408198 | 2018-04-04 | 2025-12-31 | 1 | ok | 1948 | 2021 | 0.964 | 2018-04-04 | 2025-12-31 |  |
| MSFT | 0000789019 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MSI | 0000068505 | 2011-01-04 | 2025-12-31 | 1 | ok | 3771 | 3912 | 0.964 | 2011-01-04 | 2025-12-31 |  |
| MTB | 0000036270 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MTCH | 0000891103 | 2021-09-20 | 2025-12-31 | 1 | ok | 1076 | 1118 | 0.962 | 2021-09-20 | 2025-12-31 |  |
| MTD | 0001037646 | 2016-09-06 | 2025-12-31 | 1 | ok | 2344 | 2432 | 0.964 | 2016-09-06 | 2025-12-31 |  |
| MU | 0000723125 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| MUR | 0000717423 | 2010-01-04 | 2017-07-25 | 1 | ok | 1903 | 1972 | 0.965 | 2010-01-04 | 2017-07-25 |  |
| MWV | 0001159297 | 2010-01-04 | 2015-07-01 | 1 | ok | 1383 | 1433 | 0.965 | 2010-01-04 | 2015-07-01 |  |
| MWW | 0001020416 | 2010-01-04 | 2011-12-16 | 1 | ok | 495 | 510 | 0.971 | 2010-01-04 | 2011-12-16 |  |
| MXIM | 0000743316 | 2018-12-03 | 2021-08-25 | 1 | ok | 687 | 713 | 0.964 | 2018-12-03 | 2021-08-25 |  |
| MYL | 0001792044 | 2010-01-04 | 2020-11-16 | 1 | ok | 2738 | 2836 | 0.965 | 2010-01-04 | 2020-11-16 |  |
| NAVI | 0001593538 | 2014-05-01 | 2018-06-04 | 1 | ok | 1030 | 1068 | 0.964 | 2014-05-02 | 2018-06-04 |  |
| NBL | 0000072207 | 2010-01-04 | 2020-10-02 | 1 | ok | 2707 | 2805 | 0.965 | 2010-01-04 | 2020-10-02 |  |
| NBR | 0001163739 | 2010-01-04 | 2015-03-20 | 1 | ok | 1312 | 1360 | 0.965 | 2010-01-04 | 2015-03-20 |  |
| NCLH | 0001513761 | 2017-10-13 | 2025-12-31 | 1 | ok | 2065 | 2144 | 0.963 | 2017-10-13 | 2025-12-31 |  |
| NDAQ | 0001120193 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| NDSN | 0000072331 | 2022-02-15 | 2025-12-31 | 1 | ok | 973 | 1012 | 0.961 | 2022-02-15 | 2025-12-31 |  |
| NE | 0001895262 | 2011-01-18 | 2015-07-17 | 1 | ok | 1132 | 1174 | 0.964 | 2011-01-18 | 2015-07-17 |  |
| NEE | 0000753308 | 2010-06-23 | 2025-12-31 | 1 | ok | 3906 | 4051 | 0.964 | 2010-06-23 | 2025-12-31 |  |
| NEM | 0001164727 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| NFLX | 0001065280 | 2010-12-20 | 2025-12-31 | 1 | ok | 3781 | 3923 | 0.964 | 2010-12-20 | 2025-12-31 |  |
| NFX | 0000912750 | 2010-12-20 | 2019-02-13 | 1 | ok | 2051 | 2128 | 0.964 | 2010-12-20 | 2019-02-13 |  |
| NGVT | 0001653477 | 2016-05-16 | 2016-05-16 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| NI | 0001111711 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| NKE | 0000320187 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| NKTR | 0000906709 | 2018-03-19 | 2019-10-02 | 1 | ok | 389 | 403 | 0.965 | 2018-03-19 | 2019-10-02 |  |
| NLOK | 0000849399 | 2019-11-05 | 2022-11-07 | 1 | ok | 758 | 785 | 0.966 | 2019-11-05 | 2022-11-07 |  |
| NLSN | 0001492633 | 2013-07-09 | 2022-10-11 | 1 | ok | 2333 | 2416 | 0.966 | 2013-07-09 | 2022-10-11 |  |
| NOC | 0001133421 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| NOV | 0001021860 | 2010-01-04 | 2021-09-17 | 1 | ok | 2948 | 3055 | 0.965 | 2010-01-04 | 2021-09-17 |  |
| NOVL | 0000758004 | 2010-01-04 | 2011-04-27 | 1 | ok | 332 | 343 | 0.968 | 2010-01-04 | 2011-04-27 |  |
| NOW | 0001373715 | 2019-11-21 | 2025-12-31 | 1 | ok | 1535 | 1595 | 0.962 | 2019-11-21 | 2025-12-31 |  |
| NRG | 0001013871 | 2010-01-29 | 2025-12-31 | 1 | ok | 4006 | 4154 | 0.964 | 2010-01-29 | 2025-12-31 |  |
| NSC | 0000702165 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| NSM | 0000070530 | 2010-01-04 | 2011-09-23 | 1 | ok | 436 | 450 | 0.969 | 2010-01-04 | 2011-09-23 |  |
| NTAP | 0001002047 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| NTRS | 0000073124 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| NU | 0000072741 | 2010-01-04 | 2015-02-18 | 1 | ok | 1290 | 1338 | 0.964 | 2010-01-04 | 2015-02-18 |  |
| NUE | 0000073309 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| NVDA | 0001045810 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| NVLS | 0000836106 | 2010-01-04 | 2012-06-04 | 1 | ok | 610 | 631 | 0.967 | 2010-01-04 | 2012-06-04 |  |
| NVR | 0000906163 | 2019-09-26 | 2025-12-31 | 1 | ok | 1575 | 1635 | 0.963 | 2019-09-26 | 2025-12-31 |  |
| NWL | 0000814453 | 2010-01-04 | 2023-09-15 | 1 | ok | 3449 | 3575 | 0.965 | 2010-01-04 | 2023-09-15 |  |
| NWS | 0001564708 | 2015-09-21 | 2025-12-31 | 1 | ok | 2586 | 2683 | 0.964 | 2015-09-21 | 2025-12-31 |  |
| NWSA | 0001308161 | 2010-01-04 | 2013-06-28 | 1 | ok | 878 | 910 | 0.965 | 2010-01-04 | 2013-06-28 |  |
| NWSA | 0001564708 | 2013-07-01 | 2025-12-31 | 1 | ok | 3145 | 3263 | 0.964 | 2013-07-02 | 2025-12-31 |  |
| NXPI | 0001413447 | 2021-03-22 | 2025-12-31 | 1 | ok | 1202 | 1248 | 0.963 | 2021-03-22 | 2025-12-31 |  |
| NYT | 0000071691 | 2010-01-04 | 2010-12-17 | 1 | ok | 243 | 250 | 0.972 | 2010-01-04 | 2010-12-17 |  |
| NYX | 0001368007 | 2010-01-04 | 2013-11-12 | 1 | ok | 973 | 1007 | 0.966 | 2010-01-04 | 2013-11-12 |  |
| O | 0000726728 | 2015-04-07 | 2025-12-31 | 1 | ok | 2702 | 2802 | 0.964 | 2015-04-07 | 2025-12-31 |  |
| ODFL | 0000878927 | 2019-12-09 | 2025-12-31 | 1 | ok | 1524 | 1583 | 0.963 | 2019-12-09 | 2025-12-31 |  |
| ODP | 0000800240 | 2010-01-04 | 2010-12-17 | 1 | ok | 243 | 250 | 0.972 | 2010-01-04 | 2010-12-17 |  |
| OGN | 0001821825 | 2021-06-03 | 2023-10-17 | 1 | ok | 597 | 619 | 0.964 | 2021-06-04 | 2023-10-17 |  |
| OI | 0000812074 | 2010-01-04 | 2016-12-01 | 1 | ok | 1742 | 1804 | 0.966 | 2010-01-04 | 2016-12-01 |  |
| OKE | 0001039684 | 2010-03-15 | 2025-12-31 | 1 | ok | 3976 | 4123 | 0.964 | 2010-03-15 | 2025-12-31 |  |
| OMC | 0000029989 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ON | 0001097864 | 2022-06-21 | 2025-12-31 | 1 | ok | 887 | 922 | 0.962 | 2022-06-21 | 2025-12-31 |  |
| ORCL | 0001341439 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ORLY | 0000898173 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| OTIS | 0001781335 | 2020-04-03 | 2025-12-31 | 1 | ok | 1443 | 1499 | 0.963 | 2020-04-06 | 2025-12-31 |  |
| OXY | 0000797468 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PANW | 0001327567 | 2023-06-20 | 2025-12-31 | 1 | ok | 637 | 662 | 0.962 | 2023-06-20 | 2025-12-31 |  |
| PARA | 0002041610 | 2022-02-17 | 2025-08-06 | 1 | ok | 869 | 905 | 0.960 | 2022-02-17 | 2025-08-06 |  |
| PAYC | 0001590955 | 2020-01-28 | 2025-12-31 | 1 | ok | 1491 | 1547 | 0.964 | 2020-01-28 | 2025-12-31 |  |
| PAYX | 0000723531 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PBCT | 0001378946 | 2010-01-04 | 2022-04-01 | 1 | ok | 3084 | 3195 | 0.965 | 2010-01-04 | 2022-04-01 |  |
| PBG | 0001076405 | 2010-01-04 | 2010-02-26 | 1 | ok | 38 | 40 | 0.950 | 2010-01-04 | 2010-02-26 |  |
| PBI | 0000078814 | 2010-01-04 | 2017-02-28 | 1 | ok | 1801 | 1867 | 0.965 | 2010-01-04 | 2017-02-28 |  |
| PCAR | 0000075362 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PCG | 0001004980 | 2010-01-04 | 2019-01-17 | 1 | ok | 2276 | 2359 | 0.965 | 2010-01-04 | 2019-01-17 |  |
| PCG | 0001004980 | 2022-10-03 | 2025-12-31 | 1 | ok | 815 | 848 | 0.961 | 2022-10-03 | 2025-12-31 |  |
| PCL | 0000849213 | 2010-01-04 | 2016-02-19 | 1 | ok | 1543 | 1600 | 0.964 | 2010-01-04 | 2016-02-19 |  |
| PCLN | 0001075531 | 2010-01-04 | 2018-02-26 | 1 | ok | 2051 | 2126 | 0.965 | 2010-01-04 | 2018-02-26 |  |
| PCP | 0000079958 | 2010-01-04 | 2016-01-29 | 1 | ok | 1529 | 1585 | 0.965 | 2010-01-04 | 2016-01-29 |  |
| PCS | 0001283699 | 2010-01-04 | 2013-04-30 | 1 | ok | 836 | 867 | 0.964 | 2010-01-04 | 2013-04-30 |  |
| PDCO | 0000891024 | 2010-01-04 | 2018-03-16 | 1 | ok | 2065 | 2140 | 0.965 | 2010-01-04 | 2018-03-16 |  |
| PEAK | 0000765880 | 2019-11-05 | 2024-03-01 | 1 | ok | 1087 | 1129 | 0.963 | 2019-11-05 | 2024-03-01 |  |
| PEG | 0000788784 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PENN | 0000921738 | 2021-03-22 | 2022-09-16 | 1 | ok | 377 | 390 | 0.967 | 2021-03-22 | 2022-09-16 |  |
| PEP | 0000077476 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PETM | 0000863157 | 2012-10-05 | 2015-03-11 | 1 | ok | 609 | 634 | 0.961 | 2012-10-05 | 2015-03-11 |  |
| PFE | 0000078003 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PFG | 0001126328 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PG | 0000080424 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PGN | 0001094093 | 2010-01-04 | 2012-06-29 | 1 | ok | 629 | 650 | 0.968 | 2010-01-04 | 2012-06-29 |  |
| PGR | 0000080661 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PH | 0000076334 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PHIN | 0001968915 | 2023-07-05 | 2023-07-05 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| PHM | 0000822416 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PKG | 0000075677 | 2017-07-26 | 2025-12-31 | 1 | ok | 2121 | 2201 | 0.964 | 2017-07-26 | 2025-12-31 |  |
| PKI | 0000031791 | 2010-01-04 | 2023-05-15 | 1 | ok | 3364 | 3486 | 0.965 | 2010-01-04 | 2023-05-15 |  |
| PLD | 0001045609 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PLL | 0000075829 | 2010-01-04 | 2015-08-28 | 1 | ok | 1424 | 1475 | 0.965 | 2010-01-04 | 2015-08-28 |  |
| PLTR | 0001321655 | 2024-09-23 | 2025-12-31 | 1 | ok | 320 | 333 | 0.961 | 2024-09-23 | 2025-12-31 |  |
| PM | 0001413329 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PNC | 0000713676 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PNR | 0000077360 | 2012-10-01 | 2025-12-31 | 1 | ok | 3331 | 3458 | 0.963 | 2012-10-02 | 2025-12-31 |  |
| PNW | 0000764622 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PODD | 0001145197 | 2023-03-15 | 2025-12-31 | 1 | ok | 703 | 731 | 0.962 | 2023-03-15 | 2025-12-31 |  |
| POM | 0001135971 | 2010-01-04 | 2016-03-23 | 1 | ok | 1566 | 1623 | 0.965 | 2010-01-04 | 2016-03-23 |  |
| POOL | 0000945841 | 2020-10-07 | 2025-12-31 | 1 | ok | 1315 | 1366 | 0.963 | 2020-10-07 | 2025-12-31 |  |
| PPG | 0000079879 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PPL | 0000922224 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PRGO | 0001585364 | 2011-12-19 | 2021-09-17 | 1 | ok | 2453 | 2545 | 0.964 | 2011-12-19 | 2021-09-17 |  |
| PRU | 0001137774 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PSA | 0001393311 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PSKY | 0002041610 | 2025-08-07 | 2025-12-31 | 1 | ok | 101 | 105 | 0.962 | 2025-08-08 | 2025-12-31 |  |
| PSX | 0001534701 | 2012-05-01 | 2025-12-31 | 1 | ok | 3437 | 3567 | 0.964 | 2012-05-02 | 2025-12-31 |  |
| PTC | 0000857005 | 2021-04-20 | 2025-12-31 | 1 | ok | 1182 | 1227 | 0.963 | 2021-04-20 | 2025-12-31 |  |
| PTV | 0001089976 | 2010-01-04 | 2010-11-16 | 1 | ok | 221 | 227 | 0.974 | 2010-01-04 | 2010-11-16 |  |
| PVH | 0000078239 | 2013-02-14 | 2022-09-16 | 1 | ok | 2415 | 2502 | 0.965 | 2013-02-14 | 2022-09-16 |  |
| PWR | 0001050915 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| PX | 0001707925 | 2010-01-04 | 2018-10-30 | 1 | ok | 2223 | 2302 | 0.966 | 2010-01-04 | 2018-10-30 |  |
| PXD | 0001038357 | 2010-01-04 | 2024-05-02 | 1 | ok | 3607 | 3739 | 0.965 | 2010-01-04 | 2024-05-02 |  |
| PYPL | 0001633917 | 2015-07-20 | 2025-12-31 | 1 | ok | 2629 | 2728 | 0.964 | 2015-07-21 | 2025-12-31 |  |
| Q | 0001037949 | 2010-01-04 | 2011-03-31 | 1 | ok | 314 | 324 | 0.969 | 2010-01-04 | 2011-03-31 |  |
| Q | 0001478242 | 2017-08-29 | 2017-11-14 | 1 | ok | 55 | 56 | 0.982 | 2017-08-29 | 2017-11-14 |  |
| Q | 0002058873 | 2025-11-03 | 2025-12-31 | 1 | ok | 40 | 43 | 0.930 | 2025-11-04 | 2025-12-31 |  |
| QCOM | 0000804328 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| QCP | 0001677203 | 2016-11-01 | 2016-11-01 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| QEP | 0001108827 | 2010-07-01 | 2015-06-30 | 1 | ok | 1257 | 1304 | 0.964 | 2010-07-02 | 2015-06-30 |  |
| QLGC | 0000918386 | 2010-01-04 | 2011-01-14 | 1 | ok | 262 | 270 | 0.970 | 2010-01-04 | 2011-01-14 |  |
| QRVO | 0001604778 | 2015-06-12 | 2024-12-20 | 1 | ok | 2399 | 2486 | 0.965 | 2015-06-12 | 2024-12-20 |  |
| R | 0000085961 | 2010-01-04 | 2017-06-16 | 1 | ok | 1877 | 1945 | 0.965 | 2010-01-04 | 2017-06-16 |  |
| RAI | 0001275283 | 2010-01-04 | 2017-07-24 | 1 | ok | 1902 | 1971 | 0.965 | 2010-01-04 | 2017-07-24 |  |
| RCL | 0000884887 | 2014-12-05 | 2025-12-31 | 1 | ok | 2784 | 2889 | 0.964 | 2014-12-05 | 2025-12-31 |  |
| RDC | 0000085408 | 2010-01-04 | 2014-08-18 | 1 | ok | 1164 | 1206 | 0.965 | 2010-01-04 | 2014-08-18 |  |
| RE | 0001095073 | 2017-06-19 | 2023-07-07 | 1 | ok | 1523 | 1580 | 0.964 | 2017-06-19 | 2023-07-07 |  |
| REG | 0000910606 | 2017-03-02 | 2025-12-31 | 1 | ok | 2222 | 2305 | 0.964 | 2017-03-02 | 2025-12-31 |  |
| REGN | 0000872589 | 2013-05-01 | 2025-12-31 | 1 | ok | 3187 | 3306 | 0.964 | 2013-05-01 | 2025-12-31 |  |
| REZI | 0001740332 | 2018-10-29 | 2018-10-29 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| RF | 0001281761 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| RHI | 0000315213 | 2010-01-04 | 2024-06-21 | 1 | ok | 3641 | 3775 | 0.965 | 2010-01-04 | 2024-06-21 |  |
| RHT | 0001087423 | 2010-01-04 | 2019-07-08 | 1 | ok | 2393 | 2481 | 0.965 | 2010-01-04 | 2019-07-08 |  |
| RIG | 0001451505 | 2013-10-29 | 2017-07-25 | 1 | ok | 941 | 976 | 0.964 | 2013-10-29 | 2017-07-25 |  |
| RJF | 0000720005 | 2017-03-20 | 2025-12-31 | 1 | ok | 2210 | 2293 | 0.964 | 2017-03-20 | 2025-12-31 |  |
| RL | 0001037038 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| RMD | 0000943819 | 2017-07-26 | 2025-12-31 | 1 | ok | 2121 | 2201 | 0.964 | 2017-07-26 | 2025-12-31 |  |
| ROK | 0001024478 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ROL | 0000084839 | 2018-10-01 | 2025-12-31 | 1 | ok | 1823 | 1893 | 0.963 | 2018-10-01 | 2025-12-31 |  |
| ROP | 0000882835 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| ROST | 0000745732 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| RRC | 0000315852 | 2010-01-04 | 2018-06-15 | 1 | ok | 2128 | 2205 | 0.965 | 2010-01-04 | 2018-06-15 |  |
| RRD | 0000029669 | 2010-01-04 | 2012-12-11 | 1 | ok | 741 | 767 | 0.966 | 2010-01-04 | 2012-12-11 |  |
| RSG | 0001060391 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| RSH | 0000096289 | 2010-01-04 | 2011-06-30 | 1 | ok | 377 | 389 | 0.969 | 2010-01-04 | 2011-06-30 |  |
| RTN | 0001047122 | 2010-01-04 | 2020-04-02 | 1 | ok | 2580 | 2674 | 0.965 | 2010-01-04 | 2020-04-02 |  |
| RTX | 0000101829 | 2020-04-03 | 2025-12-31 | 1 | ok | 1444 | 1499 | 0.963 | 2020-04-03 | 2025-12-31 |  |
| RVTY | 0000031791 | 2023-05-16 | 2025-12-31 | 1 | ok | 660 | 687 | 0.961 | 2023-05-16 | 2025-12-31 |  |
| RX | 0001595262 | 2010-01-04 | 2010-02-25 | 1 | ok | 37 | 39 | 0.949 | 2010-01-04 | 2010-02-25 |  |
| S | 0000101830 | 2010-01-04 | 2013-07-08 | 1 | ok | 883 | 916 | 0.964 | 2010-01-04 | 2013-07-08 |  |
| SAI | 0001336920 | 2010-01-04 | 2013-09-20 | 1 | ok | 936 | 970 | 0.965 | 2010-01-04 | 2013-09-20 |  |
| SBAC | 0001034054 | 2017-09-01 | 2025-12-31 | 1 | ok | 2094 | 2174 | 0.963 | 2017-09-01 | 2025-12-31 |  |
| SBNY | 0001288784 | 2021-12-20 | 2023-03-10 | 1 | ok | 307 | 320 | 0.959 | 2021-12-20 | 2023-03-10 |  |
| SBUX | 0000829224 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| SCG | 0000754737 | 2010-01-04 | 2018-12-31 | 1 | ok | 2264 | 2346 | 0.965 | 2010-01-04 | 2018-12-31 |  |
| SCHW | 0000316709 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| SE | 0001373835 | 2010-01-04 | 2017-02-24 | 1 | ok | 1799 | 1865 | 0.965 | 2010-01-04 | 2017-02-24 |  |
| SEDG | 0001419612 | 2021-12-20 | 2023-12-15 | 1 | ok | 501 | 520 | 0.963 | 2021-12-20 | 2023-12-15 |  |
| SEE | 0001012100 | 2010-01-04 | 2023-12-15 | 1 | ok | 3513 | 3640 | 0.965 | 2010-01-04 | 2023-12-15 |  |
| SHLD | 0001310067 | 2010-01-04 | 2012-09-04 | 1 | ok | 674 | 697 | 0.967 | 2010-01-04 | 2012-09-04 |  |
| SHW | 0000089800 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| SIAL | 0000090185 | 2010-01-04 | 2015-11-17 | 1 | ok | 1480 | 1532 | 0.966 | 2010-01-04 | 2015-11-17 |  |
| SIG | 0000832988 | 2015-07-29 | 2018-03-16 | 1 | ok | 664 | 688 | 0.965 | 2015-07-29 | 2018-03-16 |  |
| SII | 0000721083 | 2010-01-04 | 2010-08-26 | 1 | ok | 164 | 169 | 0.970 | 2010-01-04 | 2010-08-26 |  |
| SIVB | 0000719739 | 2018-03-19 | 2023-03-10 | 1 | ok | 1254 | 1300 | 0.965 | 2018-03-19 | 2023-03-10 |  |
| SJM | 0000091419 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| SLB | 0000087347 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| SLE | 0000023666 | 2010-01-04 | 2012-06-28 | 1 | ok | 628 | 649 | 0.968 | 2010-01-04 | 2012-06-28 |  |
| SLG | 0001040971 | 2015-03-23 | 2021-03-19 | 1 | ok | 1510 | 1565 | 0.965 | 2015-03-23 | 2021-03-19 |  |
| SLM | 0001593538 | 2010-01-04 | 2014-04-30 | 1 | ok | 1088 | 1128 | 0.965 | 2010-01-04 | 2014-04-30 |  |
| SMCI | 0001375365 | 2024-03-18 | 2025-12-31 | 1 | ok | 450 | 468 | 0.962 | 2024-03-18 | 2025-12-31 |  |
| SNA | 0000091440 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| SNDK | 0001000180 | 2010-01-04 | 2016-05-11 | 1 | ok | 1600 | 1658 | 0.965 | 2010-01-04 | 2016-05-11 |  |
| SNDK | 0002023554 | 2025-11-28 | 2025-12-31 | 1 | ok | 23 | 24 | 0.958 | 2025-11-28 | 2025-12-31 |  |
| SNI | 0001430602 | 2010-01-04 | 2018-03-06 | 1 | ok | 2057 | 2132 | 0.965 | 2010-01-04 | 2018-03-06 |  |
| SNPS | 0000883241 | 2017-03-16 | 2025-12-31 | 1 | ok | 2212 | 2295 | 0.964 | 2017-03-16 | 2025-12-31 |  |
| SO | 0000092122 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| SOLS | 0002064953 | 2025-10-30 | 2025-12-19 | 1 | ok | 35 | 37 | 0.946 | 2025-10-31 | 2025-12-19 |  |
| SOLV | 0001964738 | 2024-04-01 | 2025-12-31 | 1 | ok | 440 | 458 | 0.961 | 2024-04-02 | 2025-12-31 |  |
| SPG | 0001063761 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| SPGI | 0000064040 | 2016-04-28 | 2025-12-31 | 1 | ok | 2434 | 2525 | 0.964 | 2016-04-28 | 2025-12-31 |  |
| SPLS | 0000791519 | 2010-01-04 | 2017-09-12 | 1 | ok | 1937 | 2007 | 0.965 | 2010-01-04 | 2017-09-12 |  |
| SRCL | 0000861878 | 2010-01-04 | 2018-11-30 | 1 | ok | 2245 | 2325 | 0.966 | 2010-01-04 | 2018-11-30 |  |
| SRE | 0001032208 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| STE | 0001757898 | 2019-12-23 | 2025-12-31 | 1 | ok | 1514 | 1573 | 0.962 | 2019-12-23 | 2025-12-31 |  |
| STI | 0000750556 | 2010-01-04 | 2019-12-06 | 1 | ok | 2500 | 2590 | 0.965 | 2010-01-04 | 2019-12-06 |  |
| STJ | 0000203077 | 2010-01-04 | 2017-01-04 | 1 | ok | 1764 | 1828 | 0.965 | 2010-01-04 | 2017-01-04 |  |
| STLD | 0001022671 | 2022-12-22 | 2025-12-31 | 1 | ok | 758 | 790 | 0.959 | 2022-12-22 | 2025-12-31 |  |
| STR | 0000751652 | 2010-01-04 | 2010-06-30 | 1 | ok | 124 | 128 | 0.969 | 2010-01-04 | 2010-06-30 |  |
| STT | 0000093751 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| STX | 0001137789 | 2012-07-02 | 2025-12-31 | 1 | ok | 3395 | 3523 | 0.964 | 2012-07-02 | 2025-12-31 |  |
| STZ | 0000016918 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| SUN | 0000095304 | 2010-01-04 | 2012-10-04 | 1 | ok | 696 | 719 | 0.968 | 2010-01-04 | 2012-10-04 |  |
| SVU | 0000095521 | 2010-01-04 | 2012-04-30 | 1 | ok | 586 | 606 | 0.967 | 2010-01-04 | 2012-04-30 |  |
| SW | 0002005951 | 2024-07-08 | 2025-12-31 | 1 | ok | 373 | 388 | 0.961 | 2024-07-09 | 2025-12-31 |  |
| SWK | 0000093556 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| SWKS | 0000004127 | 2015-03-12 | 2025-12-31 | 1 | ok | 2719 | 2820 | 0.964 | 2015-03-12 | 2025-12-31 |  |
| SWN | 0000007332 | 2010-01-04 | 2017-04-03 | 1 | ok | 1825 | 1891 | 0.965 | 2010-01-04 | 2017-04-03 |  |
| SWY | 0000086144 | 2010-01-04 | 2015-01-26 | 1 | ok | 1274 | 1321 | 0.964 | 2010-01-04 | 2015-01-26 |  |
| SYF | 0001601712 | 2015-11-18 | 2025-12-31 | 1 | ok | 2544 | 2641 | 0.963 | 2015-11-18 | 2025-12-31 |  |
| SYK | 0000310764 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| SYMC | 0000849399 | 2010-01-04 | 2019-11-04 | 1 | ok | 2477 | 2566 | 0.965 | 2010-01-04 | 2019-11-04 |  |
| SYY | 0000096021 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| T | 0000732717 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| TAP | 0000024545 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| TDC | 0000816761 | 2010-01-04 | 2017-06-16 | 1 | ok | 1877 | 1945 | 0.965 | 2010-01-04 | 2017-06-16 |  |
| TDG | 0001260221 | 2016-06-03 | 2025-12-31 | 1 | ok | 2409 | 2499 | 0.964 | 2016-06-03 | 2025-12-31 |  |
| TDY | 0001094285 | 2020-06-22 | 2025-12-31 | 1 | ok | 1390 | 1443 | 0.963 | 2020-06-22 | 2025-12-31 |  |
| TE | 0000350563 | 2010-01-04 | 2016-06-30 | 1 | ok | 1635 | 1694 | 0.965 | 2010-01-04 | 2016-06-30 |  |
| TECH | 0000842023 | 2021-08-30 | 2025-12-31 | 1 | ok | 1090 | 1133 | 0.962 | 2021-08-30 | 2025-12-31 |  |
| TEG | 0000916863 | 2010-01-04 | 2015-06-29 | 1 | ok | 1381 | 1431 | 0.965 | 2010-01-04 | 2015-06-29 |  |
| TEL | 0001385157 | 2011-10-17 | 2025-12-31 | 1 | ok | 3573 | 3708 | 0.964 | 2011-10-17 | 2025-12-31 |  |
| TER | 0000097210 | 2010-01-04 | 2013-12-20 | 1 | ok | 1000 | 1035 | 0.966 | 2010-01-04 | 2013-12-20 |  |
| TER | 0000097210 | 2020-09-21 | 2025-12-31 | 1 | ok | 1327 | 1378 | 0.963 | 2020-09-21 | 2025-12-31 |  |
| TFC | 0000092230 | 2019-12-09 | 2025-12-31 | 1 | ok | 1524 | 1583 | 0.963 | 2019-12-09 | 2025-12-31 |  |
| TFCF | 0001308161 | 2019-03-19 | 2019-03-19 | 1 | ok | 1 | 1 | 1.000 | 2019-03-19 | 2019-03-19 |  |
| TFCFA | 0001308161 | 2019-03-19 | 2019-03-19 | 1 | ok | 1 | 1 | 1.000 | 2019-03-19 | 2019-03-19 |  |
| TFX | 0000096943 | 2019-01-18 | 2025-03-21 | 1 | ok | 1552 | 1611 | 0.963 | 2019-01-18 | 2025-03-21 |  |
| TGNA | 0000039899 | 2015-06-29 | 2017-06-01 | 1 | ok | 486 | 504 | 0.964 | 2015-06-29 | 2017-06-01 |  |
| TGT | 0000027419 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| THC | 0000070318 | 2010-01-04 | 2016-04-15 | 1 | ok | 1582 | 1640 | 0.965 | 2010-01-04 | 2016-04-15 |  |
| TIE | 0001011657 | 2010-01-04 | 2012-12-21 | 1 | ok | 749 | 775 | 0.966 | 2010-01-04 | 2012-12-21 |  |
| TIF | 0000098246 | 2010-01-04 | 2021-01-06 | 1 | ok | 2772 | 2873 | 0.965 | 2010-01-04 | 2021-01-06 |  |
| TJX | 0000109198 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| TKO | 0001973266 | 2025-03-24 | 2025-12-31 | 1 | ok | 196 | 203 | 0.966 | 2025-03-24 | 2025-12-31 |  |
| TLAB | 0000317771 | 2010-01-04 | 2011-12-20 | 1 | ok | 497 | 512 | 0.971 | 2010-01-04 | 2011-12-20 |  |
| TMK | 0000320335 | 2010-01-04 | 2019-08-08 | 1 | ok | 2416 | 2504 | 0.965 | 2010-01-04 | 2019-08-08 |  |
| TMO | 0000097745 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| TMUS | 0001283699 | 2019-07-15 | 2025-12-31 | 1 | ok | 1627 | 1688 | 0.964 | 2019-07-15 | 2025-12-31 |  |
| TPL | 0001811074 | 2024-11-26 | 2025-12-31 | 1 | ok | 274 | 287 | 0.955 | 2024-11-26 | 2025-12-31 |  |
| TPR | 0001116132 | 2017-10-31 | 2025-12-31 | 1 | ok | 2053 | 2132 | 0.963 | 2017-10-31 | 2025-12-31 |  |
| TRGP | 0001389170 | 2022-10-12 | 2025-12-31 | 1 | ok | 808 | 841 | 0.961 | 2022-10-12 | 2025-12-31 |  |
| TRIP | 0001526520 | 2011-12-21 | 2019-12-20 | 1 | ok | 2012 | 2088 | 0.964 | 2011-12-22 | 2019-12-20 |  |
| TRMB | 0000864749 | 2021-01-21 | 2025-12-31 | 1 | ok | 1243 | 1290 | 0.964 | 2021-01-21 | 2025-12-31 |  |
| TROW | 0001113169 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| TRV | 0000086312 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| TSCO | 0000916365 | 2014-01-24 | 2025-12-31 | 1 | ok | 3003 | 3114 | 0.964 | 2014-01-24 | 2025-12-31 |  |
| TSLA | 0001318605 | 2020-12-21 | 2025-12-31 | 1 | ok | 1263 | 1313 | 0.962 | 2020-12-21 | 2025-12-31 |  |
| TSN | 0000100493 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| TSO | 0000050104 | 2010-01-04 | 2017-07-31 | 1 | ok | 1907 | 1976 | 0.965 | 2010-01-04 | 2017-07-31 |  |
| TSS | 0000721683 | 2010-01-04 | 2019-09-17 | 1 | ok | 2443 | 2532 | 0.965 | 2010-01-04 | 2019-09-17 |  |
| TT | 0001466258 | 2020-03-02 | 2025-12-31 | 1 | ok | 1468 | 1523 | 0.964 | 2020-03-02 | 2025-12-31 |  |
| TTD | 0001671933 | 2025-07-18 | 2025-12-31 | 1 | ok | 116 | 119 | 0.975 | 2025-07-18 | 2025-12-31 |  |
| TTWO | 0000946581 | 2018-03-19 | 2025-12-31 | 1 | ok | 1959 | 2033 | 0.964 | 2018-03-19 | 2025-12-31 |  |
| TWC | 0001377013 | 2010-01-04 | 2016-05-17 | 1 | ok | 1604 | 1662 | 0.965 | 2010-01-04 | 2016-05-17 |  |
| TWTR | 0001418091 | 2018-06-07 | 2022-10-27 | 1 | ok | 1107 | 1146 | 0.966 | 2018-06-07 | 2022-10-27 |  |
| TWX | 0001105705 | 2010-01-04 | 2018-06-14 | 1 | ok | 2127 | 2204 | 0.965 | 2010-01-04 | 2018-06-14 |  |
| TXN | 0000097476 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| TXT | 0000217346 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| TYC | 0001608109 | 2010-08-27 | 2016-09-02 | 1 | ok | 1516 | 1571 | 0.965 | 2010-08-27 | 2016-09-02 |  |
| TYL | 0000860731 | 2020-06-22 | 2025-12-31 | 1 | ok | 1390 | 1443 | 0.963 | 2020-06-22 | 2025-12-31 |  |
| UA | 0001336917 | 2014-05-01 | 2022-06-17 | 1 | ok | 2049 | 2122 | 0.966 | 2014-05-01 | 2022-06-17 |  |
| UAA | 0001336917 | 2016-12-07 | 2022-06-17 | 1 | ok | 1392 | 1443 | 0.965 | 2016-12-07 | 2022-06-17 |  |
| UAL | 0000100517 | 2015-09-03 | 2025-12-31 | 1 | ok | 2597 | 2695 | 0.964 | 2015-09-03 | 2025-12-31 |  |
| UBER | 0001543151 | 2023-12-18 | 2025-12-31 | 1 | ok | 511 | 533 | 0.959 | 2023-12-18 | 2025-12-31 |  |
| UDR | 0000074208 | 2016-03-07 | 2025-12-31 | 1 | ok | 2471 | 2563 | 0.964 | 2016-03-07 | 2025-12-31 |  |
| UHS | 0000352915 | 2014-09-22 | 2025-12-31 | 1 | ok | 2837 | 2943 | 0.964 | 2014-09-22 | 2025-12-31 |  |
| ULTA | 0001403568 | 2016-04-18 | 2025-12-31 | 1 | ok | 2442 | 2533 | 0.964 | 2016-04-18 | 2025-12-31 |  |
| UNH | 0000731766 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| UNM | 0000005513 | 2010-01-04 | 2021-09-17 | 1 | ok | 2948 | 3055 | 0.965 | 2010-01-04 | 2021-09-17 |  |
| UNP | 0000100885 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| UPS | 0001090727 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| URBN | 0000912615 | 2010-02-08 | 2017-03-17 | 1 | ok | 1790 | 1855 | 0.965 | 2010-02-08 | 2017-03-17 |  |
| URI | 0001067701 | 2014-09-22 | 2025-12-31 | 1 | ok | 2837 | 2943 | 0.964 | 2014-09-22 | 2025-12-31 |  |
| USB | 0000036104 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| UTX | 0000101829 | 2010-01-04 | 2020-04-02 | 1 | ok | 2580 | 2674 | 0.965 | 2010-01-04 | 2020-04-02 |  |
| V | 0001403161 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| VAR | 0000203527 | 2010-01-04 | 2021-04-14 | 1 | ok | 2839 | 2943 | 0.965 | 2010-01-04 | 2021-04-14 |  |
| VFC | 0000103379 | 2010-01-04 | 2024-04-02 | 1 | ok | 3585 | 3717 | 0.964 | 2010-01-04 | 2024-04-02 |  |
| VIA | 0001339947 | 2010-01-04 | 2011-11-30 | 1 | ok | 483 | 498 | 0.970 | 2010-01-04 | 2011-11-30 |  |
| VIAB | 0001339947 | 2011-12-01 | 2019-12-04 | 1 | ok | 2015 | 2090 | 0.964 | 2011-12-01 | 2019-12-04 |  |
| VIAC | 0002041610 | 2019-12-05 | 2022-02-16 | 1 | ok | 555 | 575 | 0.965 | 2019-12-05 | 2022-02-16 |  |
| VICI | 0001705696 | 2022-06-08 | 2025-12-31 | 1 | ok | 895 | 931 | 0.961 | 2022-06-08 | 2025-12-31 |  |
| VLO | 0001035002 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| VLTO | 0001967680 | 2023-10-02 | 2025-12-31 | 1 | ok | 564 | 588 | 0.959 | 2023-10-03 | 2025-12-31 |  |
| VMC | 0001396009 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| VNO | 0000899689 | 2010-01-04 | 2023-01-04 | 1 | ok | 3274 | 3393 | 0.965 | 2010-01-04 | 2023-01-04 |  |
| VNT | 0001786842 | 2020-10-09 | 2021-03-19 | 1 | ok | 110 | 116 | 0.948 | 2020-10-12 | 2021-03-19 |  |
| VRSK | 0001442145 | 2015-10-08 | 2025-12-31 | 1 | ok | 2573 | 2670 | 0.964 | 2015-10-08 | 2025-12-31 |  |
| VRSN | 0001014473 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| VRTX | 0000875320 | 2013-09-23 | 2025-12-31 | 1 | ok | 3087 | 3203 | 0.964 | 2013-09-23 | 2025-12-31 |  |
| VSM | 0001660690 | 2016-10-03 | 2016-10-03 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| VST | 0001692819 | 2024-05-08 | 2025-12-31 | 1 | ok | 414 | 431 | 0.961 | 2024-05-08 | 2025-12-31 |  |
| VTR | 0000740260 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| VTRS | 0001792044 | 2020-11-17 | 2025-12-31 | 1 | ok | 1285 | 1337 | 0.961 | 2020-11-18 | 2025-12-31 |  |
| VZ | 0000732712 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| WAB | 0000943452 | 2019-02-27 | 2025-12-31 | 1 | ok | 1722 | 1786 | 0.964 | 2019-02-27 | 2025-12-31 |  |
| WAG | 0001618921 | 2010-01-04 | 2014-12-30 | 1 | ok | 1257 | 1302 | 0.965 | 2010-01-04 | 2014-12-30 |  |
| WAT | 0001000697 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| WBA | 0001618921 | 2014-12-31 | 2025-08-27 | 1 | ok | 2680 | 2781 | 0.964 | 2014-12-31 | 2025-08-27 |  |
| WBD | 0001437107 | 2022-04-11 | 2025-12-31 | 1 | ok | 934 | 973 | 0.960 | 2022-04-12 | 2025-12-31 |  |
| WCG | 0001279363 | 2018-09-17 | 2020-01-23 | 1 | ok | 340 | 354 | 0.960 | 2018-09-17 | 2020-01-23 |  |
| WDAY | 0001327811 | 2024-12-23 | 2025-12-31 | 1 | ok | 256 | 268 | 0.955 | 2024-12-23 | 2025-12-31 |  |
| WDC | 0000106040 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| WEC | 0000783325 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| WELL | 0000766704 | 2018-02-28 | 2025-12-31 | 1 | ok | 1972 | 2046 | 0.964 | 2018-02-28 | 2025-12-31 |  |
| WFC | 0000072971 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| WFM | 0000865436 | 2011-05-06 | 2017-08-25 | 1 | ok | 1588 | 1646 | 0.965 | 2011-05-06 | 2017-08-25 |  |
| WFMI | 0000865436 | 2010-01-04 | 2011-05-05 | 1 | ok | 338 | 349 | 0.968 | 2010-01-04 | 2011-05-05 |  |
| WFR | 0000945436 | 2010-01-04 | 2011-12-16 | 1 | ok | 495 | 510 | 0.971 | 2010-01-04 | 2011-12-16 |  |
| WHR | 0000106640 | 2010-01-04 | 2024-03-15 | 1 | ok | 3574 | 3705 | 0.965 | 2010-01-04 | 2024-03-15 |  |
| WIN | 0001282266 | 2010-01-04 | 2015-04-06 | 1 | ok | 1322 | 1371 | 0.964 | 2010-01-04 | 2015-04-06 |  |
| WLP | 0001156039 | 2010-01-04 | 2014-12-02 | 1 | ok | 1238 | 1282 | 0.966 | 2010-01-04 | 2014-12-02 |  |
| WLTW | 0001140536 | 2016-01-05 | 2022-01-07 | 1 | ok | 1515 | 1569 | 0.966 | 2016-01-05 | 2022-01-07 |  |
| WM | 0000823768 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| WMB | 0000107263 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| WMT | 0000104169 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| WPI | 0001578845 | 2010-01-04 | 2013-01-23 | 1 | ok | 769 | 798 | 0.964 | 2010-01-04 | 2013-01-23 |  |
| WPO | 0000104889 | 2010-01-04 | 2013-11-27 | 1 | ok | 984 | 1018 | 0.967 | 2010-01-04 | 2013-11-27 |  |
| WPX | 0001518832 | 2012-01-03 | 2014-03-21 | 1 | ok | 556 | 579 | 0.960 | 2012-01-04 | 2014-03-21 |  |
| WRB | 0000011544 | 2019-12-05 | 2025-12-31 | 1 | ok | 1526 | 1585 | 0.963 | 2019-12-05 | 2025-12-31 |  |
| WRK | 0001732845 | 2015-07-02 | 2024-07-05 | 1 | ok | 2267 | 2352 | 0.964 | 2015-07-02 | 2024-07-05 |  |
| WSM | 0000719955 | 2025-03-24 | 2025-12-31 | 1 | ok | 196 | 203 | 0.966 | 2025-03-24 | 2025-12-31 |  |
| WST | 0000105770 | 2020-05-22 | 2025-12-31 | 1 | ok | 1410 | 1464 | 0.963 | 2020-05-22 | 2025-12-31 |  |
| WTW | 0001140536 | 2022-01-10 | 2025-12-31 | 1 | ok | 998 | 1038 | 0.961 | 2022-01-10 | 2025-12-31 |  |
| WU | 0001365135 | 2010-01-04 | 2021-12-17 | 1 | ok | 3012 | 3120 | 0.965 | 2010-01-04 | 2021-12-17 |  |
| WY | 0000106535 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| WYN | 0001361658 | 2010-01-04 | 2018-05-30 | 1 | ok | 2116 | 2193 | 0.965 | 2010-01-04 | 2018-05-30 |  |
| WYNN | 0001174922 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| X | 0001163302 | 2010-01-04 | 2014-07-01 | 1 | ok | 1131 | 1172 | 0.965 | 2010-01-04 | 2014-07-01 |  |
| XEC | 0001168054 | 2014-06-23 | 2020-03-02 | 1 | ok | 1433 | 1486 | 0.964 | 2014-06-23 | 2020-03-02 |  |
| XEL | 0000072903 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| XL | 0000875159 | 2010-01-04 | 2018-09-11 | 1 | ok | 2188 | 2267 | 0.965 | 2010-01-04 | 2018-09-11 |  |
| XLNX | 0000743988 | 2010-01-04 | 2022-02-11 | 1 | ok | 3050 | 3160 | 0.965 | 2010-01-04 | 2022-02-11 |  |
| XOM | 0000034088 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| XRAY | 0000818479 | 2010-01-04 | 2024-04-02 | 1 | ok | 3585 | 3717 | 0.964 | 2010-01-04 | 2024-04-02 |  |
| XRX | 0001770450 | 2010-01-04 | 2021-03-19 | 1 | ok | 2822 | 2925 | 0.965 | 2010-01-04 | 2021-03-19 |  |
| XTO | 0000868809 | 2010-01-04 | 2010-06-25 | 1 | ok | 121 | 125 | 0.968 | 2010-01-04 | 2010-06-25 |  |
| XYL | 0001524472 | 2011-11-01 | 2025-12-31 | 1 | ok | 3561 | 3697 | 0.963 | 2011-11-02 | 2025-12-31 |  |
| XYZ | 0001512673 | 2025-07-23 | 2025-12-31 | 1 | ok | 113 | 116 | 0.974 | 2025-07-23 | 2025-12-31 |  |
| YHOO | 0001011006 | 2010-01-04 | 2017-06-16 | 1 | ok | 1877 | 1945 | 0.965 | 2010-01-04 | 2017-06-16 |  |
| YUM | 0001041061 | 2010-01-04 | 2025-12-31 | 1 | ok | 4024 | 4173 | 0.964 | 2010-01-04 | 2025-12-31 |  |
| YUMC | 0001673358 | 2016-11-01 | 2016-11-01 | 1 | failed | 0 | 1 |  |  |  | no rows in CRSP store for membership window |
| ZBH | 0001136869 | 2015-06-29 | 2025-12-31 | 1 | ok | 2644 | 2743 | 0.964 | 2015-06-29 | 2025-12-31 |  |
| ZBRA | 0000877212 | 2019-12-23 | 2025-12-31 | 1 | ok | 1514 | 1573 | 0.962 | 2019-12-23 | 2025-12-31 |  |
| ZION | 0000109380 | 2010-01-04 | 2024-03-15 | 1 | ok | 3574 | 3705 | 0.965 | 2010-01-04 | 2024-03-15 |  |
| ZMH | 0001136869 | 2010-01-04 | 2015-06-26 | 1 | ok | 1380 | 1430 | 0.965 | 2010-01-04 | 2015-06-26 |  |
| ZTS | 0001555280 | 2013-06-24 | 2025-12-31 | 1 | ok | 3151 | 3268 | 0.964 | 2013-06-24 | 2025-12-31 |  |

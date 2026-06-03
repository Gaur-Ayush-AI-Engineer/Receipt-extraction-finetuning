# Failure Analysis

**Total examples:** 197  
**Correct (all fields):** 119 (60.4%)  
**Failed (≥1 field wrong):** 78 (39.6%)

## Failures by Field

| Field | Failures | Error Rate |
|:------|--------:|-----------:|
| company | 25 | 12.7% |
| date | 5 | 2.5% |
| address | 61 | 31.0% |
| total | 6 | 3.0% |

## Failures by Receipt Source

| Source | Failed | Total | Failure Rate |
|:-------|-------:|------:|-------------:|
| sroie | 30 | 81 | 37.0% |
| synthetic | 44 | 106 | 41.5% |
| unknown | 4 | 10 | 40.0% |

## Error Type Breakdown

| Field | Missing | Partial | Wrong Value | Hallucinated |
|:------|--------:|--------:|------------:|-------------:|
| company | 0 | 6 | 19 | 0 |
| date | 0 | 0 | 5 | 0 |
| address | 0 | 21 | 40 | 0 |
| total | 0 | 2 | 4 | 0 |

## Co-occurrence: How Many Fields Fail Together

| Fields Failed | Examples |
|:-------------:|---------:|
| 1 | 61 |
| 2 | 15 |
| 3 | 2 |

## Address Failure Samples (first 10)

**Source:** synthetic  
**GT:** `Near Flyover | Odhav Rd
Ahmedabad - 382415`  
**Pred:** `Near Flyover | Odhav Rd, Ahmedabad - 382415`  
**OCR snippet:** PETROL PUMP STATION 9 Gujarat Fuel Near Flyover | Odhav Rd Ahmedabad - 382415 GSTIN: 24AAABC1234Z1Z Date: 15/10/2023  Fu

**Source:** synthetic  
**GT:** `G-45, New Market, IndO re, Madhya PRADESH 452001`  
**Pred:** `G-45, New Market, Indore, Madhya Pradesh 452001`  
**OCR snippet:** MediCare PHaRmaCY G-45, New Market, IndO re Madhya PRADESH 452001   GSTIN: 23AAAPM1234Z1ZP  ----------------------------

**Source:** sroie  
**GT:** `NO.17,18 & 41,JALAN BESAR,39100 BRINCHANG, CAMERON HIGHLANDS, PAHANG`  
**Pred:** `NO.17,18 & 41,JALAN BESAR, 39100 BRINCHANG, CAMERON HIGHLANDS, PAHANG`  
**OCR snippet:** CASH SALE SINNATHAMBY HOLDINGS SDN. BHD. (249336-T) NO.17,18 & 41,JALAN BESAR, 39100 BRINCHANG, CAMERON HIGHLANDS, PAHAN

**Source:** sroie  
**GT:** `NO. 1 JALAN EURO 1 OFF JALAN BATU TIGA SUNGAI BULOH SEKSYEN U3 SHAH ALAM, 40150`  
**Pred:** `PASARAYA BORONG SUPER SEVEN NO. 1 JALAN EURO 1 OFF JALAN BATU TIGA SUNGAI BULOH SEKSYEN U3 SHAH ALAM, 40150`  
**OCR snippet:** SUPER SEVEN CASH & CARRY SDN BHD (590150-A) PASARAYA BORONG SUPER SEVEN NO. 1 JALAN EURO 1 OFF JALAN BATU TIGA SUNGAI BU

**Source:** sroie  
**GT:** `NO.42, PASAR BORONG SELANGOR. JALAN PUTRA PERMAI, 43300 SERI KEMBANGAN. SELANGOR.`  
**Pred:** `NO.42, PASAR BORONG SELANGOR, JALAN PUTRA PERMAI, 43300 SERI KEMBANGAN, SELANGOR.`  
**OCR snippet:** SATU KAMPUNG ENTERPRISE SDN BHD (1127686-K) NO.42, PASAR BORONG SELANGOR, JALAN PUTRA PERMAI, 43300 SERI KEMBANGAN, SELA

**Source:** synthetic  
**GT:** `Near Sadar Bazar, Gurgaon - 122001`  
**Pred:** `Near Sadar Bazar, Gurga0n - 122001`  
**OCR snippet:** GroCery SuPerMarket   GSTIN: 06ABCDE1234FZ1   Near Sadar Bazar,   Gurga0n - 122001   --------------------------------   

**Source:** sroie  
**GT:** `NO. 1 JALAN EURO 1 OFF JALAN BATU TIGA SUNGAI BULOH SEKSYEN U3 SHAH ALAM,40150`  
**Pred:** `PASARAYA BORONG SUPER SEVEN NO. 1 JALAN EURO 1 OFF JALAN BATU TIGA SONGGAI BULOH SEKSYEN U3 SHAH ALAM,40150`  
**OCR snippet:** SUPER.SEVEN CASH & CARRY SDN BHD (590150-A) PASARAYA BORONG SUPER SEVEN NO. 1 JALAN EURO 1 OFF JALAN BATU TIGA SUNGAI BU

**Source:** synthetic  
**GT:** `No. 45, 2nd Avenue, 9th dSt, [Chennai] 600001`  
**Pred:** `No. 45, 2nd Avenue, 9th dSt, Chennai`  
**OCR snippet:** GroCery SuPermkt    No. 45, 2nd Avenue,     9th dSt, [Chennai]       600001     GSTIN: 33ABCDE1234F1Z5    --------------

**Source:** sroie  
**GT:** `1-1, CENTRAL PARK , JALAN PJU5/13, DATARAN SUNWAY, 47810 KOTA DAMANSARA, SELANGOR`  
**Pred:** `1-1, CENTRAL PARK , JALAN PJU5/13, DATARAN SUNWAY, 47810 KOTA DAMANSARA , SELANGOR`  
**OCR snippet:** CHA FOR TEA 1-1, CENTRAL PARK , JALAN PJU5/13, DATARAN SUNWAY, 47810 KOTA DAMANSARA , SELANGOR GST ID NO:001698439168 TA

**Source:** sroie  
**GT:** `NO: 28, JALAN ASTANA 1C, BANDAR BUKIT RAJA, 41050 KLANG SELANGOR D.E, MALAYSIA`  
**Pred:** `NO: 28, JALAN ASTANA 1C, BANDAR BUKIT RAJA, 41050 KLANG SELANGOR D. E, MALAYSIA`  
**OCR snippet:** LIGHTROOM GALLERY SDN BHD NO: 28, JALAN ASTANA 1C, BANDAR BUKIT RAJA, 41050 KLANG SELANGOR D. E, MALAYSIA ROC NO. : (107

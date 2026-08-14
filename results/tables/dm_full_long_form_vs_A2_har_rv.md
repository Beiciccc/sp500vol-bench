# DM: full long_form test vs A2_har_rv

=== Diebold-Mariano vs A2_har_rv (dataset=full, disclosure=long_form, split=test) ===
Positive DM = challenger WORSE than baseline. p<0.05 = significant.

--- B1_bow_ridge vs A2_har_rv ---
  h= 5: DM=+5.120 p=0.0000  ** p<0.01  -> B1_bow_ridge WORSE   MAE(C/B)=0.1332/0.1071
  h=10: DM=+4.377 p=0.0000  ** p<0.01  -> B1_bow_ridge WORSE   MAE(C/B)=0.1176/0.0927
  h=20: DM=+4.296 p=0.0000  ** p<0.01  -> B1_bow_ridge WORSE   MAE(C/B)=0.1021/0.0778

--- B2_tfidf_ridge vs A2_har_rv ---
  h= 5: DM=+13.016 p=0.0000  ** p<0.01  -> B2_tfidf_ridge WORSE   MAE(C/B)=0.1177/0.1071
  h=10: DM=+13.320 p=0.0000  ** p<0.01  -> B2_tfidf_ridge WORSE   MAE(C/B)=0.1061/0.0927
  h=20: DM=+12.114 p=0.0000  ** p<0.01  -> B2_tfidf_ridge WORSE   MAE(C/B)=0.0934/0.0778

--- B3_lm_linear vs A2_har_rv ---
  h= 5: DM=+15.517 p=0.0000  ** p<0.01  -> B3_lm_linear WORSE   MAE(C/B)=0.1202/0.1071
  h=10: DM=+15.902 p=0.0000  ** p<0.01  -> B3_lm_linear WORSE   MAE(C/B)=0.1084/0.0927
  h=20: DM=+13.853 p=0.0000  ** p<0.01  -> B3_lm_linear WORSE   MAE(C/B)=0.0939/0.0778

--- B4_lm_features vs A2_har_rv ---
  h= 5: DM=+17.107 p=0.0000  ** p<0.01  -> B4_lm_features WORSE   MAE(C/B)=0.1227/0.1071
  h=10: DM=+16.421 p=0.0000  ** p<0.01  -> B4_lm_features WORSE   MAE(C/B)=0.1097/0.0927
  h=20: DM=+14.559 p=0.0000  ** p<0.01  -> B4_lm_features WORSE   MAE(C/B)=0.0952/0.0778

--- C1_bert_s1 vs A2_har_rv ---
  h= 5: DM=+22.987 p=0.0000  ** p<0.01  -> C1_bert_s1 WORSE   MAE(C/B)=0.1369/0.1071
  h=10: DM=+22.651 p=0.0000  ** p<0.01  -> C1_bert_s1 WORSE   MAE(C/B)=0.1322/0.0927
  h=20: DM=+19.653 p=0.0000  ** p<0.01  -> C1_bert_s1 WORSE   MAE(C/B)=0.1176/0.0778

--- C2_finbert_s1 vs A2_har_rv ---
  h= 5: DM=+17.382 p=0.0000  ** p<0.01  -> C2_finbert_s1 WORSE   MAE(C/B)=0.1236/0.1071
  h=10: DM=+14.360 p=0.0000  ** p<0.01  -> C2_finbert_s1 WORSE   MAE(C/B)=0.1072/0.0927
  h=20: DM=+18.449 p=0.0000  ** p<0.01  -> C2_finbert_s1 WORSE   MAE(C/B)=0.1122/0.0778

--- C2_finbert_s2 vs A2_har_rv ---
  h= 5: DM=+15.383 p=0.0000  ** p<0.01  -> C2_finbert_s2 WORSE   MAE(C/B)=0.1284/0.1071
  h=10: DM=+18.941 p=0.0000  ** p<0.01  -> C2_finbert_s2 WORSE   MAE(C/B)=0.1193/0.0927
  h=20: DM=+14.047 p=0.0000  ** p<0.01  -> C2_finbert_s2 WORSE   MAE(C/B)=0.0983/0.0778

--- C2_finbert_s3 vs A2_har_rv ---
  h= 5: DM=+18.806 p=0.0000  ** p<0.01  -> C2_finbert_s3 WORSE   MAE(C/B)=0.1317/0.1071
  h=10: DM=+13.154 p=0.0000  ** p<0.01  -> C2_finbert_s3 WORSE   MAE(C/B)=0.1082/0.0927
  h=20: DM=+15.662 p=0.0000  ** p<0.01  -> C2_finbert_s3 WORSE   MAE(C/B)=0.1064/0.0778

--- C2_finbert_s4 vs A2_har_rv ---
  h= 5: DM=+20.117 p=0.0000  ** p<0.01  -> C2_finbert_s4 WORSE   MAE(C/B)=0.1325/0.1071
  h=10: DM=+18.820 p=0.0000  ** p<0.01  -> C2_finbert_s4 WORSE   MAE(C/B)=0.1190/0.0927
  h=20: DM=+14.721 p=0.0000  ** p<0.01  -> C2_finbert_s4 WORSE   MAE(C/B)=0.1025/0.0778

--- C3_roberta_s1 vs A2_har_rv ---
  h= 5: DM=+16.582 p=0.0000  ** p<0.01  -> C3_roberta_s1 WORSE   MAE(C/B)=0.1217/0.1071
  h=10: DM=+16.947 p=0.0000  ** p<0.01  -> C3_roberta_s1 WORSE   MAE(C/B)=0.1117/0.0927
  h=20: DM=+14.499 p=0.0000  ** p<0.01  -> C3_roberta_s1 WORSE   MAE(C/B)=0.1011/0.0778

--- C4_longformer vs A2_har_rv ---
  h= 5: DM=+12.222 p=0.0000  ** p<0.01  -> C4_longformer WORSE   MAE(C/B)=0.1169/0.1071
  h=10: DM=+12.683 p=0.0000  ** p<0.01  -> C4_longformer WORSE   MAE(C/B)=0.1044/0.0927
  h=20: DM=+13.988 p=0.0000  ** p<0.01  -> C4_longformer WORSE   MAE(C/B)=0.0962/0.0778

--- D1_concat_mlp vs A2_har_rv ---
  h= 5: DM=+3.307 p=0.0009  ** p<0.01  -> D1_concat_mlp WORSE   MAE(C/B)=0.1079/0.1071
  h=10: DM=+5.513 p=0.0000  ** p<0.01  -> D1_concat_mlp WORSE   MAE(C/B)=0.0942/0.0927
  h=20: DM=+13.220 p=0.0000  ** p<0.01  -> D1_concat_mlp WORSE   MAE(C/B)=0.0878/0.0778

--- D2_gated_fusion vs A2_har_rv ---
  h= 5: DM=-0.789 p=0.4302    ns      -> D2_gated_fusion BETTER   MAE(C/B)=0.1075/0.1071
  h=10: DM=+15.359 p=0.0000  ** p<0.01  -> D2_gated_fusion WORSE   MAE(C/B)=0.1008/0.0927
  h=20: DM=+14.170 p=0.0000  ** p<0.01  -> D2_gated_fusion WORSE   MAE(C/B)=0.0893/0.0778


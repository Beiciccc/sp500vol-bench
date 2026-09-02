# DM: full long_form test vs D1_concat_mlp

=== Diebold-Mariano vs D1_concat_mlp (dataset=full, disclosure=long_form, split=test) ===
Positive DM = challenger WORSE than baseline. p<0.05 = significant.

--- A2_har_rv vs D1_concat_mlp ---
  h= 5: DM=-3.307 p=0.0009  ** p<0.01  -> A2_har_rv BETTER   MAE(C/B)=0.1071/0.1079
  h=10: DM=-5.513 p=0.0000  ** p<0.01  -> A2_har_rv BETTER   MAE(C/B)=0.0927/0.0942
  h=20: DM=-13.220 p=0.0000  ** p<0.01  -> A2_har_rv BETTER   MAE(C/B)=0.0778/0.0878

--- B1_bow_ridge vs D1_concat_mlp ---
  h= 5: DM=+4.984 p=0.0000  ** p<0.01  -> B1_bow_ridge WORSE   MAE(C/B)=0.1332/0.1079
  h=10: DM=+4.057 p=0.0000  ** p<0.01  -> B1_bow_ridge WORSE   MAE(C/B)=0.1176/0.0942
  h=20: DM=+2.900 p=0.0037  ** p<0.01  -> B1_bow_ridge WORSE   MAE(C/B)=0.1021/0.0878

--- B2_tfidf_ridge vs D1_concat_mlp ---
  h= 5: DM=+14.492 p=0.0000  ** p<0.01  -> B2_tfidf_ridge WORSE   MAE(C/B)=0.1177/0.1079
  h=10: DM=+14.730 p=0.0000  ** p<0.01  -> B2_tfidf_ridge WORSE   MAE(C/B)=0.1061/0.0942
  h=20: DM=+7.326 p=0.0000  ** p<0.01  -> B2_tfidf_ridge WORSE   MAE(C/B)=0.0934/0.0878

--- B3_lm_linear vs D1_concat_mlp ---
  h= 5: DM=+16.992 p=0.0000  ** p<0.01  -> B3_lm_linear WORSE   MAE(C/B)=0.1202/0.1079
  h=10: DM=+17.047 p=0.0000  ** p<0.01  -> B3_lm_linear WORSE   MAE(C/B)=0.1084/0.0942
  h=20: DM=+10.386 p=0.0000  ** p<0.01  -> B3_lm_linear WORSE   MAE(C/B)=0.0939/0.0878

--- B4_lm_features vs D1_concat_mlp ---
  h= 5: DM=+18.704 p=0.0000  ** p<0.01  -> B4_lm_features WORSE   MAE(C/B)=0.1227/0.1079
  h=10: DM=+17.885 p=0.0000  ** p<0.01  -> B4_lm_features WORSE   MAE(C/B)=0.1097/0.0942
  h=20: DM=+11.717 p=0.0000  ** p<0.01  -> B4_lm_features WORSE   MAE(C/B)=0.0952/0.0878

--- C1_bert_s1 vs D1_concat_mlp ---
  h= 5: DM=+25.459 p=0.0000  ** p<0.01  -> C1_bert_s1 WORSE   MAE(C/B)=0.1369/0.1079
  h=10: DM=+25.103 p=0.0000  ** p<0.01  -> C1_bert_s1 WORSE   MAE(C/B)=0.1322/0.0942
  h=20: DM=+21.437 p=0.0000  ** p<0.01  -> C1_bert_s1 WORSE   MAE(C/B)=0.1176/0.0878

--- C2_finbert_s1 vs D1_concat_mlp ---
  h= 5: DM=+19.159 p=0.0000  ** p<0.01  -> C2_finbert_s1 WORSE   MAE(C/B)=0.1236/0.1079
  h=10: DM=+15.667 p=0.0000  ** p<0.01  -> C2_finbert_s1 WORSE   MAE(C/B)=0.1072/0.0942
  h=20: DM=+19.946 p=0.0000  ** p<0.01  -> C2_finbert_s1 WORSE   MAE(C/B)=0.1122/0.0878

--- C2_finbert_s2 vs D1_concat_mlp ---
  h= 5: DM=+15.621 p=0.0000  ** p<0.01  -> C2_finbert_s2 WORSE   MAE(C/B)=0.1284/0.1079
  h=10: DM=+20.892 p=0.0000  ** p<0.01  -> C2_finbert_s2 WORSE   MAE(C/B)=0.1193/0.0942
  h=20: DM=+9.770 p=0.0000  ** p<0.01  -> C2_finbert_s2 WORSE   MAE(C/B)=0.0983/0.0878

--- C2_finbert_s3 vs D1_concat_mlp ---
  h= 5: DM=+21.573 p=0.0000  ** p<0.01  -> C2_finbert_s3 WORSE   MAE(C/B)=0.1317/0.1079
  h=10: DM=+13.931 p=0.0000  ** p<0.01  -> C2_finbert_s3 WORSE   MAE(C/B)=0.1082/0.0942
  h=20: DM=+14.756 p=0.0000  ** p<0.01  -> C2_finbert_s3 WORSE   MAE(C/B)=0.1064/0.0878

--- C2_finbert_s4 vs D1_concat_mlp ---
  h= 5: DM=+21.449 p=0.0000  ** p<0.01  -> C2_finbert_s4 WORSE   MAE(C/B)=0.1325/0.1079
  h=10: DM=+20.076 p=0.0000  ** p<0.01  -> C2_finbert_s4 WORSE   MAE(C/B)=0.1190/0.0942
  h=20: DM=+12.478 p=0.0000  ** p<0.01  -> C2_finbert_s4 WORSE   MAE(C/B)=0.1025/0.0878

--- C3_roberta_s1 vs D1_concat_mlp ---
  h= 5: DM=+18.721 p=0.0000  ** p<0.01  -> C3_roberta_s1 WORSE   MAE(C/B)=0.1217/0.1079
  h=10: DM=+18.821 p=0.0000  ** p<0.01  -> C3_roberta_s1 WORSE   MAE(C/B)=0.1117/0.0942
  h=20: DM=+13.122 p=0.0000  ** p<0.01  -> C3_roberta_s1 WORSE   MAE(C/B)=0.1011/0.0878

--- C4_longformer vs D1_concat_mlp ---
  h= 5: DM=+13.685 p=0.0000  ** p<0.01  -> C4_longformer WORSE   MAE(C/B)=0.1169/0.1079
  h=10: DM=+13.636 p=0.0000  ** p<0.01  -> C4_longformer WORSE   MAE(C/B)=0.1044/0.0942
  h=20: DM=+11.853 p=0.0000  ** p<0.01  -> C4_longformer WORSE   MAE(C/B)=0.0962/0.0878

--- D2_gated_fusion vs D1_concat_mlp ---
  h= 5: DM=-4.580 p=0.0000  ** p<0.01  -> D2_gated_fusion BETTER   MAE(C/B)=0.1075/0.1079
  h=10: DM=+13.456 p=0.0000  ** p<0.01  -> D2_gated_fusion WORSE   MAE(C/B)=0.1008/0.0942
  h=20: DM=+2.620 p=0.0088  ** p<0.01  -> D2_gated_fusion WORSE   MAE(C/B)=0.0893/0.0878


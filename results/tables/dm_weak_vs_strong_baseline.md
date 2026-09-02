# L2 Manufactured-signal demonstration (long-form test).
A text model's apparent skill flips with baseline strength: it can beat a naive persistence baseline while losing to HAR-RV.

| text model | vs baseline | h5 | h10 | h20 |
|---|---|---|---|---|
| C4_longformer | weak: naive RV persistence | -0.75 ns | -0.74 ns | -1.44 ns |
| C4_longformer | strong: HAR-RV | +12.22* | +12.68* | +13.99* |
| C2_finbert_s1 | weak: naive RV persistence | +1.49 ns | -0.12 ns | +1.45 ns |
| C2_finbert_s1 | strong: HAR-RV | +17.39* | +14.36* | +18.45* |

Positive DM = text worse than that baseline. Text loses to HAR everywhere; against the naive persistence baseline the gap shrinks, illustrating how a weak baseline flatters text.
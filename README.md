# SAPLMA


<img src="pics/mean_median_std.png" alt="Mean_Median_Std_image" width="300" />




"""
1. Embeddings animals
2. Get results like in the paper - throw it to latex
3. Generate new sentence till it gets to ".". Then use Saplma with threshold 0.5 to determine if we can use it.
4. Use the reward model as well with midian threshold

"""



"""
31/08/2025:

1. Completion - 5 Examples  - with and without guard
2. Instruct - 5 Examples  - with and without guard


3. Saplma without user - Check "Tell me a a true" and "Tell me a false" datasets
results:

| test(capitals) \ train(All but cap.) |          empty         |      tell_me_true      |     tell_me_false      |
-------------------------------------------------------------------------------------------------------------------
|          empty                       | acc:     92.59         | acc:     92.94         | acc:     63.79         | 
|                                      | acc@0.5: 88.61         | acc@0.5: 92.39         | acc@0.5: 69.62         |
|                                      | AUC:     97.99         | AUC:     97.97         | AUC:     95.54         |
-------------------------------------------------------------------------------------------------------------------
|      tell_me_true                    | acc:     91.70         | acc:     92.80         | acc:     67.49         | 
|                                      | acc@0.5: 77.91         | acc@0.5: 91.77         | acc@0.5: 71.12         |
|                                      | AUC:     97.61         | AUC:     97.60         | AUC:     94.05         |
-------------------------------------------------------------------------------------------------------------------
|     tell_me_false                    | acc:     77.57         | acc:     92.11         | acc:     92.73         | 
|                                      | acc@0.5: 52.33         | acc@0.5: 82.10         | acc@0.5: 92.25         |
|                                      | AUC:     94.33         | AUC:     96.72         | AUC:     96.86         |
-------------------------------------------------------------------------------------------------------------------

"""



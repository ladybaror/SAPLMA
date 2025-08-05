import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

# classify_sentences_new.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from paper.config import (
    MODEL_NAME,
    LAYERS_TO_USE,
    DATASET_FOLDER,
    REPEAT_EACH,
    CHECK_UNCOMMON,
    CHECK_GENERATED,
    KEEP_PROBABILITIES
)


from paper.config import LIST_OF_DATASETS

def correct_str(str_arr):
    val_to_ret = str_arr.replace("[array(", "").replace("dtype=float32)]", "").replace("\n", "").replace(" ", "").replace("],", "]").replace("[", "").replace("]", "")
    return val_to_ret

for layer_num_from_end in LAYERS_TO_USE:
    # if CHECK_UNCOMMON:
    #     dataset_names = ["uncommon", "capitals", "inventions", "elements", "animals", "facts", "companies"]
    # elif CHECK_GENERATED:
    #     dataset_names = ["generated", "capitals", "inventions", "elements", "animals", "facts", "companies"]
    # else:
    #     # dataset_names = ["capitals", "inventions", "elements", "animals", "facts", "companies", "olympics", "movies"]
    #     dataset_names = ["capitals", "inventions", "elements", "animals", "facts", "companies"]

    dataset_names = LIST_OF_DATASETS
    datasets = [pd.read_csv(DATASET_FOLDER + f"/embeddings_with_labels_{name}{MODEL_NAME}_{abs(layer_num_from_end)}_rmv_period.csv") for name in dataset_names]
    results = []
    dataset_loop_length = 1 if CHECK_UNCOMMON or CHECK_GENERATED else len(dataset_names)

    for ds in range(dataset_loop_length):
        test_df = datasets[0] if CHECK_UNCOMMON or CHECK_GENERATED else datasets[ds]
        train_df = pd.concat(datasets[:ds] + datasets[ds + 1:], ignore_index=True) if not (CHECK_UNCOMMON or CHECK_GENERATED) else pd.concat(datasets[1:], ignore_index=True)
        all_probs = np.zeros((len(test_df), 1))

        for i in range(REPEAT_EACH):
            train_embeddings = np.array([np.fromstring(correct_str(e), sep=',') for e in train_df['embeddings'].tolist()])
            test_embeddings = np.array([np.fromstring(correct_str(e), sep=',') for e in test_df['embeddings'].tolist()])
            train_labels = np.array(train_df['label'])
            test_labels = np.array(test_df['label'])

            model = Sequential()
            model.add(Dense(256, activation='relu', input_dim=train_embeddings.shape[1]))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(train_embeddings, train_labels, epochs=5, batch_size=32, validation_data=(test_embeddings, test_labels))
            loss, accuracy = model.evaluate(test_embeddings, test_labels)
            test_pred_prob = model.predict(test_embeddings)

            if KEEP_PROBABILITIES:
                all_probs += test_pred_prob

            fpr, tpr, _ = roc_curve(test_labels, test_pred_prob)
            roc_auc = auc(fpr, tpr)
            print("AUC of the classifier:", roc_auc)

            X_val, X_test, y_val, y_test = train_test_split(test_embeddings, test_labels, test_size=0.7, random_state=42)
            y_val_pred_prob = model.predict(X_val)
            thresholds_val = roc_curve(y_val, y_val_pred_prob)[2]
            optimal_threshold = thresholds_val[np.argmax([accuracy_score(y_val, y_val_pred_prob > thr) for thr in thresholds_val])]
            y_test_pred_prob = model.predict(X_test)
            y_test_pred = (y_test_pred_prob > optimal_threshold).astype(int)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            print("Optimal threshold:", optimal_threshold)
            print("Test accuracy:", test_accuracy)

            results.append((dataset_names[ds], i, accuracy, roc_auc, optimal_threshold, test_accuracy))

        if KEEP_PROBABILITIES:
            all_probs /= REPEAT_EACH
            print("-- Averaged probabilities --")
            print(all_probs)

        accs = [t[2] for t in results[ds * REPEAT_EACH:(ds + 1) * REPEAT_EACH]]
        aucs = [t[3] for t in results[ds * REPEAT_EACH:(ds + 1) * REPEAT_EACH]]
        opt_thresh = [t[4] for t in results[ds * REPEAT_EACH:(ds + 1) * REPEAT_EACH]]
        acc_thr_test = [t[5] for t in results[ds * REPEAT_EACH:(ds + 1) * REPEAT_EACH]]

        print(f"dataset: {dataset_names[ds]} layer: {layer_num_from_end} Avg_acc: {np.mean(accs):.4f} Avg_AUC: {np.mean(aucs):.4f} Avg_threshold: {np.mean(opt_thresh):.4f} Avg_thrs_acc: {np.mean(acc_thr_test):.4f}")








# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import confusion_matrix, accuracy_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# #import ast

# def correct_str(str_arr):
#     val_to_ret = str_arr.replace("[array(", "").replace("dtype=float32)]", "").replace("\n","").replace(" ","").replace("],","]").replace("[","").replace("]","")
#     return val_to_ret

# check_uncommon = False #True
# check_generated = False
# repeat_each = 10 # 3 # 10 #14
# layer_num_list = [-12] #["BERT"] #,[-1, -4, -8, -12, -16]
# model_to_use = "1.3b" #"6.7b" "2.7b" "1.3b" "350m"

# keep_probabilities = check_uncommon
# check_single_first = check_uncommon or check_generated
# overall_res = []
# for layer_num_from_end in layer_num_list:

#     if check_uncommon:
#         dataset_names = ["uncommon", "capitals", "inventions", "elements", "animals", "facts", "companies"]
#     elif check_generated:
#         dataset_names = ["generated", "capitals", "inventions", "elements", "animals", "facts", "companies"]
#     else:
#         dataset_names = ["capitals", "inventions", "elements", "animals", "facts", "companies", "olympics", "movies"]
#     datasets = []
#     for dataset_name in dataset_names:
#         if layer_num_from_end == "BERT":
#             datasets.append(pd.read_csv("datasets\\LLMTF\\bert_" + dataset_name + ".csv"))
#         else:
#             datasets.append(pd.read_csv('datasets\\LLMTF\\embeddings_with_labels_'+dataset_name+str(model_to_use)+'_'+str(abs(layer_num_from_end))+'_rmv_period.csv'))
#         #datasets.append(pd.read_csv('resources\\embeddings_with_labels_'+dataset_name+'6.7b_5fromend_rmv_period.csv'))
#         #datasets.append(pd.read_csv('resources\\embeddings_with_labels_' + dataset_name + '6.7b_1_rmv_period.csv'))

#     results = []
#     dataset_loop_length = 1 if check_single_first else len(dataset_names)
#     for ds in range(dataset_loop_length):
#         if check_single_first:
#             test_df = datasets[0]
#             dfs_to_concatenate = datasets[1:]  # excluding "capitals" [2:]
#         else:
#             test_df = datasets[ds]
#             dfs_to_concatenate = datasets[:ds] + datasets[ds + 1:]
#         train_df = pd.concat(dfs_to_concatenate, ignore_index=True)
#         all_probs = np.zeros((len(test_df),1))
#         for i in range(repeat_each):

#             #test_df = pd.read_csv('resources\\embeddings_with_labels_grammar6.7brmv_period.csv')
#             #df5 = pd.read_csv('resources\\embeddings_with_labels_colors6.7brmv_period.csv')
#             #df = pd.concat([df1,df2,df5,df3,df6,df4], ignore_index=True)
#             #df = df[df['next_id'] == 4]  #only those in which the next token is supposed to be '.'

#             # test_df = pd.read_csv('resources\\gen_text_embed_label2.csv')
#             # train_df = pd.read_csv('resources\\gen_text_embed_label3.csv')

#             # Split the data into train and test sets
#             #train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

#             #np.fromstring(train_df['embeddings'].tolist()[0].replace("[array(", "").replace("dtype=float32)]", "").replace("\n","").replace(" ","").replace("],","]"), sep=',')
#             # Extract the embeddings and labels from the train and test sets
#             train_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in train_df['embeddings'].tolist()])
#             test_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in test_df['embeddings'].tolist()])
#             train_labels = np.array(train_df['label'])
#             test_labels = np.array(test_df['label'])


#             # train_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in train_df['embed'].tolist()])
#             # test_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in test_df['embed'].tolist()])
#             # train_labels = np.array(train_df['truth_label'])
#             # test_labels = np.array(test_df['truth_label'])


#             # Define the neural network model
#             model = Sequential()
#             model.add(Dense(256, activation='relu', input_dim=train_embeddings.shape[1])) #change input_dim to match the number of elements in train_embeddings...
#             model.add(Dense(128, activation='relu'))
#             #model.add(Dense(64, activation='relu'))
#             model.add(Dense(64, activation='relu'))
#             model.add(Dense(1, activation='sigmoid'))
#             model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Compile the model
#             #from tensorflow.keras.models import load_model
#             #model = load_model("resources\\classifier_all.h5")

#             # Train the model
#             model.fit(train_embeddings, train_labels, epochs=5, batch_size=32, validation_data=(test_embeddings, test_labels))
#             loss, accuracy = model.evaluate(test_embeddings, test_labels)

#             # Evaluate the model on the test data
#             test_pred_prob = model.predict(test_embeddings)
#             #test_pred = np.where(test_pred_prob > 0.8, 1, 0)

#             if keep_probabilities:
#                 all_probs += test_pred_prob

#             # Compute the confusion matrix and accuracy
#             #cm = confusion_matrix(test_labels, test_pred)
#             #print(cm)
#             #accuracy = accuracy_score(test_labels, test_pred)
#             #print(accuracy)

#             # Compute ROC curve and ROC area
#             fpr, tpr, _ = roc_curve(test_labels, test_pred_prob)  # Assuming binary classification
#             roc_auc = auc(fpr, tpr)
#             print("AUC of the classifier on the test set:", roc_auc)


#             # Find the optimal threshold
#             X_val, X_test, y_val, y_test = train_test_split(test_embeddings, test_labels, test_size=0.7, random_state=42)
#             # Evaluate the model on the test data
#             y_val_pred_prob = model.predict(X_val)
#             fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_val_pred_prob)  # Assuming binary classification
#             optimal_threshold = thresholds_val[np.argmax([accuracy_score(y_val, y_val_pred_prob > thr) for thr in thresholds_val])]

#             # Use the optimal threshold to predict labels on the test set
#             y_test_pred_prob = model.predict(X_test)
#             y_test_pred = (y_test_pred_prob > optimal_threshold).astype(int)

#             # Evaluate the classifier on the test set
#             test_accuracy = accuracy_score(y_test, y_test_pred)

#             print("Optimal threshold:", optimal_threshold)
#             print("Test set accuracy:", test_accuracy)

#             results.append((dataset_names[ds], i, accuracy, roc_auc, optimal_threshold, test_accuracy))
#             # model.save("resources\\classifier_all.h5")
#         all_probs = all_probs / repeat_each
#         print("----probs:----")
#         print(all_probs)
#         print("----end probs----")

#     print(results)
#     for ds in range(dataset_loop_length):
#         relevant_results_portion = results[repeat_each*ds:repeat_each*(ds+1)]
#         # Extract the second item from each tuple and put it in a list
#         acc_list = [t[2] for t in relevant_results_portion]
#         auc_list = [t[3] for t in relevant_results_portion]
#         opt_thresh_list = [t[4] for t in relevant_results_portion]
#         acc_thr_test_list = [t[5] for t in relevant_results_portion]
#         # Calculate the average of the numbers in the list
#         avg_acc = sum(acc_list) / len(acc_list)
#         avg_auc = sum(auc_list) / len(auc_list)
#         avg_thrsh = sum(opt_thresh_list) / len(opt_thresh_list)
#         avg_thr_test_acc = sum(acc_thr_test_list) / len(acc_thr_test_list)
#         text_res = "dataset: " + str(dataset_names[ds]) + " layer_num_from_end:" + str(layer_num_from_end) + " Avg_acc:" + str(avg_acc) + " Avg_AUC:" + str(avg_auc) + " Avg_threshold:" + str(avg_thrsh) + " Avg_thrs_acc:" + str(avg_thr_test_acc)
#         print(text_res)
#         overall_res.append(text_res)

# print(overall_res)
# for res in overall_res:
#     print(res)
# #results:
# #trained on all but cities:
# #[(0, 0.5478927493095398), (1, 0.5977011322975159), (2, 0.5632184147834778), (3, 0.5977011322975159), (4, 0.6130267977714539)]
# #0.5839080452919007
# #[(0, 0.5708812475204468), (1, 0.5862069129943848), (2, 0.5938697457313538), (3, 0.6321839094161987), (4, 0.6015325784683228)]
# #0.5969348788261414
# #trained on all including cities:
# # [(0, 0.6590038537979126), (1, 0.6360152959823608), (2, 0.6283524632453918), (3, 0.6513410210609436), (4, 0.6436781883239746), (5, 0.6091954112052917), (6, 0.6819923520088196)]
# # 0.6442255122320992
# # [(0, 0.6743295192718506), (1, 0.6245210766792297), (2, 0.6704980731010437), (3, 0.6091954112052917), (4, 0.6781609058380127), (5, 0.6360152959823608), (6, 0.6206896305084229)]
# # 0.6447728446551731
# # [(0, 0.7088122367858887), (1, 0.6896551847457886), (2, 0.6283524632453918), (3, 0.6321839094161987), (4, 0.6551724076271057), (5, 0.6551724076271057), (6, 0.6245210766792297)]
# # 0.6562670980181012

# #trained only on generated: 0.7622641444206237 [(0, 0.7169811129570007), (1, 0.7547169923782349), (2, 0.8301886916160583), (3, 0.7169811129570007), (4, 0.7924528121948242)]
# #trained only on generated: 0.7886792540550231 [(0, 0.7547169923782349), (1, 0.7547169923782349), (2, 0.849056601524353), (3, 0.8113207817077637), (4, 0.7735849022865295)]

# #layer -1, trained on all (but generated): 0.6387520432472229 [(0, 0.6283524632453918), (1, 0.6206896305084229), (2, 0.6475095748901367), (3, 0.6513410210609436), (4, 0.6475095748901367), (5, 0.6206896305084229), (6, 0.6551724076271057)]
# #layer -1, trained on all (but generated): 0.6174055763653347 [(0, 0.6436781883239746), (1, 0.6091954112052917), (2, 0.6053639650344849), (3, 0.6283524632453918), (4, 0.6283524632453918), (5, 0.5977011322975159), (6, 0.6091954112052917)]

# #layer -16 trained on all (but generated): 0.566502468926566 [(0, 0.5938697457313538), (1, 0.5632184147834778), (2, 0.540229856967926), (3, 0.5517241358757019), (4, 0.6015325784683228), (5, 0.5785440802574158), (6, 0.5363984704017639)]

# ########################new generated:
# #layer -16: 0.5580174922943115 [(0, 0.5510203838348389), (1, 0.518367350101471), (2, 0.5510203838348389), (3, 0.5510203838348389), (4, 0.5714285969734192), (5, 0.6000000238418579), (6, 0.563265323638916)]
# # 0.5626822199140277 [(0, 0.5387755036354065), (1, 0.5918367505073547), (2, 0.5714285969734192), (3, 0.5795918107032776), (4, 0.5306122303009033), (5, 0.5551020503044128), (6, 0.5714285969734192)]

# #layer -12: 0.5947521839823041 [(0, 0.6122449040412903), (1, 0.6040816307067871), (2, 0.5755102038383484), (3, 0.5510203838348389), (4, 0.6000000238418579), (5, 0.6122449040412903), (6, 0.6081632375717163)]
# # 0.6023323621068682 [(0, 0.5673469305038452), (1, 0.6204081773757935), (2, 0.5918367505073547), (3, 0.6204081773757935), (4, 0.6163265109062195), (5, 0.5959183573722839), (6, 0.6040816307067871)]

# #layer -8: 0.6034985440117973 [(0, 0.5714285969734192), (1, 0.5673469305038452), (2, 0.6204081773757935), (3, 0.5959183573722839), (4, 0.5959183573722839), (5, 0.6285714507102966), (6, 0.6448979377746582)]
# # 0.6005831020218986 [(0, 0.6326530575752258), (1, 0.6000000238418579), (2, 0.6000000238418579), (3, 0.6204081773757935), (4, 0.5836734771728516), (5, 0.5959183573722839), (6, 0.5714285969734192)]
# #(avg 0.6020)
# # [(0, 0.6448979377746582), (1, 0.6326530575752258), (2, 0.6081632375717163), (3, 0.6000000238418579), (4, 0.6000000238418579), (5, 0.6122449040412903), (6, 0.6448979377746582)]
# # 0.620408160345895
# #0.8 threshold
# # [(0, 0.6693877551020408), (1, 0.7061224489795919), (2, 0.6653061224489796), (3, 0.7061224489795919), (4, 0.6285714285714286), (5, 0.636734693877551), (6, 0.6693877551020408)]
# # 0.668804664723032

# #layer -5: 0.6373177681650434 [(0, 0.636734664440155), (1, 0.6897959113121033), (2, 0.6163265109062195), (3, 0.6326530575752258), (4, 0.6163265109062195), (5, 0.6612244844436646), (6, 0.6081632375717163)]
# #          0.630903788975307 [(0, 0.6612244844436646), (1, 0.5918367505073547), (2, 0.5836734771728516), (3, 0.6816326379776001), (4, 0.6326530575752258), (5, 0.6285714507102966), (6, 0.636734664440155)]
# # 0.62682215656553 [(0, 0.6122449040412903), (1, 0.6897959113121033), (2, 0.5877550840377808), (3, 0.6693877577781677), (4, 0.5877550840377808), (5, 0.6285714507102966), (6, 0.6122449040412903)]
# # 0.6349854213850838 [(0, 0.6000000238418579), (1, 0.6489796042442322), (2, 0.640816330909729), (3, 0.6081632375717163), (4, 0.6775510311126709), (5, 0.6612244844436646), (6, 0.6081632375717163)]
# # 0.6454810500144958 [(0, 0.6163265109062195), (1, 0.6122449040412903), (2, 0.6693877577781677), (3, 0.6571428775787354), (4, 0.6244897842407227), (5, 0.6571428775787354), (6, 0.6816326379776001)]
# # (avg 0.6348)

# #layer -1: 0.6209912470408848 [(0, 0.6163265109062195), (1, 0.6040816307067871), (2, 0.640816330909729), (3, 0.6081632375717163), (4, 0.6530612111091614), (5, 0.6204081773757935), (6, 0.6040816307067871)]
# # 0.630903788975307 [(0, 0.6000000238418579), (1, 0.6857143044471741), (2, 0.6081632375717163), (3, 0.6326530575752258), (4, 0.6122449040412903), (5, 0.6326530575752258), (6, 0.6448979377746582)]
# # 0.6233236193656921 [(0, 0.6204081773757935), (1, 0.6000000238418579), (2, 0.6122449040412903), (3, 0.6040816307067871), (4, 0.6489796042442322), (5, 0.640816330909729), (6, 0.636734664440155)]
# # 0.608746349811554 [(0, 0.6244897842407227), (1, 0.6163265109062195), (2, 0.6000000238418579), (3, 0.6244897842407227), (4, 0.6244897842407227), (5, 0.5755102038383484), (6, 0.5959183573722839)]
# # (avg 0.6210)
# # 0.8 0.6658892128279883 [(0, 0.7020408163265306), (1, 0.6693877551020408), (2, 0.6857142857142857), (3, 0.636734693877551), (4, 0.6530612244897959), (5, 0.6571428571428571), (6, 0.6571428571428571)]


# #layer -4: 0.6274052432605198 [(0, 0.6163265109062195), (1, 0.6530612111091614), (2, 0.6775510311126709), (3, 0.6122449040412903), (4, 0.6081632375717163), (5, 0.5959183573722839), (6, 0.6285714507102966)]
# # 0.6309037974902562 [(0, 0.6489796042442322), (1, 0.6653061509132385), (2, 0.6204081773757935), (3, 0.6326530575752258), (4, 0.5959183573722839), (5, 0.6000000238418579), (6, 0.6530612111091614)]
# #                   0.6041, 0.6286, 0.6776, 0.6163, 0.6245
# # with test_pred > 0.8: 0.685131195335277 [(0, 0.6938775510204082), (1, 0.689795918367347), (2, 0.710204081632653), (3, 0.6571428571428571), (4, 0.6653061224489796), (5, 0.6816326530612244), (6, 0.6979591836734694)]
# #0.7026239067055393 [(0, 0.6938775510204082), (1, 0.7183673469387755), (2, 0.710204081632653), (3, 0.710204081632653), (4, 0.6816326530612244), (5, 0.7061224489795919), (6, 0.6979591836734694)]

# ############ With AUC ############
# # layer -4:
# # [(0, 0.6530612111091614, 0.740562891823396), (1, 0.6204081773757935, 0.7670401493930905), (2, 0.6081632375717163, 0.7443644124316393), (3, 0.6244897842407227, 0.7592370281445912), (4, 0.5877550840377808, 0.7641723356009071), (5, 0.6448979377746582, 0.7550353474723223), (6, 0.6163265109062195, 0.7643057222889156)]
# # Avg_acc:0.6221574204308646 Avg_AUC:0.7563882695935517
# # [(0, 0.6693877577781677, 0.7575696945444845), (1, 0.6653061509132385, 0.7715086034413765), (2, 0.6081632375717163, 0.7513672135520875), (3, 0.6530612111091614, 0.7739095638255302), (4, 0.6571428775787354, 0.7614379084967321), (5, 0.6530612111091614, 0.78031212484994), (6, 0.6448979377746582, 0.7679738562091504)]
# # Avg_acc:0.6501457691192627 Avg_AUC:0.7662969949884715
#  #avge acc: 0.636153  avg auc: 0.76135
# ##### layer -1
# # [(0, 0.6122449040412903, 0.7520341469921302), (1, 0.6204081773757935, 0.7807122849139656), (2, 0.636734664440155, 0.7789115646258503), (3, 0.6244897842407227, 0.775110044017607), (4, 0.6326530575752258, 0.7591036414565826), (5, 0.6204081773757935, 0.7591703348005869), (6, 0.6081632375717163, 0.7669734560490863), (7, 0.6612244844436646, 0.7592370281445913), (8, 0.6244897842407227, 0.7463652127517674), (9, 0.6122449040412903, 0.7715086034413767), (10, 0.6122449040412903, 0.772375616913432), (11, 0.6285714507102966, 0.7619714552487663), (12, 0.6489796042442322, 0.7595038015206081), (13, 0.6571428775787354, 0.7637721755368814)]
# # ignore!!! Avg_acc:0.6285714294229235 Avg_AUC:0.7647678118866593
# # [(0, 0.5959183573722839, 0.7753768173936242), (1, 0.6326530575752258, 0.7460984393757503), (2, 0.6163265109062195, 0.7268240629585169), (3, 0.6530612111091614, 0.7671735360810991), (4, 0.6244897842407227, 0.760437508336668), (5, 0.6122449040412903, 0.7697078831532613), (6, 0.6040816307067871, 0.7535680939042284), (7, 0.5918367505073547, 0.7675070028011204), (8, 0.6081632375717163, 0.7453648125917033), (9, 0.6163265109062195, 0.7533013205282113), (10, 0.6040816307067871, 0.781045751633987), (11, 0.6489796042442322, 0.7549019607843137), (12, 0.640816330909729, 0.7594371081766039), (13, 0.6122449040412903, 0.7615712951847405)]
# # Avg_acc:0.6186588874885014 Avg_AUC:0.7587368280645592

# ######## layer -8
# # [(0, 0.6285714507102966, 0.729358410030679), (1, 0.6000000238418579, 0.7293584100306788), (2, 0.6244897842407227, 0.741830065359477), (3, 0.6693877577781677, 0.7592370281445912), (4, 0.6244897842407227, 0.7603041216486595), (5, 0.6612244844436646, 0.7572362278244631), (6, 0.6000000238418579, 0.7458316659997334), (7, 0.5877550840377808, 0.7377617713752168), (8, 0.6163265109062195, 0.7401627317593704), (9, 0.559183657169342, 0.7392290249433106), (10, 0.6040816307067871, 0.7434973989595838), (11, 0.5959183573722839, 0.7376283846872083), (12, 0.5959183573722839, 0.7380285447512338), (13, 0.6204081773757935, 0.7491663331999466)]
# # Avg_acc:0.6134110774312701 Avg_AUC:0.7434735799081537


# ########## layer -12
# # [(0, 0.5795918107032776, 0.719954648526077), (1, 0.6040816307067871, 0.7135520875016673), (2, 0.6244897842407227, 0.7093504068293985), (3, 0.5877550840377808, 0.7200880352140856), (4, 0.636734664440155, 0.7216886754701881), (5, 0.5918367505073547, 0.7097505668934242), (6, 0.6000000238418579, 0.7212885154061626), (7, 0.6204081773757935, 0.7187541683340004), (8, 0.5918367505073547, 0.7231559290382821), (9, 0.5714285969734192, 0.7202881152460983), (10, 0.5918367505073547, 0.7008136587968521), (11, 0.5959183573722839, 0.7193544084300387), (12, 0.5959183573722839, 0.7208883553421369), (13, 0.6489796042442322, 0.735360810991063)]
# # Avg_acc:0.6029154530593327 Avg_AUC:0.7181634558585338

# ########## layer -16
# # [(0, 0.5346938967704773, 0.6647325596905429), (1, 0.5877550840377808, 0.6566626650660264), (2, 0.5469387769699097, 0.6592637054821929), (3, 0.6000000238418579, 0.6658663465386155), (4, 0.5714285969734192, 0.6571962118180605), (5, 0.5918367505073547, 0.6441243163932241), (6, 0.5346938967704773, 0.6487928504735228), (7, 0.5755102038383484, 0.662531679338402), (8, 0.563265323638916, 0.6649326397225556), (9, 0.5510203838348389, 0.6710684273709484), (10, 0.5265306234359741, 0.6840736294517807), (11, 0.559183657169342, 0.6481926103774844), (12, 0.5306122303009033, 0.6710684273709483), (13, 0.518367350101471, 0.6559957316259838)]
# # Avg_acc:0.5565597712993622 Avg_AUC:0.661035842908592


# # layer_num_from_end:-1 Avg_acc:0.6261224448680878 Avg_AUC:0.7673202614379085 Avg_threshold:0.8687718331813812 Avg_thrs_acc:0.7052325581395349
# # layer_num_from_end:-4 Avg_acc:0.6253061115741729 Avg_AUC:0.761844737895158 Avg_threshold:0.8837663412094117 Avg_thrs_acc:0.7133720930232558
# # layer_num_from_end:-8 Avg_acc:0.616326528787613 Avg_AUC:0.7499666533279978 Avg_threshold:0.8800795197486877 Avg_thrs_acc:0.6988372093023256
# # layer_num_from_end:-12 Avg_acc:0.5967346906661988 Avg_AUC:0.7169601173802854 Avg_threshold:0.9062659919261933 Avg_thrs_acc:0.6587209302325583
# # layer_num_from_end:-16 Avg_acc:0.570204085111618 Avg_AUC:0.6645324796585301 Avg_threshold:0.8123523414134979 Avg_thrs_acc:0.65

# #[(0, 0.517241358757019, 0.6246135552913199, 0.9454254, 0.5956284153005464), (1, 0.5747126340866089, 0.6234244946492271, 0.92485607, 0.546448087431694), (2, 0.50957852602005, 0.6167657550535077, 0.9948394, 0.5683060109289617), (3, 0.517241358757019, 0.5862068965517241, 0.9913551, 0.5573770491803278), (4, 0.49808427691459656, 0.6186087990487515, 0.7887548, 0.6229508196721312), (5, 0.532567024230957, 0.6184898929845422, 0.95375496, 0.5573770491803278), (6, 0.49808427691459656, 0.5919143876337694, 0.96815073, 0.5792349726775956), (7, 0.5747126340866089, 0.6090368608799049, 0.92266995, 0.5519125683060109), (8, 0.4789271950721741, 0.5821046373365042, 0.95878595, 0.5409836065573771), (9, 0.48275861144065857, 0.5953626634958382, 0.954416, 0.5846994535519126)]
# #layer_num_from_end:BERT Avg_acc:0.5183907896280289 Avg_AUC:0.6066527942925088 Avg_threshold:0.9403008341789245 Avg_thrs_acc:0.5704918032786885



# # % cities: layer_num_from_end:BERT Avg_acc:0.5695016185442606 Avg_AUC:0.6114008892802776 Avg_threshold:0.6035094857215881 Avg_thrs_acc:0.554031994776363
# # % inventions: layer_num_from_end:BERT Avg_acc:0.5525114138921102 Avg_AUC:0.5773159385113269 Avg_threshold:0.3826653063297272 Avg_thrs_acc:0.5646036916395223
# # % elements: layer_num_from_end:BERT Avg_acc:0.5261648694674174 Avg_AUC:0.5750368057964312 Avg_threshold:0.5912142097949982 Avg_thrs_acc:0.5780849974398361
# # % animals: layer_num_from_end:BERT Avg_acc:0.5238095124562582 Avg_AUC:0.4869155223817922 Avg_threshold:0.6154211362202963 Avg_thrs_acc:0.5136921624173749
# # % facts: layer_num_from_end:BERT Avg_acc:0.5399673779805502 Avg_AUC:0.5150429697756772 Avg_threshold:0.4674496253331502 Avg_thrs_acc:0.5271317829457365
# # % companies: layer_num_from_end:BERT Avg_acc:0.5475000143051147 Avg_AUC:0.5656277777777778 Avg_threshold:0.4158678948879242 Avg_thrs_acc:0.5198412698412698
# # %Average of all for thrs_acc BERT: 0.5428

# # dataset: capitals layer_num_from_end:-1 Avg_acc:0.7501143018404642 Avg_AUC:0.8749193231233572 Avg_threshold:0.4894796311855316 Avg_thrs_acc:0.8126020241593209
# # dataset: inventions layer_num_from_end:-1 Avg_acc:0.5353881120681763 Avg_AUC:0.6893238756835175 Avg_threshold:0.008471367104599873 Avg_thrs_acc:0.6384364820846905
# # dataset: elements layer_num_from_end:-1 Avg_acc:0.5698924660682678 Avg_AUC:0.6122172120090955 Avg_threshold:0.7350498040517172 Avg_thrs_acc:0.5796210957501281
# # dataset: animals layer_num_from_end:-1 Avg_acc:0.5727513035138448 Avg_AUC:0.6637521520954061 Avg_threshold:0.07138730088869731 Avg_thrs_acc:0.6067044381491974
# # dataset: facts layer_num_from_end:-1 Avg_acc:0.6574225028355917 Avg_AUC:0.8069074535351599 Avg_threshold:0.8830281893412272 Avg_thrs_acc:0.724031007751938
# # dataset: companies layer_num_from_end:-1 Avg_acc:0.6866666674613953 Avg_AUC:0.8456666666666668 Avg_threshold:0.8988116979598999 Avg_thrs_acc:0.7547619047619047
# # dataset: capitals layer_num_from_end:-4 Avg_acc:0.786236842473348 Avg_AUC:0.8820257877481539 Avg_threshold:0.5279973049958547 Avg_thrs_acc:0.7992164544564152
# # dataset: inventions layer_num_from_end:-4 Avg_acc:0.6099695563316345 Avg_AUC:0.7347062967302757 Avg_threshold:0.036926506708065666 Avg_thrs_acc:0.6574375678610206
# # dataset: elements layer_num_from_end:-4 Avg_acc:0.5885304808616638 Avg_AUC:0.6410143754576637 Avg_threshold:0.6939984957377116 Avg_thrs_acc:0.5924219150025601
# # dataset: animals layer_num_from_end:-4 Avg_acc:0.5968915422757467 Avg_AUC:0.6702779877383053 Avg_threshold:0.19016421834627786 Avg_thrs_acc:0.6260623229461757
# # dataset: facts layer_num_from_end:-4 Avg_acc:0.6634040276209513 Avg_AUC:0.8145398224436354 Avg_threshold:0.8624147772789001 Avg_thrs_acc:0.7333333333333333
# # dataset: companies layer_num_from_end:-4 Avg_acc:0.7247222264607748 Avg_AUC:0.8804138888888889 Avg_threshold:0.7447126706441244 Avg_thrs_acc:0.8039682539682539
# # dataset: capitals layer_num_from_end:-8 Avg_acc:0.8235025207201639 Avg_AUC:0.9187692832631783 Avg_threshold:0.6955945690472921 Avg_thrs_acc:0.8338230492980738
# # dataset: inventions layer_num_from_end:-8 Avg_acc:0.6902587413787842 Avg_AUC:0.7964042099096084 Avg_threshold:0.19760993619759878 Avg_thrs_acc:0.6959826275787188
# # dataset: elements layer_num_from_end:-8 Avg_acc:0.6211469570795695 Avg_AUC:0.6704050564612479 Avg_threshold:0.4687371551990509 Avg_thrs_acc:0.6216077828981055
# # dataset: animals layer_num_from_end:-8 Avg_acc:0.6074735323588053 Avg_AUC:0.7029071659528009 Avg_threshold:0.20368099709351858 Avg_thrs_acc:0.6288951841359773
# # dataset: facts layer_num_from_end:-8 Avg_acc:0.6721044182777405 Avg_AUC:0.818840348300015 Avg_threshold:0.8364983002344767 Avg_thrs_acc:0.7286821705426356
# # dataset: companies layer_num_from_end:-8 Avg_acc:0.7827777663866679 Avg_AUC:0.8990560185185185 Avg_threshold:0.747175415356954 Avg_thrs_acc:0.8416666666666667
# # dataset: capitals layer_num_from_end:-12 Avg_acc:0.8214448889096578 Avg_AUC:0.9284530173622283 Avg_threshold:0.8497183322906494 Avg_thrs_acc:0.8374142997061704
# # dataset: inventions layer_num_from_end:-12 Avg_acc:0.7351597944895426 Avg_AUC:0.8339558224528513 Avg_threshold:0.5325792531172434 Avg_thrs_acc:0.7247557003257329
# # dataset: elements layer_num_from_end:-12 Avg_acc:0.6172043085098267 Avg_AUC:0.6851736231548927 Avg_threshold:0.42461466292540234 Avg_thrs_acc:0.6303123399897593
# # dataset: animals layer_num_from_end:-12 Avg_acc:0.6044973532358805 Avg_AUC:0.7020161459645587 Avg_threshold:0.07232549041509628 Avg_thrs_acc:0.6298394711992447
# # dataset: facts layer_num_from_end:-12 Avg_acc:0.6797172427177429 Avg_AUC:0.7771018287879755 Avg_threshold:0.8029263615608215 Avg_thrs_acc:0.6837209302325581
# # dataset: companies layer_num_from_end:-12 Avg_acc:0.8086111148198446 Avg_AUC:0.9015921296296296 Avg_threshold:0.6920703848203024 Avg_thrs_acc:0.8444444444444444
# # dataset: capitals layer_num_from_end:-16 Avg_acc:0.7439414660135905 Avg_AUC:0.8701862796936379 Avg_threshold:0.6388656298319498 Avg_thrs_acc:0.7939928174991838
# # dataset: inventions layer_num_from_end:-16 Avg_acc:0.6457381844520569 Avg_AUC:0.7545265595357661 Avg_threshold:0.36088673770427704 Avg_thrs_acc:0.6612377850162866
# # dataset: elements layer_num_from_end:-16 Avg_acc:0.5609319011370341 Avg_AUC:0.626265849616526 Avg_threshold:0.3760940432548523 Avg_thrs_acc:0.5944700460829493
# # dataset: animals layer_num_from_end:-16 Avg_acc:0.5780423283576965 Avg_AUC:0.615168598303519 Avg_threshold:0.4999268054962158 Avg_thrs_acc:0.5854579792256847
# # dataset: facts layer_num_from_end:-16 Avg_acc:0.6340402364730835 Avg_AUC:0.6893491019281401 Avg_threshold:0.7114833990732828 Avg_thrs_acc:0.6348837209302326
# # dataset: companies layer_num_from_end:-16 Avg_acc:0.7522222201029459 Avg_AUC:0.857938425925926 Avg_threshold:0.6227341592311859 Avg_thrs_acc:0.7746031746031746

# #dataset: uncommon layer_num_from_end:-4 Avg_acc:0.8666666746139526 Avg_AUC:0.9472222222222225

# ############### after correcting scientific facts:

# # dataset: capitals layer_num_from_end:-12 Avg_acc:0.8079560935497284 Avg_AUC:0.9211938860569658 Avg_threshold:0.8858885288238525 Avg_thrs_acc:0.8354554358472086
# # dataset: inventions layer_num_from_end:-12 Avg_acc:0.7077625572681427 Avg_AUC:0.8079406595246066 Avg_threshold:0.3017324350774288 Avg_thrs_acc:0.7037459283387622
# # dataset: elements layer_num_from_end:-12 Avg_acc:0.6216129064559937 Avg_AUC:0.6932412995722049 Avg_threshold:0.4005685932934284 Avg_thrs_acc:0.6250384024577573
# # dataset: animals layer_num_from_end:-12 Avg_acc:0.6188492000102996 Avg_AUC:0.7085541855631141 Avg_threshold:0.2503149498254061 Avg_thrs_acc:0.638385269121813
# # dataset: facts layer_num_from_end:-12 Avg_acc:0.6789559602737427 Avg_AUC:0.7874912179855655 Avg_threshold:0.7623784601688385 Avg_thrs_acc:0.6927906976744186
# # dataset: companies layer_num_from_end:-12 Avg_acc:0.8182499945163727 Avg_AUC:0.9083679166666668 Avg_threshold:0.7695685088634491 Avg_thrs_acc:0.8439285714285714
# # 0.7088977853457133
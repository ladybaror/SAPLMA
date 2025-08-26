from guarded_infer.saplma_api import load_best_bundle, embed_texts_last_token, predict_embeddings
from paper.config import BASE_MODEL_PATH   # ensure you embed with the same base model used in training

BUNDLE = "pretrained_saplma/completion/saplma_checkpoints_LLAMA7/BEST_layer12__heldout_data/elements"

model, thr, meta, path = load_best_bundle(bundle=BUNDLE)  # defaults to paper.config
layer_from_end = int(meta["layer_from_end"])              # use the trained layer

text = ["Venus is the hottest planet in the Solar System."]

X = embed_texts_last_token(
    text,
    model_path=BASE_MODEL_PATH,
    layer_from_end=layer_from_end,
    device="cuda"     # or "cpu" / "auto"
)

res = predict_embeddings(model, thr, X)
print("Prob(True):", float(res["prob_true"][0]))
print("Decision (best):", res["pred_best"][0])   # 'True' or 'Lie'
print("Decision (0.5):",  res["pred_0p5"][0])

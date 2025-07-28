from transformers import AutoModelForSequenceClassification, AutoTokenizer
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)

question = "I just came out of from jail, any suggestion of my future?"
helpful = "It's great to hear that you have been released from jail."
# helpful = "It's great to hear that you"
bad = "Go back to jail you scum"

inputs = tokenizer(question, helpful, return_tensors='pt')
good_score = rank_model(**inputs).logits[0].cpu().detach()

inputs = tokenizer(question, bad, return_tensors='pt')
bad_score = rank_model(**inputs).logits[0].cpu().detach()

print("good_score: ", good_score)
print("bad_score: ", bad_score)
print(good_score > bad_score) # tensor([True])



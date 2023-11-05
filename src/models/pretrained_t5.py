from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class PretrainedT5:
    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained("s-nlp/t5-paranmt-detox")
        self.tokenizer = AutoTokenizer.from_pretrained("s-nlp/t5-paranmt-detox")

    def __call__(self, text):
        tokens = self.tokenizer(text, return_tensors="pt")
        out = self.model.generate(tokens["input_ids"])
        out = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return out

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class PretrainedT5:
    """
    The T5 model pretrained on ParaNMT dataset
    """
    def __init__(self):
        """
        Initialize the PretrainedT5 model.

        This constructor loads a pre-trained T5 model and tokenizer.
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained("s-nlp/t5-paranmt-detox")
        self.tokenizer = AutoTokenizer.from_pretrained("s-nlp/t5-paranmt-detox")

    def __call__(self, text: str) -> str:
        """
        Generate detoxified text using the T5 model.

        :param text: The input text to detoxify.

        :return: Detoxified text as a result of T5 model processing.
        """
        tokens = self.tokenizer(text, return_tensors="pt")
        out = self.model.generate(tokens["input_ids"],  max_new_tokens=100)
        out = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return out

import torch
from peft import get_peft_model, LoraConfig
from transformers import GPTNeoXForQuestionAnswering, AutoTokenizer


class generate_response:
    def __init__(self, path, config_path):
        model = GPTNeoXForQuestionAnswering.from_pretrained(path, device_map="auto")
        lora_config = LoraConfig.from_pretrained(config_path)
        self.device = "cuda:0"
        self.model = get_peft_model(model, lora_config).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    def __call__(self, query, context):
        inputs = self.tokenizer(query, context, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        if answer_start_index > answer_end_index:
            predict_answer_tokens = inputs.input_ids[
                0, answer_end_index : answer_start_index + 10]
        else:
            predict_answer_tokens = inputs.input_ids[
                0, answer_start_index : answer_end_index + 10]
        answer = self.tokenizer.decode(predict_answer_tokens)
        if "?" in answer:
            answer = answer.split('?')[1]
        return answer

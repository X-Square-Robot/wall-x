import torch
from PIL import Image
from transformers import AutoProcessor

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction


class VQAWrapper(object):
    def __init__(self, model_path: str):
        self.device = self._setup_device()
        self.processor = self._load_processor(model_path)
        self.model = self._load_model(model_path)

    def _setup_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _load_processor(self, model_path: str) -> AutoProcessor:
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def _load_model(self, model_path: str) -> Qwen2_5_VLMoEForAction:
        model = Qwen2_5_VLMoEForAction.from_pretrained(model_path)
        if self.device == "cuda":
            model = model.to(self.device, dtype=torch.bfloat16)
        else:
            model.to(self.device)
        model.eval()
        return model

    def generate(self, image: Image.Image, text: str, **kwargs) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generation_params = {
            "max_new_tokens": 1024,  # default value, can be overridden by kwargs
            "do_sample": False,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            **kwargs,
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_params)

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response


if __name__ == "__main__":
    MODEL_PATH_FOR_MODULE_TEST = "/path/to/model"
    wrapper = VQAWrapper(model_path=MODEL_PATH_FOR_MODULE_TEST)

    try:
        test_question = "To move the red block in the plate with same color, what should you do next? Think step by step."

        # Local Image
        img = Image.open("/path/to/wall-x/assets/cot_example_frame.png").convert("RGB")
        # Internet Image
        # import requests
        # test_image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        # img = Image.open(requests.get(test_image_url, stream=True).raw).convert("RGB")

        answer = wrapper.generate(img, test_question)

        print("model answer:", answer)
    except Exception as e:
        print(f"model answer fail: {e}")

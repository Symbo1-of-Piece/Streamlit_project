from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from config import MODEL_PATH, MAX_LENGTH, NUM_BEAMS

class ImageCaptioningModel:
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
        self.feature_extractor = ViTImageProcessor.from_pretrained(MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.gen_kwargs = {"max_length": MAX_LENGTH, "num_beams": NUM_BEAMS}

    def predict_step(self, image_paths):
        images = []
        for image_path in image_paths:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert(mode="RGB")

            images.append(image)

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

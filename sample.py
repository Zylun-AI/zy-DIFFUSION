import torch
from model import ImprovedUNet
from transformers import CLIPTokenizer, CLIPTextModel
import torchvision.utils as vutils

def sample(model_ckpt, prompt, out_file="sample.png"):
    device = torch.device("cuda")
    model = ImprovedUNet().to(device)
    model.load_state_dict(torch.load(model_ckpt))
    clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    text_embs = clip_text_model(tokens.input_ids.to(device))[0][:,0]
    img = torch.randn(1, 3, 256, 256, device=device)
    for t in range(50):
        img = model(img, text_embs)
    vutils.save_image(img, out_file, normalize=True)

if __name__ == "__main__":
    sample("model_ckpt.pt", "um astronauta surfando na lua")
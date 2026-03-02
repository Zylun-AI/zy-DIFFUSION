import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import csv
from model import ImprovedUNet
import torchvision.transforms as T
from transformers import CLIPTextModel, CLIPTokenizer

class TextImgDataset(Dataset):
    def __init__(self, datadir):
        self.data = []
        with open(f"{datadir}/metadata.txt", encoding="utf-8") as f:
            for line in f:
                fname, prompt = line.strip().split("\t")
                self.data.append((f"{datadir}/{fname}", prompt))
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def __getitem__(self, idx):
        img_file, prompt = self.data[idx]
        img = Image.open(img_file).convert("RGB").resize((256,256))
        img = self.transform(img)
        tokens = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        return img, tokens.input_ids.squeeze(0)

    def __len__(self):
        return len(self.data)

def train_step(batch, model, optimizer, clip_text_model, device):
    imgs, token_ids = batch
    imgs = imgs.to(device)
    token_ids = token_ids.to(device)
    text_embs = clip_text_model(token_ids)[0][:,0]
    noise = torch.randn_like(imgs)
    noisy = imgs + 0.1 * noise
    pred = model(noisy, text_embs)
    loss = torch.nn.functional.mse_loss(pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    datadir = "data/text2img"
    device = torch.device("cuda")
    dataset = TextImgDataset(datadir)
    loader = DataLoader(dataset, batch_size=24, shuffle=True, num_workers=4)
    model = ImprovedUNet().to(device)
    clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    for epoch in range(50):
        for batch in loader:
            loss = train_step(batch, model, optimizer, clip_text_model, device)
            print("Loss", loss)
        # opcional: salvar checkpoint aqui

if __name__ == "__main__":
    main()
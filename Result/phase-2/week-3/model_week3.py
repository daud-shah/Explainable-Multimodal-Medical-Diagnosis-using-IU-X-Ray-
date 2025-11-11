# FINAL CORRECT model_week3.py — 100% WORKING
import torch
import torch.nn as nn
from transformers import ViTModel, AutoModel, AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VisionEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
        self.hidden_dim = 768
    def forward(self, x):
        return self.vit(x).last_hidden_state[:, 0, :]

class TextEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.hidden_dim = 768
    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
    def tokenize(self, texts, max_length=512):
        return self.tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.3):
        super().__init__()
        # CRITICAL: These two lines must exist
        self.img_to_text_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.text_to_img_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, img_emb, text_emb):
        img_seq = img_emb.unsqueeze(1)
        text_seq = text_emb.unsqueeze(1)

        img_out, _ = self.img_to_text_attn(img_seq, text_seq, text_seq)
        img_out = self.ln1(img_out.squeeze(1) + img_emb)

        text_out, _ = self.text_to_img_attn(text_seq, img_seq, img_seq)
        text_out = self.ln2(text_out.squeeze(1) + text_emb)

        fused = torch.cat([img_out, text_out], dim=1)
        fused = self.ffn(fused)
        fused = self.ln3(fused + img_emb)

        return fused

class ClassificationHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=15, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class MultimodalFusionClassifier(nn.Module):
    def __init__(self, num_classes=15, freeze_encoders=True):
        super().__init__()
        self.vision_encoder = VisionEncoder(freeze_encoders)
        self.text_encoder = TextEncoder(freeze_encoders)
        self.fusion = CrossAttentionFusion()
        self.classifier = ClassificationHead(num_classes=num_classes)

    def forward(self, images, input_ids, attention_mask):
        img_emb = self.vision_encoder(images)
        txt_emb = self.text_encoder(input_ids, attention_mask)
        fused = self.fusion(img_emb, txt_emb)
        return self.classifier(fused)

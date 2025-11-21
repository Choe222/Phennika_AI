import pandas as pd
import torch
from torch.utils.data import DataLoader
from bestModel import MLPb
from dataProcess import MyData
from trainModel import evaluate
import os

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPb(freeze_encoder=False).to(device)
    model_path = "Url_to_your_trained_model.pt"
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded trained model successfully.")
    else:
        raise FileNotFoundError("File not found.")

    test_df = pd.read_csv("test_data.csv")
    test_data = MyData(test_df)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, pin_memory=True, drop_last=True)
    micro_f1, sentiment_score, overall_score = evaluate(model, test_loader, device)
    print(f"Micro-F1: {micro_f1:.4f} | Sentiment Score: {sentiment_score:.4f} | "f"Overall Score: {overall_score:.4f}")
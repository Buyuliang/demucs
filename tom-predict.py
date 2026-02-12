import argparse
import torch
import torchaudio
from demucs.tasnet import ConvTasNet


def load_model(model_path, device):
    print("Loading checkpoint...")
    ckpt = torch.load(model_path, map_location=device)

    if "best_state" not in ckpt:
        raise ValueError("Checkpoint does not contain 'best_state'")

    model = ConvTasNet(
        sources=["vocals", "accompaniment"]
    )

    model.load_state_dict(ckpt["best_state"], strict=True)
    model.to(device)
    model.eval()

    print("✅ Model loaded successfully")
    return model


def separate(model, wav_path, out_dir, device):
    print(f"Loading audio: {wav_path}")
    wav, sr = torchaudio.load(wav_path)

    # ✅ 不要转 mono！
    # 模型训练时是 2 channel，就必须保持 2 channel

    wav = wav.to(device)

    with torch.no_grad():
        out = model(wav.unsqueeze(0))  # [1, 2, C, T] or [1, 2, T]

    out = out.cpu()

    vocals = out[0, 0]
    accomp = out[0, 1]

    torchaudio.save(f"{out_dir}/vocals.wav", vocals, sr)
    torchaudio.save(f"{out_dir}/accompaniment.wav", accomp, sr)

    print("✅ Separation complete")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .th checkpoint")
    parser.add_argument("--input", required=True, help="Input wav file")
    parser.add_argument("--out", default="outputs", help="Output folder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = torch.device(args.device)

    model = load_model(args.model, device)
    separate(model, args.input, args.out, device)


if __name__ == "__main__":
    main()

# python tom-predict.py \
#   --model models/mydemucs.th \
#   --input mydata/train/track_0005/mixture.wav \
#   --out output

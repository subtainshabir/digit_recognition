from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.label_embed = nn.Embedding(10, 10)

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28 + 10, 400),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 10, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid(),
        )

    def decode(self, z, y):
        y_embed = self.label_embed(y)
        z = torch.cat([z, y_embed], dim=1)
        return self.decoder(z)

model = CVAE().to(device)
model.load_state_dict(torch.load("cvae_mnist.pth", map_location=device))
model.eval()

# === Generate and Save Images ===
def generate_images(digit):
    os.makedirs('static/generated', exist_ok=True)
    digit_tensor = torch.tensor([digit] * 5).to(device)
    z = torch.randn(5, model.latent_dim).to(device)
    with torch.no_grad():
        generated = model.decode(z, digit_tensor).cpu()
    images = generated.view(-1, 28, 28)

    paths = []
    for i, img in enumerate(images):
        path = f'static/generated/digit_{i}.png'
        plt.imsave(path, img.numpy(), cmap='gray')
        paths.append(path)
    return paths

# === Flask Routes ===
@app.route("/", methods=["GET", "POST"])
def index():
    image_paths = []
    digit = None
    if request.method == "POST":
        digit = int(request.form["digit"])
        if 0 <= digit <= 9:
            image_paths = generate_images(digit)
    return render_template("index.html", image_paths=image_paths, digit=digit)

if __name__ == "__main__":
    app.run(debug=True)

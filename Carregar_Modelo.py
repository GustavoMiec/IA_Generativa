import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os

# --- Modelo VAE ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(400, 20)
        self.fc2 = nn.Linear(20, 400)
        self.fc3 = nn.Linear(400, 28*28)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h2 = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h2))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28*28))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- Função de perda ---
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- Treinamento ---
def treinar_vae(model, device, epochs=10):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader.dataset):.4f}")
    
    torch.save(model.state_dict(), "vae_model.pt")
    print("✅ Modelo salvo como 'vae_model.pt'")

# --- Gerar imagens ---
def gerar_imagens(model, device):
    model.load_state_dict(torch.load("vae_model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, 20).to(device)
        samples = model.decode(z).cpu().view(-1, 1, 28, 28)

        plt.figure(figsize=(8, 8))
        for i in range(64):
            plt.subplot(8, 8, i + 1)
            plt.imshow(samples[i][0], cmap='gray')
            plt.axis('off')
        plt.show()

# --- Menu ---
def menu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)

    while True:
        print("\n===== MENU VAE =====")
        print("[1] Treinar VAE")
        print("[2] Gerar imagens")
        print("[0] Sair")
        escolha = input("Escolha: ")

        if escolha == "1":
            ep = input("Quantas épocas? (ex: 10): ")
            treinar_vae(model, device, epochs=int(ep))
        elif escolha == "2":
            if not os.path.exists("vae_model.pt"):
                print("⚠️ Treine o modelo primeiro!")
            else:
                gerar_imagens(model, device)
        elif escolha == "0":
            break
        else:
            print("Opção inválida.")

# --- Executar ---
if __name__ == "__main__":
    menu()

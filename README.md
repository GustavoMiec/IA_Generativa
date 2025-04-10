
# 🧠 Variational Autoencoder (VAE) - MNIST Generator

Este projeto implementa um **Autoencoder Variacional (VAE)** para gerar imagens de dígitos escritos à mão usando o dataset **MNIST**. Tudo está integrado em um script com **menu interativo** via terminal para facilitar o uso!

## 📌 Funcionalidades

- [x] Treinar um VAE com MNIST
- [x] Salvar modelo treinado
- [x] Gerar imagens novas usando vetores aleatórios no espaço latente
- [x] Interface de menu simples via terminal

## 🖼️ Exemplo de Imagens Geradas

Após o treinamento, o modelo pode gerar imagens como estas:

![image](https://github.com/user-attachments/assets/98b41944-6464-4d4c-b043-4fff64e20472)


## ▶️ Como usar

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/seu-repo-vae.git
cd seu-repo-vae
```

### 2. Instalar dependências

Crie um ambiente virtual (opcional) e instale os pacotes necessários:

```bash
pip install torch torchvision matplotlib
```

### 3. Rodar o script

```bash
python vae_menu.py
```

### 4. Escolher uma opção:

```
===== MENU VAE =====
[1] Treinar VAE
[2] Gerar imagens
[0] Sair
Escolha:
```

- `1` → Treina o modelo e salva como `vae_model.pt`
- `2` → Gera 64 imagens com base no modelo salvo
- `0` → Sai do programa

## 📁 Estrutura do Projeto

```
.
├── vae_menu.py              # Script principal com menu
├── vae_model.pt             # (gerado após o treinamento)
├── vae_output_example.png   # Exemplo de imagens geradas
└── README.md                # Este arquivo
```

## 📚 Referências

- Variational Autoencoder (Kingma & Welling, 2013)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- MNIST dataset - http://yann.lecun.com/exdb/mnist/

## ✨ Créditos

Feito por [Seu Nome](https://github.com/seu-usuario) ✨

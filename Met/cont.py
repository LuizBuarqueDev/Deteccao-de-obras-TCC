import os

# Caminho do seu dataset
dataset_path = "dataset/train"

# Extensões consideradas como imagem
extensoes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

contagem = {}

# Percorre todas as subpastas
for root, dirs, files in os.walk(dataset_path):
    # Nome da classe = nome da pasta atual
    classe = os.path.basename(root)

    # Conta apenas arquivos de imagem
    qtd = sum(1 for f in files if os.path.splitext(f)[1].lower() in extensoes)

    if qtd > 0:
        contagem[classe] = qtd

# Ordena das classes com mais imagens para menos
ordenado = sorted(contagem.items(), key=lambda x: x[1], reverse=True)

# Exibe resultado
print("\n=== Contagem de imagens por classe ===")
for classe, qtd in ordenado:
    print(f"{classe}: {qtd} imagens")

print("\nTotal de classes:", len(ordenado))

import os
from pathlib import Path
from my_transformer import my_transformer

def main():
    transformer=my_transformer(
            embedding_dim = 32,
            heads = 4,
            batch_size = 8,
            learning_rate = 0.001
        )
    script_dir = Path(__file__).resolve().parent
    data_folder = script_dir / "../data"
    for epoch in range(20):
        for f in os.listdir(data_folder):
            if f.endswith(".txt"):
                path = os.path.join(data_folder, f)
                loss=transformer.train(path)
                print(f"Epoch: {epoch} | loss: {loss}")
    # Llamada directa con un string
    resultado = transformer.predict("en un lugar", max_length=10)

    # Imprimir el resultado unido como una frase legible
    print(" ".join(resultado)) 
if __name__ == "__main__":
    main()

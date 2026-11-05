import numpy as np

from tokenizer import read_file_and_tokenize, str_to_int, get_batches
from attention import create_random_matrix, generate_qkv, attention_forward, backward_attention
from feed_forward import feed_forward, backward_feed_forward
from lm_head import lm_head, cross_entropy_loss, backward_loss_and_softmax, backward_lm_head
from attention import softmax

class my_transformer():
    def __init__(self, embedding_dim:int,heads:int,batch_size:int,learning_rate:float,token_dict:dict=None):
        """Initializes the transformer model with the specified parameters."""
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.token_dict = token_dict if token_dict is not None else {}
        self.W = generate_qkv(embedding_dim, embedding_dim)
        self.W_ffn = {
            "expansion_matrix": create_random_matrix(embedding_dim, embedding_dim * 4),
            "compression_matrix": create_random_matrix(embedding_dim * 4, embedding_dim)
        }
        vocab_size = len(self.token_dict)
        if vocab_size > 0:
            self.embedding_matrix = create_random_matrix(vocab_size, embedding_dim)
            self.W_out = create_random_matrix(embedding_dim, vocab_size)
        else:
            self.embedding_matrix = None
            self.W_out = None

    def train(self,file_path:str):
        txt = read_file_and_tokenize(file_path)
        for word in txt:
            if word not in self.token_dict:
                self.token_dict[word]=len(self.token_dict)
        vocab_size = len(self.token_dict)
        if self.embedding_matrix is None:
            self.embedding_matrix = create_random_matrix(vocab_size, self.embedding_dim)
            self.W_out = create_random_matrix(self.embedding_dim, vocab_size)
        elif vocab_size > self.embedding_matrix.shape[0]:
            new_rows = vocab_size - self.embedding_matrix.shape[0]
            self.embedding_matrix = np.vstack([self.embedding_matrix, create_random_matrix(new_rows, self.embedding_dim)])
            self.W_out = np.hstack([self.W_out, create_random_matrix(self.embedding_dim, new_rows)])
        int_list = str_to_int(txt, self.token_dict)
        batches = get_batches(int_list, self.batch_size)

        for i,(fragment,target) in enumerate(batches):
            # IDA
            attention, attention_weights = attention_forward(self.embedding_matrix, fragment, self.W, self.heads) 
            h=self.embedding_matrix[fragment]+attention
            final_output = h+feed_forward(h, self.W_ffn["expansion_matrix"], self.W_ffn["compression_matrix"])
            logits = lm_head(final_output, self.W_out)


            # vuelta
            # Head
            d_logits = backward_loss_and_softmax(logits, target)
            d_final_output, self.W_out = backward_lm_head(d_logits, final_output, self.W_out, self.learning_rate)
            # gradiente de la red feed forward
            d_attention, self.W_ffn = backward_feed_forward(d_final_output, attention, self.W_ffn, self.learning_rate)
            self.W, self.embedding_matrix = backward_attention(d_attention, self.embedding_matrix, attention_weights, fragment, self.W, self.heads, self.learning_rate)

            #probabilities = softmax(logits)
            #loss = cross_entropy_loss(probabilities, target)
            #predicted_token_id = np.argmax(logits[-1])
            #predicted_word = list(self.token_dict.keys())[predicted_token_id]
            #print(f"Bloque {i} | Predicted token: {predicted_word} | Loss: {loss:.4f}")
        
    def predict(self, input_sequence, max_length: int = 10) -> list:
        """Predicts the next tokens based on an input string or list."""
        if not self.token_dict:
            raise ValueError("The model has not been trained yet. Please train the model before making predictions.")
        
        # 1. Si entra un string, lo convertimos a lista de palabras (limpiando y separando)
        if isinstance(input_sequence, str):
            input_sequence = input_sequence.lower().split()
        
        # 2. Convertir la secuencia de palabras a enteros usando tu diccionario
        int_sequence = str_to_int(input_sequence, self.token_dict)
        
        if not int_sequence:
            return ["Ninguna palabra de la entrada está en el vocabulario del modelo."]
        
        # 3. Bucle autorregresivo para generar nuevos tokens
        for _ in range(max_length):
            attention, _ = attention_forward(self.embedding_matrix, int_sequence, self.W, self.heads)
            x=self.embedding_matrix[fragment]
            h=x+attention
            final_output = h+feed_forward(h, self.W_ffn["expansion_matrix"], self.W_ffn["compression_matrix"])
            logits = lm_head(final_output, self.W_out)
            
            predicted_token_id = np.argmax(logits[-1])
            int_sequence.append(predicted_token_id)
        
        # 4. Diccionario inverso eficiente para traducir los IDs de vuelta a texto
        inv_dict = {v: k for k, v in self.token_dict.items()}
        predicted_words = [inv_dict.get(token_id, "<unk>") for token_id in int_sequence]
        
        return predicted_words
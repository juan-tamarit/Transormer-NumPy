import numpy as np

from tokenizer import read_file_and_tokenize, str_to_int, get_batches
from attention import create_random_matrix, generate_qkv, attention_forward, backward_attention
from feed_forward import feed_forward, backward_feed_forward
from lm_head import lm_head, cross_entropy_loss, backward_loss_and_softmax, backward_lm_head
from attention import softmax

def positional_encoding(seq_len:int,embedding_dim:int)->np.array:
    """Sinusoidal positional encoding: without it, attention cannot tell word order apart."""
    positions=np.arange(seq_len)[:,None]
    freqs =1 / 10000**(np.arange(0,embedding_dim,2)/embedding_dim)
    pe=np.zeros((seq_len,embedding_dim))
    pe[:,0::2]=np.sin(positions*freqs)
    pe[:,1::2]=np.cos(positions*freqs)
    return pe

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

    def forward(self, fragment:list)->tuple:
        """Forward pass. Returns the logits and the intermediate values needed by backward pass."""
        x=self.embedding_matrix[fragment] + positional_encoding(len(fragment),self.embedding_dim)
        attention, attention_weights = attention_forward(x, self.W, self.heads) 
        h=x+attention
        final_output = h+feed_forward(h, self.W_ffn["expansion_matrix"], self.W_ffn["compression_matrix"])
        logits = lm_head(final_output, self.W_out)
        return logits,(x,attention_weights,h,final_output)

    def train_step(self,fragment:list,target:list)->float:
        """Runs forward and barward on one fragment, updates the weights and returns the loss"""
        logits,(x,attention_weights,h,final_output)=self.forward(fragment)
        loss = cross_entropy_loss(softmax(logits), target)

        d_logits = backward_loss_and_softmax(logits, target)
        d_final_output, self.W_out = backward_lm_head(d_logits, final_output, self.W_out, self.learning_rate)
        
        d_ffn, self.W_ffn = backward_feed_forward(d_final_output, h, self.W_ffn, self.learning_rate)
        d_h=d_final_output+d_ffn

        self.W, d_attention_x = backward_attention(d_h,x, attention_weights, self.W, self.heads, self.learning_rate)
        np.add.at(self.embedding_matrix,fragment,-self.learning_rate*(d_h + d_attention_x))
        return loss

    def train(self,file_path:str)->float:
        """Trains one pass over the file and returns the mean loss"""
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

        losses=[self.train_step(fragment,target) for fragment,target in batches]
        return float(np.mean(losses))
        
    def predict(self, input_sequence, max_length: int = 10) -> list:
        """Predicts the next tokens based on an input string or list."""
        if not self.token_dict:
            raise ValueError("The model has not been trained yet. Please train the model before making predictions.")
        
        # 1. Si entra un string, lo convertimos a lista de palabras (limpiando y separando)
        if isinstance(input_sequence, str):
            input_sequence = input_sequence.lower().split()
        
        # 2. Convertir la secuencia de palabras a enteros usando tu diccionario
        int_sequence = str_to_int([w for w in input_sequence if w in self.token_dict], self.token_dict)
        
        if not int_sequence:
            return ["Ninguna palabra de la entrada está en el vocabulario del modelo."]
        
        # 3. Bucle autorregresivo para generar nuevos tokens
        for _ in range(max_length):
            logits,_ = self.forward(int_sequence)
            predicted_token_id = int(np.argmax(logits[-1]))
            int_sequence.append(predicted_token_id)
        
        # 4. Diccionario inverso eficiente para traducir los IDs de vuelta a texto
        inv_dict = {v: k for k, v in self.token_dict.items()}
        predicted_words = [inv_dict.get(token_id, "<unk>") for token_id in int_sequence]
        
        return predicted_words
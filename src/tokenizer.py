import re


def read_file_and_tokenize(file_path: str) -> list:
    """
    Reads a text file and tokenizes its content into a list of words and punctuation marks.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        text_lower = text.lower()
        pat = r"\w+|[:,;¡!¿?—\"\"\'\'()]|[.]+"
        txt = re.findall(pat, text_lower)
    return txt

def str_to_int(txt: list, token_dict: dict) -> list:
    """Converts a list of tokens into a list of corresponding integer indices based on the provided token dictionary."""
    result = []
    for word in txt:
        result.append(token_dict[word])
    return result


def get_batches(int_list: list, batch_size: int) -> list:
    """Splits the input list of integers into batches of the specified size, creating input-target pairs for training."""
    batches = []
    for i in range(0, len(int_list), batch_size):
        fragment = int_list[i:i + batch_size]
        target = int_list[i + 1:i + batch_size + 1]
        if len(fragment) < batch_size or len(target) < batch_size:
            break
        batches.append((fragment, target))
    return batches

def encrypt_railfence(text: str, rails: int) -> str:
    """
    Encrypts text using the Rail Fence (zig-zag transposition) cipher.
    This cipher keeps all original characters (including punctuation) but scrambles their order.
    """
    if rails <= 1 or len(text) <= rails:
        return text

    fence = [[] for _ in range(rails)]
    rail = 0
    direction = 1

    for char in text:
        fence[rail].append(char)
        rail += direction
        
        # Turn around if we hit the top or bottom rail
        if rail == rails - 1 or rail == 0:
            direction *= -1

    return "".join(["".join(r) for r in fence])

def decrypt_railfence(text: str, rails: int) -> str:
    """
    Decrypts text using the Rail Fence cipher by reconstructing the grid pattern.
    """
    if rails <= 1 or len(text) <= rails:
        return text

    # Step 1: Trace the zig-zag pattern to find where letters *should* go
    fence = [[] for _ in range(rails)]
    rail = 0
    direction = 1

    for i in range(len(text)):
        fence[rail].append(i)
        rail += direction
        if rail == rails - 1 or rail == 0:
            direction *= -1

    # Step 2: Fill in the reconstructed fence with the ciphertext characters
    pos_to_char = {}
    idx = 0
    for r in range(rails):
        for original_pos in fence[r]:
            pos_to_char[original_pos] = text[idx]
            idx += 1

    # Step 3: Read off the characters in their original positions
    result = "".join([pos_to_char[i] for i in range(len(text))])
    return result
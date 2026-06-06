import json
import math
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


def quantize_measurement(x, delta=0.25, bits=8):
    q = int(round(x / delta))
    return max(-(2 ** (bits - 1)), min(2 ** (bits - 1) - 1, q))


def shannon_entropy(byte_data):
    if not byte_data:
        return 0.0
    freq = {}
    for b in byte_data:
        freq[b] = freq.get(b, 0) + 1
    n = len(byte_data)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def encrypt_packet(measurement_dict, key=None):
    if key is None:
        key = get_random_bytes(16)
    nonce = get_random_bytes(12)
    aad = f"{measurement_dict['meter_id']}|{measurement_dict['time']}|AMI".encode()
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(aad)
    plaintext = json.dumps(measurement_dict).encode()
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    expansion_ratio = (len(ciphertext) + len(tag) + len(nonce)) / max(1, len(plaintext))
    entropy_gain = shannon_entropy(ciphertext) - shannon_entropy(plaintext)
    return {
        'ciphertext': ciphertext,
        'tag': tag,
        'nonce': nonce,
        'aad': aad,
        'key': key,
        'expansion_ratio': expansion_ratio,
        'entropy_gain': entropy_gain,
    }


def verify_and_decrypt(packet):
    cipher = AES.new(packet['key'], AES.MODE_GCM, nonce=packet['nonce'])
    cipher.update(packet['aad'])
    plaintext = cipher.decrypt_and_verify(packet['ciphertext'], packet['tag'])
    return json.loads(plaintext.decode()), 1

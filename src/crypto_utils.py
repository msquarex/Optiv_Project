import os
import base64
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class CryptoUtils:
    # A fixed salt for deriving the key from the admin password.
    # In a production environment, this could be randomized and stored securely.
    SALT = b'optic-project-salt-2026'

    @classmethod
    def get_key_from_password(cls, password: str) -> bytes:
        """Derives a secure 32-byte key from the given password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=cls.SALT,
            iterations=480000,
        )
        return kdf.derive(password.encode())

    @classmethod
    def encrypt_data(cls, password: str, plaintext: str) -> Tuple[str, str]:
        """
        Encrypts plaintext using ChaCha20-Poly1305.
        Returns base64 encoded (nonce, ciphertext).
        """
        key = cls.get_key_from_password(password)
        chacha = ChaCha20Poly1305(key)
        nonce = os.urandom(12)
        
        ciphertext = chacha.encrypt(nonce, plaintext.encode('utf-8'), None)
        
        encoded_nonce = base64.b64encode(nonce).decode('utf-8')
        encoded_ciphertext = base64.b64encode(ciphertext).decode('utf-8')
        
        return encoded_nonce, encoded_ciphertext

    @classmethod
    def decrypt_data(cls, password: str, encoded_nonce: str, encoded_ciphertext: str) -> str:
        """
        Decrypts base64 encoded nonce and ciphertext back to string.
        """
        key = cls.get_key_from_password(password)
        chacha = ChaCha20Poly1305(key)
        
        nonce = base64.b64decode(encoded_nonce)
        ciphertext = base64.b64decode(encoded_ciphertext)
        
        plaintext = chacha.decrypt(nonce, ciphertext, None)
        return plaintext.decode('utf-8')

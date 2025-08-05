"""
ops0 Hashing and Cryptographic Utilities

Secure hashing, checksums, and ID generation.
"""

import hashlib
import hmac
import secrets
import uuid
import base64
from pathlib import Path
from typing import Union, Optional, BinaryIO
from enum import Enum
import string
import time


class HashAlgorithm(Enum):
    """Supported hash algorithms"""
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"


def calculate_hash(
        data: Union[str, bytes],
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        encoding: str = 'utf-8'
) -> str:
    """
    Calculate hash of data.

    Args:
        data: Data to hash (string or bytes)
        algorithm: Hash algorithm to use
        encoding: String encoding (if data is string)

    Returns:
        Hex digest of hash

    Examples:
        >>> calculate_hash("hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
        >>> calculate_hash(b"binary data", HashAlgorithm.MD5)
        '1ebbd3e34237af26da5dc08a4e440464'
    """
    # Convert string to bytes
    if isinstance(data, str):
        data = data.encode(encoding)

    # Get hash function
    if algorithm == HashAlgorithm.MD5:
        hasher = hashlib.md5()
    elif algorithm == HashAlgorithm.SHA1:
        hasher = hashlib.sha1()
    elif algorithm == HashAlgorithm.SHA256:
        hasher = hashlib.sha256()
    elif algorithm == HashAlgorithm.SHA384:
        hasher = hashlib.sha384()
    elif algorithm == HashAlgorithm.SHA512:
        hasher = hashlib.sha512()
    elif algorithm == HashAlgorithm.SHA3_256:
        hasher = hashlib.sha3_256()
    elif algorithm == HashAlgorithm.SHA3_512:
        hasher = hashlib.sha3_512()
    elif algorithm == HashAlgorithm.BLAKE2B:
        hasher = hashlib.blake2b()
    elif algorithm == HashAlgorithm.BLAKE2S:
        hasher = hashlib.blake2s()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Calculate hash
    hasher.update(data)
    return hasher.hexdigest()


def calculate_file_hash(
        file_path: Union[str, Path],
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        chunk_size: int = 8192
) -> str:
    """
    Calculate hash of file contents.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm to use
        chunk_size: Read chunk size

    Returns:
        Hex digest of file hash

    Examples:
        >>> calculate_file_hash("/path/to/file.txt")
        'a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3'
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get hash function
    hasher_map = {
        HashAlgorithm.MD5: hashlib.md5,
        HashAlgorithm.SHA1: hashlib.sha1,
        HashAlgorithm.SHA256: hashlib.sha256,
        HashAlgorithm.SHA384: hashlib.sha384,
        HashAlgorithm.SHA512: hashlib.sha512,
        HashAlgorithm.SHA3_256: hashlib.sha3_256,
        HashAlgorithm.SHA3_512: hashlib.sha3_512,
        HashAlgorithm.BLAKE2B: hashlib.blake2b,
        HashAlgorithm.BLAKE2S: hashlib.blake2s,
    }

    hasher = hasher_map[algorithm]()

    # Read and hash file
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def calculate_content_hash(
        content: Any,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        include_type: bool = True
) -> str:
    """
    Calculate hash of any Python object.

    Args:
        content: Object to hash
        algorithm: Hash algorithm to use
        include_type: Include type information in hash

    Returns:
        Hex digest of content hash

    Examples:
        >>> calculate_content_hash({'key': 'value'})
        'a7c7b2e0b5c4d9f3e8a2b5c7d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7'
        >>> calculate_content_hash([1, 2, 3])
        'b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4'
    """
    # Create string representation
    if include_type:
        content_str = f"{type(content).__name__}:{repr(content)}"
    else:
        content_str = repr(content)

    # Sort dictionaries for consistent hashing
    if isinstance(content, dict):
        sorted_items = sorted(content.items())
        content_str = f"{type(content).__name__}:{sorted_items}" if include_type else str(sorted_items)

    return calculate_hash(content_str, algorithm)


def verify_checksum(
        data: Union[str, bytes, Path],
        expected_checksum: str,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256
) -> bool:
    """
    Verify data matches expected checksum.

    Args:
        data: Data to verify (string, bytes, or file path)
        expected_checksum: Expected checksum
        algorithm: Hash algorithm used

    Returns:
        True if checksum matches

    Examples:
        >>> data = "hello world"
        >>> checksum = calculate_hash(data)
        >>> verify_checksum(data, checksum)
        True
    """
    # Calculate actual checksum
    if isinstance(data, (str, Path)) and Path(data).exists():
        actual_checksum = calculate_file_hash(data, algorithm)
    else:
        actual_checksum = calculate_hash(data, algorithm)

    # Compare (case-insensitive)
    return actual_checksum.lower() == expected_checksum.lower()


def generate_id(
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        length: int = 8,
        include_timestamp: bool = False
) -> str:
    """
    Generate unique ID.

    Args:
        prefix: Optional prefix
        suffix: Optional suffix
        length: Length of random part
        include_timestamp: Include timestamp in ID

    Returns:
        Generated ID

    Examples:
        >>> generate_id()
        'a3f2c8e9'
        >>> generate_id(prefix='user_', length=12)
        'user_a3f2c8e9b1d4'
        >>> generate_id(prefix='job_', include_timestamp=True)
        'job_1234567890_a3f2c8e9'
    """
    parts = []

    if prefix:
        parts.append(prefix)

    if include_timestamp:
        parts.append(str(int(time.time())))

    # Generate random part
    chars = string.ascii_lowercase + string.digits
    random_part = ''.join(secrets.choice(chars) for _ in range(length))
    parts.append(random_part)

    if suffix:
        parts.append(suffix)

    return '_'.join(filter(None, parts))


def generate_token(
        length: int = 32,
        urlsafe: bool = True,
        prefix: Optional[str] = None
) -> str:
    """
    Generate secure random token.

    Args:
        length: Token length in bytes
        urlsafe: Generate URL-safe token
        prefix: Optional prefix

    Returns:
        Generated token

    Examples:
        >>> generate_token()
        'AbC123XyZ789...'  # 32 bytes, base64 encoded
        >>> generate_token(length=16, prefix='tok_')
        'tok_AbC123XyZ789...'
    """
    # Generate random bytes
    token_bytes = secrets.token_bytes(length)

    # Encode
    if urlsafe:
        token = base64.urlsafe_b64encode(token_bytes).decode('ascii').rstrip('=')
    else:
        token = token_bytes.hex()

    # Add prefix
    if prefix:
        token = f"{prefix}{token}"

    return token


def create_signature(
        message: Union[str, bytes],
        secret_key: Union[str, bytes],
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        encoding: str = 'utf-8'
) -> str:
    """
    Create HMAC signature for message.

    Args:
        message: Message to sign
        secret_key: Secret key for signing
        algorithm: Hash algorithm to use
        encoding: String encoding

    Returns:
        Hex digest of signature

    Examples:
        >>> create_signature("hello", "secret")
        '88aab3ede8d3adf94d26ab90d3bafd4a2083070c3bcce9c014ee04a443847c0b'
    """
    # Convert strings to bytes
    if isinstance(message, str):
        message = message.encode(encoding)
    if isinstance(secret_key, str):
        secret_key = secret_key.encode(encoding)

    # Map algorithm to hashlib function
    algorithm_map = {
        HashAlgorithm.MD5: hashlib.md5,
        HashAlgorithm.SHA1: hashlib.sha1,
        HashAlgorithm.SHA256: hashlib.sha256,
        HashAlgorithm.SHA384: hashlib.sha384,
        HashAlgorithm.SHA512: hashlib.sha512,
        HashAlgorithm.SHA3_256: hashlib.sha3_256,
        HashAlgorithm.SHA3_512: hashlib.sha3_512,
    }

    if algorithm not in algorithm_map:
        raise ValueError(f"Unsupported algorithm for HMAC: {algorithm}")

    # Create signature
    return hmac.new(
        secret_key,
        message,
        algorithm_map[algorithm]
    ).hexdigest()


def verify_signature(
        message: Union[str, bytes],
        signature: str,
        secret_key: Union[str, bytes],
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        encoding: str = 'utf-8'
) -> bool:
    """
    Verify HMAC signature.

    Args:
        message: Original message
        signature: Signature to verify
        secret_key: Secret key used for signing
        algorithm: Hash algorithm used
        encoding: String encoding

    Returns:
        True if signature is valid

    Examples:
        >>> msg = "hello"
        >>> key = "secret"
        >>> sig = create_signature(msg, key)
        >>> verify_signature(msg, sig, key)
        True
    """
    # Calculate expected signature
    expected_signature = create_signature(message, secret_key, algorithm, encoding)

    # Constant-time comparison to prevent timing attacks
    return hmac.compare_digest(expected_signature, signature)


# Utility functions for common use cases

def generate_uuid() -> str:
    """Generate UUID4 string"""
    return str(uuid.uuid4())


def generate_short_id() -> str:
    """Generate short unique ID (8 chars)"""
    return generate_id(length=8)


def generate_api_key(prefix: str = "key_") -> str:
    """Generate API key with prefix"""
    return generate_token(length=32, prefix=prefix)


def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, bytes]:
    """
    Hash password with salt.

    Args:
        password: Password to hash
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (hash, salt)

    Examples:
        >>> hash_value, salt = hash_password("mypassword")
        >>> verify_password("mypassword", hash_value, salt)
        True
    """
    if salt is None:
        salt = secrets.token_bytes(32)

    # Use PBKDF2 for password hashing
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000  # iterations
    )

    return key.hex(), salt


def verify_password(password: str, password_hash: str, salt: bytes) -> bool:
    """
    Verify password against hash.

    Args:
        password: Password to verify
        password_hash: Stored password hash
        salt: Salt used for hashing

    Returns:
        True if password matches
    """
    # Recalculate hash
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000
    )

    # Constant-time comparison
    return hmac.compare_digest(key.hex(), password_hash)
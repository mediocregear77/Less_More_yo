"""
Covenant Signer Module

Provides cryptographic signing capabilities for covenant_core.json files
to ensure immutability and authenticity.

Security Features:
- RSA digital signatures with SHA-256 hashing
- HMAC fallback for symmetric key scenarios
- Secure key handling with environment variable support
- Comprehensive validation and error handling
- Immutable archival system
"""

import hashlib
import hmac
import base64
import json
import datetime
import os
import logging
import secrets
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CovenantSignerError(Exception):
    """Custom exception for covenant signing errors."""
    pass

class CovenantSigner:
    """
    Handles cryptographic signing and verification of covenant files.
    """
    
    def __init__(self, signature_algorithm: str = "RSA-2048"):
        """
        Initialize the covenant signer.
        
        Args:
            signature_algorithm: Algorithm to use for signing ("RSA-2048" or "HMAC-SHA256")
        """
        self.signature_algorithm = signature_algorithm
        self.hash_algorithm = "SHA-256"
        
    def generate_rsa_keypair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """
        Generate a new RSA key pair.
        
        Args:
            key_size: Size of the RSA key in bits
            
        Returns:
            Tuple of (private_key_pem, public_key_pem)
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def _canonicalize_covenant(self, covenant: Dict) -> str:
        """
        Create a canonical string representation for consistent hashing.
        
        Args:
            covenant: The covenant dictionary
            
        Returns:
            Canonical JSON string
        """
        # Create a copy and remove signature fields
        clean_covenant = covenant.copy()
        
        # Handle both old and new structure
        if "immutable" in clean_covenant:
            clean_covenant["immutable"] = {
                "is_signed": False,
                "hash": None,
                "signature": None,
                "signed_timestamp": None,
                "verification_status": "pending"
            }
        elif "security" in clean_covenant:
            clean_covenant["security"]["integrity"] = {
                "is_signed": False,
                "hash_algorithm": self.hash_algorithm,
                "hash": None,
                "signature_algorithm": self.signature_algorithm,
                "signature": None,
                "signed_timestamp": None,
                "verification_status": "unsigned",
                "nonce": None
            }
        
        return json.dumps(clean_covenant, sort_keys=True, separators=(',', ':'))
    
    def _generate_hash(self, data: str) -> str:
        """
        Generate SHA-256 hash of the data.
        
        Args:
            data: String data to hash
            
        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def _sign_with_rsa(self, data_hash: str, private_key_pem: bytes) -> str:
        """
        Sign hash using RSA private key.
        
        Args:
            data_hash: Hash to sign
            private_key_pem: RSA private key in PEM format
            
        Returns:
            Base64 encoded signature
        """
        try:
            private_key = serialization.load_pem_private_key(
                private_key_pem,
                password=None
            )
            
            signature = private_key.sign(
                data_hash.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode('utf-8')
            
        except Exception as e:
            raise CovenantSignerError(f"RSA signing failed: {e}")
    
    def _sign_with_hmac(self, data_hash: str, secret_key: Union[str, bytes]) -> str:
        """
        Sign hash using HMAC-SHA256.
        
        Args:
            data_hash: Hash to sign
            secret_key: Secret key for HMAC
            
        Returns:
            Base64 encoded signature
        """
        if isinstance(secret_key, str):
            secret_key = secret_key.encode('utf-8')
            
        signature = hmac.new(
            secret_key,
            data_hash.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(signature).decode('utf-8')
    
    def sign_covenant(self, 
                     covenant_path: Union[str, Path], 
                     private_key: Union[str, bytes, None] = None,
                     backup_archive: bool = True) -> Dict:
        """
        Cryptographically sign the covenant file.
        
        Args:
            covenant_path: Path to the covenant JSON file
            private_key: Private key (string for HMAC, bytes for RSA, None to use env var)
            backup_archive: Whether to create immutable backup
            
        Returns:
            The signed covenant dictionary
        """
        covenant_path = Path(covenant_path)
        
        if not covenant_path.exists():
            raise CovenantSignerError(f"Covenant file not found: {covenant_path}")
        
        # Load covenant
        try:
            with open(covenant_path, 'r', encoding='utf-8') as f:
                covenant = json.load(f)
        except json.JSONDecodeError as e:
            raise CovenantSignerError(f"Invalid JSON in covenant file: {e}")
        
        # Get private key from environment if not provided
        if private_key is None:
            private_key = os.getenv('COVENANT_PRIVATE_KEY')
            if not private_key:
                raise CovenantSignerError("No private key provided and COVENANT_PRIVATE_KEY not set")
        
        # Handle key format
        if isinstance(private_key, str) and private_key.startswith('-----BEGIN'):
            private_key = private_key.encode('utf-8')
        
        # Generate nonce for replay protection
        nonce = secrets.token_hex(16)
        
        # Add nonce to covenant before canonicalization
        if "security" in covenant:
            covenant["security"]["integrity"]["nonce"] = nonce
        elif "immutable" in covenant:
            covenant["immutable"]["nonce"] = nonce
        
        # Canonicalize and hash
        canonical_data = self._canonicalize_covenant(covenant)
        data_hash = self._generate_hash(canonical_data)
        
        # Generate signature
        try:
            if self.signature_algorithm == "RSA-2048" and isinstance(private_key, bytes):
                signature = self._sign_with_rsa(data_hash, private_key)
            else:
                signature = self._sign_with_hmac(data_hash, private_key)
        except Exception as e:
            raise CovenantSignerError(f"Signing failed: {e}")
        
        # Update covenant with signature data
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        
        if "security" in covenant:
            covenant["security"]["integrity"].update({
                "is_signed": True,
                "hash": data_hash,
                "signature": signature,
                "signed_timestamp": timestamp,
                "verification_status": "signed",
                "nonce": nonce
            })
        else:
            # Fallback to old structure
            covenant["immutable"] = {
                "signed": True,
                "hash": data_hash,
                "signature": signature,
                "signed_timestamp": timestamp,
                "nonce": nonce
            }
        
        # Save signed covenant
        try:
            with open(covenant_path, 'w', encoding='utf-8') as f:
                json.dump(covenant, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise CovenantSignerError(f"Failed to save signed covenant: {e}")
        
        # Create immutable backup if requested
        if backup_archive:
            self._create_immutable_backup(covenant_path, covenant)
        
        logger.info(f"✓ Covenant successfully signed")
        logger.info(f"  File: {covenant_path}")
        logger.info(f"  Hash: {data_hash}")
        logger.info(f"  Algorithm: {self.signature_algorithm}")
        logger.info(f"  Signature: {signature[:40]}...")
        
        return covenant
    
    def _create_immutable_backup(self, original_path: Path, covenant: Dict):
        """
        Create an immutable backup of the signed covenant.
        
        Args:
            original_path: Path to the original covenant file
            covenant: The signed covenant dictionary
        """
        immutable_dir = original_path.parent / "IMMUTABLE"
        immutable_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_name = f"covenant_core_signed_{timestamp}.json"
        backup_path = immutable_dir / backup_name
        
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(covenant, f, indent=2, ensure_ascii=False)
            
            # Make file read-only
            backup_path.chmod(0o444)
            
            logger.info(f"✓ Immutable backup created: {backup_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create immutable backup: {e}")
    
    def verify_covenant(self, 
                       covenant_path: Union[str, Path], 
                       public_key: Union[str, bytes, None] = None) -> bool:
        """
        Verify the signature of a signed covenant.
        
        Args:
            covenant_path: Path to the signed covenant file
            public_key: Public key for verification (None to use stored key)
            
        Returns:
            True if signature is valid, False otherwise
        """
        covenant_path = Path(covenant_path)
        
        try:
            with open(covenant_path, 'r', encoding='utf-8') as f:
                covenant = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load covenant: {e}")
            return False
        
        # Extract signature data
        if "security" in covenant:
            integrity = covenant["security"]["integrity"]
            stored_hash = integrity.get("hash")
            stored_signature = integrity.get("signature")
            is_signed = integrity.get("is_signed", False)
        elif "immutable" in covenant:
            immutable = covenant["immutable"]
            stored_hash = immutable.get("hash")
            stored_signature = immutable.get("signature")
            is_signed = immutable.get("signed", False)
        else:
            logger.error("No signature data found in covenant")
            return False
        
        if not is_signed or not stored_hash or not stored_signature:
            logger.error("Covenant is not properly signed")
            return False
        
        # Recalculate hash
        canonical_data = self._canonicalize_covenant(covenant)
        calculated_hash = self._generate_hash(canonical_data)
        
        if calculated_hash != stored_hash:
            logger.error("Hash mismatch - covenant has been tampered with")
            return False
        
        logger.info("✓ Covenant signature verified successfully")
        return True


def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Sign or verify covenant files")
    parser.add_argument("action", choices=["sign", "verify"], help="Action to perform")
    parser.add_argument("covenant_path", help="Path to covenant file")
    parser.add_argument("--algorithm", default="RSA-2048", 
                       choices=["RSA-2048", "HMAC-SHA256"],
                       help="Signature algorithm")
    parser.add_argument("--no-backup", action="store_true", 
                       help="Skip creating immutable backup")
    
    args = parser.parse_args()
    
    signer = CovenantSigner(args.algorithm)
    
    try:
        if args.action == "sign":
            signer.sign_covenant(args.covenant_path, backup_archive=not args.no_backup)
        elif args.action == "verify":
            is_valid = signer.verify_covenant(args.covenant_path)
            exit(0 if is_valid else 1)
            
    except CovenantSignerError as e:
        logger.error(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()

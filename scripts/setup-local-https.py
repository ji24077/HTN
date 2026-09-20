"""Create a localhost certificate and trust its development CA on macOS."""

import ipaddress
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

root = Path(__file__).resolve().parents[1]
tls = root / ".local" / "tls"
tls.mkdir(parents=True, exist_ok=True)
now = datetime.now(timezone.utc)
ca_path, ca_key_path = tls / "root.pem", tls / "root-key.pem"
if ca_path.exists() and ca_key_path.exists():
    ca = x509.load_pem_x509_certificate(ca_path.read_bytes())
    ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)
else:
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "Dispatch Local Development CA")]
    )
    ca = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(False, False, False, False, False, True, True, False, False),
            critical=True,
        )
        .add_extension(
            x509.NameConstraints(
                permitted_subtrees=[
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_network("127.0.0.1/32")),
                ],
                excluded_subtrees=None,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    ca_key_path.write_bytes(
        ca_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    ca_key_path.chmod(0o600)

key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
leaf = (
    x509.CertificateBuilder()
    .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
    .issuer_name(ca.subject)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(now - timedelta(minutes=5))
    .not_valid_after(now + timedelta(days=90))
    .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
    .add_extension(
        x509.SubjectAlternativeName(
            [
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
            ]
        ),
        critical=False,
    )
    .add_extension(
        x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
    )
    .sign(ca_key, hashes.SHA256())
)
(tls / "localhost.pem").write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
key_path = tls / "localhost-key.pem"
key_path.write_bytes(
    key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
)
key_path.chmod(0o600)
if sys.platform == "darwin":
    subprocess.run(
        [
            "security",
            "add-trusted-cert",
            "-r",
            "trustRoot",
            "-p",
            "ssl",
            "-k",
            str(Path.home() / "Library/Keychains/login.keychain-db"),
            str(ca_path),
        ],
        check=True,
    )
    print("Local HTTPS ready. Restart the frontend to load the certificate.")
else:
    print(f"Trust {ca_path} in your browser before using localhost HTTPS.")

#!/bin/bash
set -euo pipefail

echo "Generating self-signed certificate for localhost..."

cat > cert_config.conf <<'EOF'
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = CA
L = San Francisco
O = Local Dev
OU = WebSocket Server
CN = localhost

[v3_req]
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

# Generate an unencrypted 2048-bit RSA private key
openssl genrsa -out server.key 2048

# Create self-signed cert (valid for 365 days) and include SAN from config
openssl req -x509 -new -nodes -key server.key -sha256 -days 365 \
    -out server.crt -config cert_config.conf -extensions v3_req

# Cleanup
rm cert_config.conf

chmod 600 server.key
chmod 644 server.crt

echo "Certificate generated successfully!"
echo "Files created:"
echo "  server.key - Private key (unencrypted)"
echo "  server.crt - Certificate"
echo ""
echo "Certificate summary:"
openssl x509 -in server.crt -noout -text | sed -n '/Subject:/,/Signature Algorithm:/p'
echo ""
echo "To test TLS handshake with openssl:"
echo "  openssl s_client -connect 127.0.0.1:3000 -servername localhost -showcerts -debug"

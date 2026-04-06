# Secrets

Secrets are NOT stored in this repo. They are created on the cluster by running:

    bash scripts/create-secrets.sh

Before running, edit the script to set strong passwords.
Secrets are injected into pods as environment variables via secretKeyRef.
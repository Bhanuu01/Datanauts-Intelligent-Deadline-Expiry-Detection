# Secrets

Secrets are NOT stored in this repo. They are created on the cluster by running:

    bash scripts/create-secrets.sh

By default, the script generates strong random values and does not print them.

Useful options:

- `SHOW_GENERATED_SECRETS=true bash scripts/create-secrets.sh`
- `SAVE_CREDENTIALS_FILE=~/datanauts-secrets.env bash scripts/create-secrets.sh`
- `USE_EXISTING_SOURCE_SECRETS=true bash scripts/create-secrets.sh`

The script creates source secrets in the owning namespaces and then mirrors the
runtime credentials needed by ML jobs into the `ml` namespace via
`scripts/sync-runtime-secrets.sh`.

If your source secrets come from Bitnami Sealed Secrets, use
`USE_EXISTING_SOURCE_SECRETS=true` so the script only creates monitoring
credentials and refreshes the mirrored runtime copies.

Secrets are injected into pods as environment variables via `secretKeyRef`.

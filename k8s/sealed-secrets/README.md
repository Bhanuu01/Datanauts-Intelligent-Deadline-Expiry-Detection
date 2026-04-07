# Sealed Secrets

Secrets in this directory are encrypted using Bitnami Sealed Secrets.
They are SAFE to commit to Git — only the cluster controller can decrypt them.

## How it works
1. Sealed Secrets controller runs in kube-system namespace
2. kubeseal CLI encrypts secrets using the controller's public key
3. SealedSecret CRDs are committed to Git
4. On apply, the controller decrypts them into regular K8s Secrets

## To apply sealed secrets to a fresh cluster:
```bash
# Install controller first
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.26.3/controller.yaml

# Then apply sealed secrets
kubectl apply -f k8s/sealed-secrets/
```

## To re-seal (if you change passwords):
```bash
kubectl create secret generic paperless-secrets \
  --namespace=paperless \
  --from-literal=POSTGRES_PASSWORD=newpassword \
  --dry-run=client -o yaml \
  | kubeseal --format yaml > k8s/sealed-secrets/paperless-sealed-secret.yaml
```

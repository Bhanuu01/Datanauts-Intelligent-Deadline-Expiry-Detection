# Infrastructure

This project now follows a split infrastructure layout:

- `infra/terraform/openstack/`: provision OpenStack / Chameleon resources
- `infra/ansible/`: bootstrap the two-node k3s cluster after provisioning

The intended flow is:

```bash
cp infra/terraform/openstack/terraform.tfvars.example infra/terraform/openstack/terraform.tfvars
./scripts/infra-up.sh
```

That flow:

1. creates the control-plane and worker instances on your existing project network
2. assigns floating IPs
3. generates an Ansible inventory from Terraform outputs
4. installs and joins `k3s`
5. fetches a local kubeconfig ready for `kubectl`

The design is based on the Chameleon details visible in your screenshots:

- private network: `network_proj11`
- private network ID: `6f076311-d633-4455-999e-b95fedb86a7d`
- control-plane lease ID: `7cb53d22-1f26-406f-9b00-e77c9bdb3d5e`
- worker lease ID: `938578cf-ed9c-467f-8204-300032edec9e`
- default flavor: `m1.xlarge`

This stack assumes your leases already exist and are active. It does not create or extend Chameleon leases.

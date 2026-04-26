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

Before running it, fill in `infra/terraform/openstack/terraform.tfvars` with your own:

- target region
- private network and subnet IDs
- subnet CIDR
- image name
- OpenStack keypair name
- local SSH private key path
- control-plane and worker lease IDs
- control-plane and worker reservation flavor IDs

This stack assumes your leases already exist and are active. It does not create or extend Chameleon leases for you.

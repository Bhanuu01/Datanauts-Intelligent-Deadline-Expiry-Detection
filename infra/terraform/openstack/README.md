# OpenStack Provisioning

This Terraform stack provisions the infrastructure layer for this project on Chameleon / OpenStack.

It creates:

- one control-plane node
- one worker node
- one security group for cluster traffic
- one router from the project subnet to the external network
- one port per node on your existing private network
- one floating IP per node

It does not install Kubernetes itself. That step is handled by `infra/ansible/`.

## Usage

```bash
cp infra/terraform/openstack/terraform.tfvars.example infra/terraform/openstack/terraform.tfvars
terraform -chdir=infra/terraform/openstack init
terraform -chdir=infra/terraform/openstack apply
```

Or use the repo wrapper:

```bash
./scripts/infra-up.sh
```

## Required values to fill in

- `external_network_name`
- `image_name`
- confirm the exact OpenStack keypair name if it differs from `id_rsa_chameleon`
- choose which local private key to use: `~/.ssh/id_rsa_chameleon` or `~/.ssh/id_rsa_chameleon_2`

## Notes

- The default values already reflect the network and lease details from your screenshots.
- The defaults also include the current subnet ID and the two reservation IDs from your screenshots.
- On KVM@TACC, those reservation IDs are used as the leased flavor IDs when launching the two instances.
- The default region is `KVM@TACC` to match the KVM OpenStack endpoint and app credential flow.
- The defaults assume one control plane and one worker, both on `m1.xlarge`.
- You can change the two node flavors independently if you want to right-size the cluster later.

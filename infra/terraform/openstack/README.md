# OpenStack Provisioning

This Terraform stack provisions the infrastructure layer for this project on Chameleon / OpenStack.

It creates:

- one control-plane node
- one worker node
- four durable OpenStack block volumes for Paperless, platform state, ML model storage, and Prometheus data
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

- `region`
- `cluster_name`
- `private_network_id`
- `private_network_name`
- `private_subnet_id`
- `private_subnet_cidr`
- `external_network_name`
- `image_name`
- `ssh_keypair_name`
- `ssh_private_key_path`
- `control_plane_lease_id`
- `control_plane_reservation_id`
- `worker_lease_id`
- `worker_reservation_id`
- `bootstrap_object_storage_container` if you want a non-default Swift container for bootstrap artifacts

The reservation IDs above are the leased flavor IDs returned by Chameleon / Blazar for your active reservations. On `KVM@TACC`, the instances launch directly with those reservation flavor IDs.

## Notes

- This stack assumes your leases already exist and are active. It does not create or extend Chameleon leases.
- Instance and router names are derived from `cluster_name`, so a cluster named `demo` becomes `demo-controlnode`, `demo-workernode1`, and `demo-router`.
- `infra-up.sh` expects a filled-in `infra/terraform/openstack/terraform.tfvars` plus sourced OpenStack credentials.
- Durable PVC data is now backed by attached OpenStack block volumes rather than the default `local-path` provisioner.

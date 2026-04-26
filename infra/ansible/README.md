# Ansible Bootstrap

This directory bootstraps the Kubernetes layer after Terraform provisions the nodes.

Current playbook:

- `playbooks/bootstrap-k3s.yml`: installs `k3s` on the control-plane node and joins one worker

The inventory is generated automatically by `scripts/generate-ansible-inventory.sh` from Terraform outputs.

Manual run:

```bash
ansible-playbook -i infra/ansible/inventory/inventory.ini infra/ansible/playbooks/bootstrap-k3s.yml
```

The generated inventory should not be edited by hand for normal use. Regenerate it from Terraform outputs after each successful `terraform apply`.

Expected inventory groups:

- `control_plane`
- `workers`
- `k3s_cluster`

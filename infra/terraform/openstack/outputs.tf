output "cluster_name" {
  value       = var.cluster_name
  description = "Cluster name prefix."
}

output "ssh_user" {
  value       = var.ssh_user
  description = "SSH user for both nodes."
}

output "ssh_private_key_path" {
  value       = var.ssh_private_key_path
  description = "Local SSH private key path."
}

output "control_plane_name" {
  value       = openstack_compute_instance_v2.control_plane.name
  description = "Control-plane instance name."
}

output "control_plane_private_ip" {
  value       = openstack_networking_port_v2.control_plane.all_fixed_ips[0]
  description = "Control-plane private IP."
}

output "control_plane_public_ip" {
  value       = openstack_networking_floatingip_v2.control_plane.address
  description = "Control-plane floating IP."
}

output "worker_name" {
  value       = openstack_compute_instance_v2.worker.name
  description = "Worker instance name."
}

output "worker_private_ip" {
  value       = openstack_networking_port_v2.worker.all_fixed_ips[0]
  description = "Worker private IP."
}

output "worker_public_ip" {
  value       = openstack_networking_floatingip_v2.worker.address
  description = "Worker floating IP."
}

output "k3s_token" {
  value       = random_password.k3s_token.result
  description = "Shared cluster join token."
  sensitive   = true
}

output "enable_durable_block_storage" {
  value       = var.enable_durable_block_storage
  description = "Whether durable OpenStack block volumes are enabled."
}

output "using_existing_durable_volume" {
  value       = var.enable_durable_block_storage && var.existing_durable_volume_id != ""
  description = "Whether an existing shared durable block volume is being reused."
}

output "existing_durable_volume_device" {
  value       = var.existing_durable_volume_device
  description = "Guest device path for the shared durable volume when reusing an existing block volume."
}

output "paperless_volume_device" {
  value       = var.paperless_volume_device
  description = "Guest device path for the Paperless durable volume."
}

output "platform_volume_device" {
  value       = var.platform_volume_device
  description = "Guest device path for the platform durable volume."
}

output "ml_volume_device" {
  value       = var.ml_volume_device
  description = "Guest device path for the ML durable volume."
}

output "monitoring_volume_device" {
  value       = var.monitoring_volume_device
  description = "Guest device path for the monitoring durable volume."
}

output "bootstrap_object_storage_container" {
  value       = var.bootstrap_object_storage_container
  description = "Chameleon object storage container used for bootstrap artifacts."
}

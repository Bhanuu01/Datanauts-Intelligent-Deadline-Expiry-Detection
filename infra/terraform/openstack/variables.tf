variable "region" {
  description = "OpenStack region for the target Chameleon site."
  type        = string
}

variable "cluster_name" {
  description = "Name prefix for the two-node cluster."
  type        = string
}

variable "private_network_id" {
  description = "Existing private network UUID to attach instances to."
  type        = string
}

variable "private_network_name" {
  description = "Human-readable private network name."
  type        = string
}

variable "private_subnet_id" {
  description = "Subnet UUID inside the private project network."
  type        = string
}

variable "private_subnet_cidr" {
  description = "CIDR of the private project subnet used for node-to-node traffic."
  type        = string
}

variable "external_network_name" {
  description = "Floating IP pool or external network name."
  type        = string
}

variable "image_name" {
  description = "Glance image name for both nodes."
  type        = string
}

variable "ssh_keypair_name" {
  description = "Existing OpenStack keypair name."
  type        = string
}

variable "ssh_private_key_path" {
  description = "Local path to the private key matching ssh_keypair_name."
  type        = string
}

variable "ssh_user" {
  description = "SSH username present on the selected image."
  type        = string
  default     = "cc"
}

variable "ssh_allowed_cidr" {
  description = "CIDR allowed to SSH to the nodes."
  type        = string
  default     = "0.0.0.0/0"
}

variable "kubernetes_api_allowed_cidr" {
  description = "CIDR allowed to reach the Kubernetes API."
  type        = string
  default     = "0.0.0.0/0"
}

variable "http_allowed_cidr" {
  description = "CIDR allowed to reach HTTP services exposed later from the cluster."
  type        = string
  default     = "0.0.0.0/0"
}

variable "https_allowed_cidr" {
  description = "CIDR allowed to reach HTTPS services exposed later from the cluster."
  type        = string
  default     = "0.0.0.0/0"
}

variable "control_plane_lease_id" {
  description = "Existing Chameleon lease ID backing the control-plane reservation."
  type        = string
}

variable "control_plane_reservation_id" {
  description = "Blazar reservation flavor ID used as the leased flavor for the control-plane instance."
  type        = string
}

variable "worker_lease_id" {
  description = "Existing Chameleon lease ID backing the worker reservation."
  type        = string
}

variable "worker_reservation_id" {
  description = "Blazar reservation flavor ID used as the leased flavor for the worker instance."
  type        = string
}

variable "region" {
  description = "OpenStack region for the target Chameleon site."
  type        = string
  default     = "KVM@TACC"
}

variable "cluster_name" {
  description = "Name prefix for the two-node cluster."
  type        = string
  default     = "mlproject11"
}

variable "private_network_id" {
  description = "Existing private network UUID to attach instances to."
  type        = string
  default     = "6f076311-d633-4455-999e-b95fedb86a7d"
}

variable "private_network_name" {
  description = "Human-readable private network name."
  type        = string
  default     = "network_proj11"
}

variable "private_subnet_id" {
  description = "Subnet UUID inside the private project network."
  type        = string
  default     = "af5b60e1-c398-4a60-bb91-4cdd6b1fe690"
}

variable "private_subnet_cidr" {
  description = "CIDR of the private project subnet used for node-to-node traffic."
  type        = string
  default     = "192.168.1.0/24"
}

variable "external_network_name" {
  description = "Floating IP pool or external network name."
  type        = string
}

variable "image_name" {
  description = "Glance image name for both nodes."
  type        = string
}

variable "control_plane_flavor_name" {
  description = "Flavor used by the control-plane node."
  type        = string
  default     = "m1.xlarge"
}

variable "worker_flavor_name" {
  description = "Flavor used by the worker node."
  type        = string
  default     = "m1.xlarge"
}

variable "ssh_keypair_name" {
  description = "Existing OpenStack keypair name."
  type        = string
}

variable "ssh_private_key_path" {
  description = "Local path to the private key matching ssh_keypair_name."
  type        = string
  default     = "~/.ssh/id_rsa"
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
  default     = "7cb53d22-1f26-406f-9b00-e77c9bdb3d5e"
}

variable "control_plane_reservation_id" {
  description = "Blazar reservation flavor ID for the control-plane instance."
  type        = string
  default     = "71b2315d-a841-4efb-a98f-d58852bfeeb9"
}

variable "worker_lease_id" {
  description = "Existing Chameleon lease ID backing the worker reservation."
  type        = string
  default     = "938578cf-ed9c-467f-8204-300032edec9e"
}

variable "worker_reservation_id" {
  description = "Blazar reservation flavor ID for the worker instance."
  type        = string
  default     = "3b97c07c-f507-434f-8af3-7c06967d13fd"
}

variable "control_plane_name" {
  description = "Instance name for the control-plane node."
  type        = string
  default     = "mlproject11-controlnode"
}

variable "worker_name" {
  description = "Instance name for the worker node."
  type        = string
  default     = "mlproject11-workernode1"
}

variable "router_name" {
  description = "Router name used to connect the private subnet to the external network."
  type        = string
  default     = "mlproject11-router"
}

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

variable "enable_durable_block_storage" {
  description = "Whether to provision durable OpenStack block volumes for Kubernetes persistent data."
  type        = bool
  default     = true
}

variable "existing_durable_volume_id" {
  description = "Optional existing OpenStack block volume UUID to attach as the shared durable Kubernetes data disk."
  type        = string
  default     = ""
}

variable "existing_durable_volume_device" {
  description = "Guest device path for the attached existing shared durable block volume."
  type        = string
  default     = "/dev/vdb"
}

variable "paperless_volume_size_gib" {
  description = "Size of the Cinder volume that stores Paperless data, media, and database files."
  type        = number
  default     = 80
}

variable "platform_volume_size_gib" {
  description = "Size of the Cinder volume that stores MinIO, MLflow, and MLflow Postgres state."
  type        = number
  default     = 80
}

variable "ml_volume_size_gib" {
  description = "Size of the Cinder volume that stores shared ML datasets and model artifacts."
  type        = number
  default     = 80
}

variable "monitoring_volume_size_gib" {
  description = "Size of the Cinder volume that stores Prometheus TSDB data."
  type        = number
  default     = 40
}

variable "paperless_volume_device" {
  description = "Guest device path for the attached Paperless block volume."
  type        = string
  default     = "/dev/vdb"
}

variable "platform_volume_device" {
  description = "Guest device path for the attached platform block volume."
  type        = string
  default     = "/dev/vdc"
}

variable "ml_volume_device" {
  description = "Guest device path for the attached ML block volume."
  type        = string
  default     = "/dev/vdd"
}

variable "monitoring_volume_device" {
  description = "Guest device path for the attached monitoring block volume."
  type        = string
  default     = "/dev/vde"
}

variable "bootstrap_object_storage_container" {
  description = "Chameleon object storage container used for bootstrap artifacts and off-node backups."
  type        = string
  default     = "datanauts-bootstrap-artifacts"
}

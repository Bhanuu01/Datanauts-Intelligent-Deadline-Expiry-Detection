resource "random_password" "k3s_token" {
  length  = 40
  special = false
}

locals {
  control_plane_name = "${var.cluster_name}-controlnode"
  worker_name        = "${var.cluster_name}-workernode1"
  router_name        = "${var.cluster_name}-router"
}

data "openstack_networking_network_v2" "external" {
  name = var.external_network_name
}

resource "openstack_networking_secgroup_v2" "cluster" {
  name        = "${var.cluster_name}-k3s"
  description = "Security group for the Datanauts k3s cluster."
}

resource "openstack_networking_secgroup_rule_v2" "ssh_ingress" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = var.ssh_allowed_cidr
  security_group_id = openstack_networking_secgroup_v2.cluster.id
}

resource "openstack_networking_secgroup_rule_v2" "k8s_api_ingress" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 6443
  port_range_max    = 6443
  remote_ip_prefix  = var.kubernetes_api_allowed_cidr
  security_group_id = openstack_networking_secgroup_v2.cluster.id
}

resource "openstack_networking_secgroup_rule_v2" "http_ingress" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 80
  port_range_max    = 80
  remote_ip_prefix  = var.http_allowed_cidr
  security_group_id = openstack_networking_secgroup_v2.cluster.id
}

resource "openstack_networking_secgroup_rule_v2" "https_ingress" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 443
  port_range_max    = 443
  remote_ip_prefix  = var.https_allowed_cidr
  security_group_id = openstack_networking_secgroup_v2.cluster.id
}

resource "openstack_networking_secgroup_rule_v2" "icmp_ingress" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "icmp"
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.cluster.id
}

resource "openstack_networking_secgroup_rule_v2" "intra_cluster_tcp_ingress" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 1
  port_range_max    = 65535
  remote_ip_prefix  = var.private_subnet_cidr
  security_group_id = openstack_networking_secgroup_v2.cluster.id
}

resource "openstack_networking_secgroup_rule_v2" "intra_cluster_udp_ingress" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "udp"
  port_range_min    = 1
  port_range_max    = 65535
  remote_ip_prefix  = var.private_subnet_cidr
  security_group_id = openstack_networking_secgroup_v2.cluster.id
}

resource "openstack_networking_secgroup_rule_v2" "intra_cluster_icmp_ingress" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "icmp"
  remote_ip_prefix  = var.private_subnet_cidr
  security_group_id = openstack_networking_secgroup_v2.cluster.id
}

resource "openstack_networking_router_v2" "cluster" {
  name                = local.router_name
  admin_state_up      = true
  external_network_id = data.openstack_networking_network_v2.external.id
}

resource "openstack_networking_router_interface_v2" "cluster_private_subnet" {
  router_id = openstack_networking_router_v2.cluster.id
  subnet_id = var.private_subnet_id
}

resource "openstack_networking_port_v2" "control_plane" {
  name                  = "${var.cluster_name}-control-plane-port"
  network_id            = var.private_network_id
  admin_state_up        = true
  port_security_enabled = false
}

resource "openstack_networking_port_v2" "worker" {
  name                  = "${var.cluster_name}-worker-port"
  network_id            = var.private_network_id
  admin_state_up        = true
  port_security_enabled = false
}

resource "openstack_networking_floatingip_v2" "control_plane" {
  pool = var.external_network_name
}

resource "openstack_networking_floatingip_v2" "worker" {
  pool = var.external_network_name
}

resource "openstack_compute_instance_v2" "control_plane" {
  name       = local.control_plane_name
  image_name = var.image_name
  flavor_id  = var.control_plane_reservation_id
  key_pair   = var.ssh_keypair_name

  user_data = templatefile("${path.module}/templates/node-cloud-init.yaml.tftpl", {
    node_name = local.control_plane_name
  })

  network {
    port = openstack_networking_port_v2.control_plane.id
  }

  metadata = {
    role            = "control-plane"
    cluster_name    = var.cluster_name
    lease_id        = var.control_plane_lease_id
    reservation_id  = var.control_plane_reservation_id
    private_network = var.private_network_name
  }

}

resource "openstack_compute_instance_v2" "worker" {
  name       = local.worker_name
  image_name = var.image_name
  flavor_id  = var.worker_reservation_id
  key_pair   = var.ssh_keypair_name

  user_data = templatefile("${path.module}/templates/node-cloud-init.yaml.tftpl", {
    node_name = local.worker_name
  })

  network {
    port = openstack_networking_port_v2.worker.id
  }

  metadata = {
    role            = "worker"
    cluster_name    = var.cluster_name
    lease_id        = var.worker_lease_id
    reservation_id  = var.worker_reservation_id
    private_network = var.private_network_name
  }

}

resource "openstack_networking_floatingip_associate_v2" "control_plane" {
  floating_ip = openstack_networking_floatingip_v2.control_plane.address
  port_id     = openstack_networking_port_v2.control_plane.id

  depends_on = [openstack_networking_router_interface_v2.cluster_private_subnet]
}

resource "openstack_networking_floatingip_associate_v2" "worker" {
  floating_ip = openstack_networking_floatingip_v2.worker.address
  port_id     = openstack_networking_port_v2.worker.id

  depends_on = [openstack_networking_router_interface_v2.cluster_private_subnet]
}

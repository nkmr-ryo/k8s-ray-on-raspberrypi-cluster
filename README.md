# 🚀 Kubernetes + Ray on Raspberry Pi Cluster (ARM64)

This repository documents how to deploy **Ray (Core / Data / Train / Tune / Serve / RLlib)**  
on a **3-node Raspberry Pi 5 Kubernetes cluster** using **KubeRay**.

---

## ✨ What this repository covers

- Installing the **KubeRay Operator**
- Deploying a multi-node **RayCluster** using ARM64-compatible images
- Running Ray **Core / Data / Train / Tune / Serve / RLlib** on Kubernetes
- Basic verification scripts and configuration examples

---

## 🏗 Architecture Overview

- Raspberry Pi 5 x 3 (8GB RAM)
- Kubernetes (v1.34)
- KubeRay Operator
- RayCluster
- Head node x 1
- Worker nodes x2

---

## 📂 Repository Structure
- docs/ ... Step-by-step instructions 
- manifests/ ... Kubernetes manifests (KubeRay 
- Operator / RayCluster)
- scripts/ ... Setup & deployment scripts
- experiments/ ... Ray examples (Core / Data / Train / Tune / Serve / RLlib)

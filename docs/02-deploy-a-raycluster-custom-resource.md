# Deploy a RayCluster custom resource

This section explains how to deploy a RayCluster custom resource.

Once the KubeRay operator is running, you’re ready to deploy a RayCluster. Create a RayCluster Custom Resource (CR) in the **default** namespace.

```bash
# Deploy a sample RayCluster CR from the KubeRay Helm chart repo:
helm install raycluster kuberay/ray-cluster --version 1.5.0
```

Expected output:

```bash
NAME: raycluster
LAST DEPLOYED: Tue Dec  2 11:55:27 2025
NAMESPACE: default
STATUS: deployed
REVISION: 1
TEST SUITE: None
```

Once the RayCluster CR has been created, you can view it by running:

```bash
# Once the RayCluster CR has been created, you can view it by running:
kubectl get rayclusters
```

Expected output:

```bash
NAME                 DESIRED WORKERS   AVAILABLE WORKERS   CPUS   MEMORY   GPUS   STATUS   AGE
raycluster-kuberay   1                 1                   2      3G       0      ready    6m7s
```

The KubeRay operator detects the RayCluster object and starts your Ray cluster by creating head and worker pods. To view Ray cluster’s pods, run the following command:

```bash
# View the pods in the RayCluster named "raycluster-kuberay"
kubectl get pods --selector=ray.io/cluster=raycluster-kuberay
```

Expected output:

```bash
NAME                                          READY   STATUS    RESTARTS   AGE
raycluster-kuberay-head-ksfzt                 1/1     Running   0          7m21s
raycluster-kuberay-workergroup-worker-7shzx   1/1     Running   0          7m21s
```

    
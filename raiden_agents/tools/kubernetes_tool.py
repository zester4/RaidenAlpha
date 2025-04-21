# raiden_agents/tools/kubernetes_tool.py

from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream, portforward
from typing import Dict, List, Optional, Any, Union
import logging
import yaml
import base64
from .base_tool import BaseTool, ToolExecutionError

class KubernetesTool(BaseTool):
    """
    Comprehensive Kubernetes integration tool for managing and monitoring
    Kubernetes clusters, deployments, services, and other resources.
    """

    def __init__(self):
        super().__init__()
        self.name = "kubernetes_operations"
        self.description = "Manage and monitor Kubernetes resources including deployments, services, pods, and configurations"
        self.logger = logging.getLogger("gemini_agent.kubernetes_tool")
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Kubernetes client with error handling"""
        try:
            # Try loading from within cluster first
            try:
                config.load_incluster_config()
                self.logger.info("Loaded in-cluster Kubernetes configuration")
            except config.ConfigException:
                # Fall back to local kubeconfig
                config.load_kube_config()
                self.logger.info("Loaded local Kubernetes configuration")
            
            self.core_v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.custom_objects = client.CustomObjectsApi()
            self.networking_v1 = client.NetworkingV1Api()
        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes client: {str(e)}")
            raise ToolExecutionError(f"Kubernetes client initialization failed: {str(e)}")

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "get_resources",
                            "deploy",
                            "scale",
                            "delete",
                            "get_logs",
                            "exec_command",
                            "get_metrics",
                            "apply_manifest",
                            "get_events",
                            "port_forward"
                        ],
                        "description": "Kubernetes operation to perform"
                    },
                    "resource_type": {
                        "type": "string",
                        "enum": [
                            "pod",
                            "deployment",
                            "service",
                            "configmap",
                            "secret",
                            "namespace",
                            "ingress",
                            "node"
                        ],
                        "description": "Type of Kubernetes resource"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Kubernetes namespace",
                        "default": "default"
                    },
                    "name": {
                        "type": "string",
                        "description": "Resource name"
                    },
                    "manifest": {
                        "type": "object",
                        "description": "Kubernetes manifest for creation/update operations"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Additional parameters for the operation",
                        "additionalProperties": True
                    }
                },
                "required": ["action"]
            }
        }

    def execute(self, action: str, resource_type: Optional[str] = None,
                namespace: str = "default", name: Optional[str] = None,
                manifest: Optional[Dict] = None, parameters: Optional[Dict] = None) -> Dict:
        """Execute Kubernetes operations with comprehensive error handling"""
        try:
            # Map actions to methods
            action_map = {
                "get_resources": self._get_resources,
                "deploy": self._deploy_resource,
                "scale": self._scale_resource,
                "delete": self._delete_resource,
                "get_logs": self._get_pod_logs,
                "exec_command": self._exec_command,
                "get_metrics": self._get_metrics,
                "apply_manifest": self._apply_manifest,
                "get_events": self._get_events,
                "port_forward": self._port_forward
            }

            if action not in action_map:
                raise ToolExecutionError(f"Unsupported action: {action}")

            # Execute the action
            result = action_map[action](
                resource_type=resource_type,
                namespace=namespace,
                name=name,
                manifest=manifest,
                parameters=parameters or {}
            )

            return {
                "success": True,
                "action": action,
                "result": result
            }

        except ApiException as e:
            error_msg = f"Kubernetes API error: {e.reason}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "action": action,
                "error": error_msg,
                "status_code": e.status
            }
        except Exception as e:
            error_msg = f"Operation failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "action": action,
                "error": error_msg
            }

    def _get_resources(self, resource_type: str, namespace: str = "default",
                      name: Optional[str] = None, **kwargs) -> Dict:
        """Get Kubernetes resources with filtering support"""
        try:
            if resource_type == "pod":
                if name:
                    result = self.core_v1.read_namespaced_pod(name, namespace)
                else:
                    result = self.core_v1.list_namespaced_pod(namespace)
            elif resource_type == "deployment":
                if name:
                    result = self.apps_v1.read_namespaced_deployment(name, namespace)
                else:
                    result = self.apps_v1.list_namespaced_deployment(namespace)
            elif resource_type == "service":
                if name:
                    result = self.core_v1.read_namespaced_service(name, namespace)
                else:
                    result = self.core_v1.list_namespaced_service(namespace)
            else:
                raise ToolExecutionError(f"Unsupported resource type: {resource_type}")

            return self._format_k8s_response(result)

        except ApiException as e:
            raise ToolExecutionError(f"Failed to get resources: {e.reason}")

    def _deploy_resource(self, manifest: Dict, namespace: str = "default", **kwargs) -> Dict:
        """Deploy or update Kubernetes resources"""
        try:
            # Validate manifest
            if not manifest or "kind" not in manifest:
                raise ToolExecutionError("Invalid manifest: missing 'kind' field")

            # Create or update based on resource kind
            kind = manifest["kind"].lower()
            name = manifest["metadata"]["name"]

            if kind == "deployment":
                try:
                    # Try to update if exists
                    self.apps_v1.replace_namespaced_deployment(
                        name=name,
                        namespace=namespace,
                        body=manifest
                    )
                    action = "updated"
                except ApiException as e:
                    if e.status == 404:
                        # Create if doesn't exist
                        self.apps_v1.create_namespaced_deployment(
                            namespace=namespace,
                            body=manifest
                        )
                        action = "created"
                    else:
                        raise

            elif kind == "service":
                try:
                    self.core_v1.replace_namespaced_service(
                        name=name,
                        namespace=namespace,
                        body=manifest
                    )
                    action = "updated"
                except ApiException as e:
                    if e.status == 404:
                        self.core_v1.create_namespaced_service(
                            namespace=namespace,
                            body=manifest
                        )
                        action = "created"
                    else:
                        raise

            else:
                raise ToolExecutionError(f"Unsupported resource kind: {kind}")

            return {
                "message": f"Successfully {action} {kind} '{name}' in namespace '{namespace}'",
                "resource": {
                    "kind": kind,
                    "name": name,
                    "namespace": namespace
                }
            }

        except Exception as e:
            raise ToolExecutionError(f"Deployment failed: {str(e)}")

    def _scale_resource(self, resource_type: str, name: str, 
                       namespace: str = "default", parameters: Dict = None, **kwargs) -> Dict:
        """Scale Kubernetes resources"""
        try:
            if resource_type != "deployment":
                raise ToolExecutionError("Scaling is only supported for deployments")

            replicas = parameters.get("replicas")
            if not isinstance(replicas, int) or replicas < 0:
                raise ToolExecutionError("Invalid replicas value")

            # Scale the deployment
            scale_obj = client.V1Scale(
                spec=client.V1ScaleSpec(replicas=replicas)
            )
            
            result = self.apps_v1.patch_namespaced_deployment_scale(
                name=name,
                namespace=namespace,
                body=scale_obj
            )

            return {
                "message": f"Successfully scaled deployment '{name}' to {replicas} replicas",
                "current_replicas": result.spec.replicas
            }

        except Exception as e:
            raise ToolExecutionError(f"Scaling failed: {str(e)}")

    def _delete_resource(self, resource_type: str, name: str, 
                        namespace: str = "default", **kwargs) -> Dict:
        """Delete Kubernetes resources"""
        try:
            if resource_type == "deployment":
                self.apps_v1.delete_namespaced_deployment(
                    name=name,
                    namespace=namespace
                )
            elif resource_type == "service":
                self.core_v1.delete_namespaced_service(
                    name=name,
                    namespace=namespace
                )
            elif resource_type == "pod":
                self.core_v1.delete_namespaced_pod(
                    name=name,
                    namespace=namespace
                )
            else:
                raise ToolExecutionError(f"Unsupported resource type for deletion: {resource_type}")

            return {
                "message": f"Successfully deleted {resource_type} '{name}' from namespace '{namespace}'"
            }

        except Exception as e:
            raise ToolExecutionError(f"Deletion failed: {str(e)}")

    def _get_pod_logs(self, name: str, namespace: str = "default", 
                     parameters: Dict = None, **kwargs) -> Dict:
        """Get pod logs with optional parameters"""
        try:
            params = {
                "container": parameters.get("container"),
                "tail_lines": parameters.get("tail_lines"),
                "timestamps": parameters.get("timestamps", True)
            }
            
            logs = self.core_v1.read_namespaced_pod_log(
                name=name,
                namespace=namespace,
                **{k: v for k, v in params.items() if v is not None}
            )

            return {
                "pod_name": name,
                "namespace": namespace,
                "logs": logs
            }

        except Exception as e:
            raise ToolExecutionError(f"Failed to get pod logs: {str(e)}")

    def _exec_command(self, name: str, namespace: str = "default", 
                     parameters: Dict = None, **kwargs) -> Dict:
        """Execute commands in pods"""
        try:
            command = parameters.get("command")
            if not command:
                raise ToolExecutionError("No command specified")

            container = parameters.get("container")
            
            exec_command = ["/bin/sh", "-c", command]
            resp = stream(
                self.core_v1.connect_get_namespaced_pod_exec,
                name,
                namespace,
                container=container,
                command=exec_command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False
            )

            return {
                "pod_name": name,
                "namespace": namespace,
                "command": command,
                "output": resp
            }

        except Exception as e:
            raise ToolExecutionError(f"Command execution failed: {str(e)}")

    def _get_metrics(self, resource_type: str = None, name: Optional[str] = None, 
                    namespace: str = "default", **kwargs) -> Dict:
        """Get resource metrics"""
        try:
            # Requires metrics-server to be installed
            custom_api = client.CustomObjectsApi()
            
            if resource_type == "pod":
                metrics = custom_api.list_namespaced_custom_object(
                    group="metrics.k8s.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="pods"
                )
            elif resource_type == "node":
                metrics = custom_api.list_cluster_custom_object(
                    group="metrics.k8s.io",
                    version="v1beta1",
                    plural="nodes"
                )
            else:
                raise ToolExecutionError(f"Metrics not supported for resource type: {resource_type}")

            # Filter by name if provided
            if name:
                metrics["items"] = [
                    item for item in metrics["items"]
                    if item["metadata"]["name"] == name
                ]

            return {
                "resource_type": resource_type,
                "metrics": metrics["items"]
            }

        except Exception as e:
            raise ToolExecutionError(f"Failed to get metrics: {str(e)}")

    def _apply_manifest(self, manifest: Dict, namespace: str = "default", **kwargs) -> Dict:
        """Apply Kubernetes manifest with strategic merge patch"""
        try:
            if not manifest or "kind" not in manifest or "metadata" not in manifest:
                raise ToolExecutionError("Invalid manifest format")

            kind = manifest["kind"].lower()
            name = manifest["metadata"]["name"]

            # Determine API based on resource kind
            if kind == "deployment":
                api_instance = self.apps_v1
                api_func = "patch_namespaced_deployment"
            elif kind == "service":
                api_instance = self.core_v1
                api_func = "patch_namespaced_service"
            elif kind == "configmap":
                api_instance = self.core_v1
                api_func = "patch_namespaced_config_map"
            else:
                raise ToolExecutionError(f"Unsupported resource kind: {kind}")

            # Apply the manifest
            result = getattr(api_instance, api_func)(
                name=name,
                namespace=namespace,
                body=manifest,
                field_manager="kubernetes-tool"
            )

            return {
                "message": f"Successfully applied {kind} manifest",
                "resource": self._format_k8s_response(result)
            }

        except Exception as e:
            raise ToolExecutionError(f"Failed to apply manifest: {str(e)}")

    def _get_events(self, resource_type: Optional[str] = None, 
                   name: Optional[str] = None, namespace: str = "default", **kwargs) -> Dict:
        """Get Kubernetes events"""
        try:
            field_selector = []
            if resource_type and name:
                field_selector.append(f"involvedObject.kind={resource_type}")
                field_selector.append(f"involvedObject.name={name}")

            events = self.core_v1.list_namespaced_event(
                namespace=namespace,
                field_selector=",".join(field_selector) if field_selector else None
            )

            return {
                "events": [
                    {
                        "type": event.type,
                        "reason": event.reason,
                        "message": event.message,
                        "count": event.count,
                        "first_timestamp": event.first_timestamp,
                        "last_timestamp": event.last_timestamp,
                        "involved_object": {
                            "kind": event.involved_object.kind,
                            "name": event.involved_object.name
                        }
                    }
                    for event in events.items
                ]
            }

        except Exception as e:
            raise ToolExecutionError(f"Failed to get events: {str(e)}")

    def _port_forward(self, resource_type: str, name: str, 
                     namespace: str = "default", parameters: Dict = None, **kwargs) -> Dict:
        """Set up port forwarding"""
        try:
            local_port = parameters.get("local_port")
            remote_port = parameters.get("remote_port")
            
            if not local_port or not remote_port:
                raise ToolExecutionError("Both local_port and remote_port are required")

            # Start port forwarding in a separate process
            import multiprocessing
            process = multiprocessing.Process(
                target=portforward(
                    namespace=namespace,
                    pod_name=name if resource_type == "pod" else None,
                    local_port=local_port,
                    remote_port=remote_port
                )
            )
            process.start()

            return {
                "message": f"Port forwarding started: localhost:{local_port} -> {remote_port}",
                "process_id": process.pid
            }

        except Exception as e:
            raise ToolExecutionError(f"Port forwarding failed: {str(e)}")

    def _format_k8s_response(self, obj: Any) -> Dict:
        """Format Kubernetes API response"""
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif isinstance(obj, (list, dict)):
            return obj
        else:
            return {"raw_response": str(obj)}
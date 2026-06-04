import torch
import torch.nn.functional as F
from src.config import IMG_SIZE, device

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.grads = None
        self.acts  = None
        
        # Register hooks for gradients and activations
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)
        
    def _fwd_hook(self, module, input, output):
        self.acts = output.detach()
        
    def _bwd_hook(self, module, grad_input, grad_output):
        self.grads = grad_output[0].detach()
        
    def generate(self, x, class_idx=None):
        """Generates a normalized Grad-CAM saliency map for a single input image."""
        self.model.eval()
        
        # Forward pass
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(1).item()
            
        # Backward pass
        self.model.zero_grad()
        out[0, class_idx].backward()
        
        # Calculate weights from global average pooling of gradients
        weights = self.grads.mean(dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps followed by ReLU
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        
        # Resize to original input size
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Min-Max normalize
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        probs = F.softmax(out, dim=1).detach().cpu().numpy()[0]
        
        return cam, class_idx, probs

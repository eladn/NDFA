import torch


__all__ = ['ModuleWithDbgTestGradsMixin']


class ModuleWithDbgTestGradsMixin:
    def __init__(self, is_enabled: bool = True):
        self.is_grads_dbg_enabled = is_enabled
        self._dbg__tensors_to_check_grads = {}

    def dbg_log_new_fwd(self):
        if not self.is_grads_dbg_enabled:
            return
        self._dbg__tensors_to_check_grads = {}

    def dbg_log_tensor_during_fwd(self, name: str, tensor: torch.Tensor):
        if not self.is_grads_dbg_enabled:
            return
        assert name not in self._dbg__tensors_to_check_grads
        self._dbg__tensors_to_check_grads[name] = tensor

    def dbg_test_grads_after_bwd(self, grad_test_example_idx: int = 3, isclose_atol=1e-07):
        assert self.is_grads_dbg_enabled
        assert all(tensor.grad.size() == tensor.size() for tensor in self._dbg__tensors_to_check_grads.values())
        assert all(tensor.grad.size(0) == next(iter(self._dbg__tensors_to_check_grads.values())).grad.size(0)
                   for tensor in self._dbg__tensors_to_check_grads.values())
        for name, tensor in self._dbg__tensors_to_check_grads.items():
            print(f'Checking tensor `{name}` of shape {tensor.size()}:')
            # print(tensor.grad.cpu())
            grad = tensor.grad.cpu()
            batch_size = grad.size(0)
            if not all(grad[example_idx].allclose(torch.tensor(0.0), atol=isclose_atol) for example_idx in
                       range(batch_size) if example_idx != grad_test_example_idx):
                print(
                    f'>>>>>>>> FAIL: Not all examples != #{grad_test_example_idx} has zero grad for tensor `{name}` of shape {tensor.size()}:')
                print(grad)
                for example_idx in range(batch_size):
                    print(f'non-zero places in grad for example #{example_idx}:')
                    print(grad[example_idx].isclose(torch.tensor(0.0), atol=isclose_atol))
                    print(grad[example_idx][~grad[example_idx].isclose(torch.tensor(0.0), atol=isclose_atol)])
            else:
                print(
                    f'Success: All examples != #{grad_test_example_idx} has zero grad for tensor `{name}` of shape {tensor.size()}:')
            if grad[grad_test_example_idx].allclose(torch.tensor(0.0), atol=isclose_atol):
                print(
                    f'>>>>>>>> FAIL: Tensor `{name}` of shape {tensor.size()} for example #{grad_test_example_idx} is all zero:')
                print(grad)
            else:
                print(
                    f'Success: Tensor `{name}` of shape {tensor.size()} for example #{grad_test_example_idx} is not all zero.')
                if tensor.ndim > 1:
                    sub_objects_iszero = torch.tensor(
                        [grad[grad_test_example_idx, sub_object_idx].allclose(torch.tensor(0.0), atol=isclose_atol)
                         for sub_object_idx in range(tensor.size(1))])
                    print(f'sub_objects_nonzero: {~sub_objects_iszero} (sum: {(~sub_objects_iszero).sum()})')
            print()

    def dbg_retain_grads_before_bwd(self):
        assert self.is_grads_dbg_enabled
        for tensor in self._dbg__tensors_to_check_grads.values():
            tensor.retain_grad()

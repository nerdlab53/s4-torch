import torch 

def torch_scan(f, init, xs):

        """
        A simple implementation of scan in PyTorch.

        !Disclaimer : This is a very simple implementation and it may not work as efficiently as jax.lax.scan so please use this with a grain of salt.!
        
        Args:
        f: A function that takes two arguments (carry and x) and returns the next carry and a current output.
        init: Initial carry value.
        xs: A tensor or a sequence of tensors over which to apply the scan function.
        
        Returns:
        A tuple of (final_carry, outputs) where final_carry is the last carry and outputs is a list of outputs at each step.
        """
        carry = init
        outputs = []
        for x in xs:
            carry, output = f(carry, x)
            outputs.append(output)
        outputs = torch.stack(outputs)
        return carry, outputs
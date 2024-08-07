import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import time

cpu = torch.device("cpu")
gpu = torch.device("cuda")


def initialise_weights(module: torch.nn.Module, weight_dist: torch.nn.init) -> nn.Module:
    weights_and_biases = set(["Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "nn.ConvTranspose2d", "nn.ConvTranspose3d"]) # TODO: check if there is a better way to do this
    if module.__class__.__name__ in weights_and_biases:
        weight_dist(module.weight) # weight initialisation
        if module.bias is not None:
            weight_dist(module.bias) # bias initialisation 


@dataclass
class Params:
    def asdict(self):
        return asdict(self)


@dataclass
class ConvParams(Params):
    in_channels: int = 1
    out_channels: int = 1
    kernel_size: int = 3 # no impact
    stride: int = 3 # 
    padding: int = 0 
    dilation: int = 1
    groups: int = 1
    bias: str = True
    padding_mode: str = "zeros"
    device: torch.device = None
    dtype: torch.dtype = None
    
@dataclass
class MaxPoolParams(Params):
    kernel_size: int = 3 # no impact
    stride: int = 3 # 
    padding: int = 0 
    dilation: int = 1
    return_indices: bool = False
    ceil_mode: bool = False

@dataclass
class ConvTransposeParams(ConvParams):
    output_padding: int = 0

@dataclass
class AdaptiveAvgPoolParams(Params):
    output_size: Tuple[Optional[int], Optional[int], Optional[int]] = None

@dataclass
class FractionalMaxPoolParams(Params):
    output_size: Tuple[Optional[int], Optional[int], Optional[int]] = None
    kernel_size: int = 3

@dataclass
class AvgPoolParams(Params):
    kernel_size: int = 3
    stride: int = 3 
    padding: int = 0 
    ceil_mode: bool = False
    count_include_pad = True
    divisor_override = None

@dataclass
class MaxUnpoolParams(Params):
    kernel_size: int = 3
    stride: int = 3 
    padding: int = 0
    
def error(a: torch.Tensor, b: torch.Tensor, tolerances: List[float]) -> Union[float, torch.Tensor]:
    
    if not isinstance(a, torch.Tensor): a = torch.tensor(a)
    if not isinstance(b, torch.Tensor): b = torch.tensor(b)
    
    if a.shape != b.shape:
        raise ValueError(f"Tensor dimension mismatch. Tensors a and b must have the same dimensions, tensor a with dimension {a.shape} != tensor b with dimension {b.shape}")
    
    epsilon = 1e-8
    relative_difference = torch.abs( (a - b) / (a + epsilon) ) # use epsilon to prevent dicision by zero
    # std_dev_relative_difference = torch.std(input=relative_difference, correction=False, keepdim=False)
    average_relative_difference = torch.mean(relative_difference)
    # print(relative_difference, average_relative_difference)
    
    if tolerances is not None:
        percent_exceeding_tolerances = list([])
        for tolerance in tolerances:
            percent_exceeding_tolerances.append((relative_difference > tolerance).float().mean().item())
        return average_relative_difference, list(zip(tolerances, percent_exceeding_tolerances))
            
    return average_relative_difference, None
    

def benchmark(
    op: torch.nn.Module,
    weight_dist: torch.nn.init,
    data_dist: torch.nn.init,
    dimensions: Dict,
    hyperparameters: Params,
    device: torch.device,
    deterministic: bool,
    autograd: bool,
    dtype: torch.dtype,
    iterations: int,
    # pytorch and cuda version   
) -> Any:
    
    Conv = set(["Conv1d", "Conv2d", "Conv3d"])
    ConvTranspose = set(["ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"])
    Padding = set(["ReplicationPad1d", "ReplicationPad2d", "ReplicationPad3d", "ReflectionPad1d", "ReflectionPad2d","ReflectionPad3d"])
    
    if op.__name__ == "index_copy":
        
        results_df = pd.DataFrame(columns=['input_dimension', 'dtype', 'reduction_ratio', 'dim', 'average_relative_error', 'average_runtime', 'unique_outputs', 'unique_outputs_per_iteration'])  
        for input_dimension in dimensions["input_dimensions"]:
            for dtype in dimensions["dtype"]:
                for reduction_ratio in dimensions["reduction_ratio"]:
                    index = torch.randint(low=0, high=int(input_dimension[0]*reduction_ratio), size=(int(input_dimension[0]*reduction_ratio), )).to(torch.int64).to(device)
                    input = data_dist(torch.zeros(input_dimension)).to(dtype).to(device)
                    source = data_dist(torch.zeros(int(input_dimension[0]*reduction_ratio), input_dimension[1])).to(dtype).to(device)
                    for dim in dimensions["dim"]:
                        if dim >= len(input_dimension): break # this generalises to arbitrary dimensional input
                        print(f"[INFO] input_dimension:{input_dimension}\t dtype:{dtype}\t reduction_ratio:{reduction_ratio}\t dim:{dim}\t" )  
                        
                        try:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                            baseline_output = input.index_copy(source=source, dim=dim, index=index) 
                        except:
                            torch.use_deterministic_algorithms(mode=False)
                            baseline_output = input.index_copy(source=source, dim=dim, index=index) 
                            
                        unique_outputs = set({baseline_output})
                        # set torch environment variables to deterministic/non-deterministic 
                        if deterministic:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                        else:
                            torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                        
                        running_error, running_time = 0, 0
                        
                        for _ in range(iterations):
                            start = time.time()
                            output = input.index_copy(source=source, dim=dim, index=index) 
                            end = time.time()
                            current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                            running_error += current_error
                            running_time += end - start
                            if not any([torch.equal(output, unique_output) for unique_output in unique_outputs]):
                                unique_outputs.add(output)    
                        
                        # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                        average_relative_error = running_error / iterations
                        average_runtime = running_time / iterations
                        print(f"[INFO]... Average Relative Error: {average_relative_error}")
                        print(f"[INFO]... Average Runtime: {average_runtime}")
                        print(f"[INFO]... Unique Outputs for {iterations+1} iterations: {len(unique_outputs)}")
                        print(f"[INFO]... Unique Outputs / iterations: {len(unique_outputs)/(iterations+1)}")
                        
                        results_df = results_df._append({
                            'input_dimension': input_dimension,
                            'dtype': str(dtype),
                            'reduction_ratio': reduction_ratio,
                            'dim': dim,
                            'average_relative_error': average_relative_error.detach().numpy(),
                            'average_runtime': average_runtime,
                            'unique_outputs': len(unique_outputs),
                            'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1), 
                        }, ignore_index=True)
                        del unique_outputs
                                    
                    del index
                    del input
                    torch.cuda.empty_cache()
        
        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv")
        
        
    if op.__name__ == "put":
            
        results_df = pd.DataFrame(columns=['input_dimension', 'dtype', 'reduction_ratio', 'accumulate', 'average_relative_error', 'average_runtime', 'unique_outputs', 'unique_outputs_per_iteration'])
        for input_dimension in dimensions["input_dimensions"]:
            for dtype in dimensions["dtype"]:
                for reduction_ratio in dimensions["reduction_ratio"]:
                    input = data_dist(torch.zeros(input_dimension)).to(dtype).to(device) 
                    index = torch.stack([
                        torch.randint(low=0, high=int(input_dimension[0]*reduction_ratio), size=(1, input_dimension[0])).to(torch.int64),
                        # torch.randint(low=0, high=int(input_dimension[0]*reduction_ratio), size=(1, input_dimension[0])).to(torch.int64),
                    ]).to(device)
                    source = data_dist(torch.zeros(int(input_dimension[0]))).to(dtype).to(device)
                    for accumulate in dimensions["accumulate"]:
                        print(f"[INFO] input_dimension:{input_dimension}\t dtype:{dtype}\t reduction_ratio:{reduction_ratio}\t, accumulate:{accumulate}\t")  
                        
                        try:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                            baseline_output = input.put(index=index, source=source, accumulate=accumulate) 
                        except:
                            torch.use_deterministic_algorithms(mode=False)
                            baseline_output = input.put(index=index, source=source, accumulate=accumulate) 
                            
                        unique_outputs = set({baseline_output})
                        # set torch environment variables to deterministic/non-deterministic 
                        if deterministic:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                        else:
                            torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                        
                        running_error, running_time = 0, 0
                        
                        for _ in range(iterations):
                            start = time.time()
                            baseline_output = input.put(index=index, source=source, accumulate=accumulate)
                            end = time.time()
                            current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                            running_error += current_error
                            running_time += end - start
                            if not any([torch.equal(output, unique_output) for unique_output in unique_outputs]):
                                unique_outputs.add(output)    
                        
                        # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                        average_relative_error = running_error / iterations
                        average_runtime = running_time / iterations
                        print(f"[INFO]... Average Relative Error: {average_relative_error}")
                        print(f"[INFO]... Average Runtime: {average_runtime}")
                        print(f"[INFO]... Unique Outputs for {iterations+1} iterations: {len(unique_outputs)}")
                        print(f"[INFO]... Unique Outputs / iterations: {len(unique_outputs)/(iterations+1)}")
                        
                        results_df = results_df._append({
                            'input_dimension': input_dimension,
                            'dtype': str(dtype),
                            'reduction_ratio': reduction_ratio,
                            'accumulate': accumulate,
                            'average_relative_error': average_relative_error.detach().numpy(),
                            'average_runtime': average_runtime,
                            'unique_outputs': len(unique_outputs),
                            'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                        }, ignore_index=True)
                        del unique_outputs
                                    
                    del indices
                    del input
                    del values
                    torch.cuda.empty_cache() 

        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv")
      
                   
    if op.__name__ in Padding: # TODO: check
        
        results_df = pd.DataFrame(columns=['batch_size', 'dim' 'dtype', 'padding',  'average_relative_error', 'average_runtime', 'unique_outputs', 'unique_outputs_per_iteration'])
        for dtype in dimensions["dtype"]:
            for batch in dimensions["batch_size"]: # loop over hyperparameters before looping over dimensions, as you need to instantiate the module with hyperparameters first
                for dim in dimensions["dim"]:
                    for pad in dimensions["pad"]:  
                        # instantiate module with the specified hyperparameter
                        module = op(pad) # initialise module with **kwargs by unpacking dict generated from Params dataclass
                        module.to(dtype) # set dtype
                        module.to(device) # send model to device
                        
                        print(f"[INFO] batch_size: {batch}\t dim:{dim}\t  padding:{pad}\t dtype:{dtype}\t")
                        torch.manual_seed(42) # TODO: check if this seeds the weight init schemes
                        input = data_dist(torch.zeros((batch, *dim))).to(dtype).to(device) # TODO: can we just initialise the data without using torch.zeros as an init buffer
                        input.requires_grad = True
                        # generate a deterministic baseline, if possible
                        try:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                            output = module(input) # initial baseline output
                            grad = torch.ones(output.shape).to(dtype).to(device)
                            output.backward(grad)
                            baseline_output = input.grad
                        except:
                            torch.use_deterministic_algorithms(mode=False)
                            output = module(input) # initial baseline output
                            grad = torch.ones(output.shape).to(dtype).to(device)
                            output.backward(grad)
                            baseline_output = input.grad
                            
                        unique_outputs = set({baseline_output}) 
                        # set torch environment variables to deterministic/non-deterministic 
                        if deterministic:
                            torch.backends.cudnn.benchmark = False # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                        else:
                            torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                        
                        running_error, running_time = 0, 0
                         
                        for _ in range(iterations):
                            torch.use_deterministic_algorithms(mode=False)
                            output = module(input) # initial baseline output
                            grad = torch.ones(output.shape).to(dtype).to(device)
                            start = time.time()
                            output.backward(grad)
                            end = time.time()
                            output = input.grad
                            # print(output)
                            current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                            running_error += current_error
                            running_time += end - start
                            if not any([torch.equal(output, unique_output) for unique_output in unique_outputs]):
                                unique_outputs.add(output)   
                        # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                        average_relative_error = running_error / iterations
                        average_runtime = running_time / iterations
                        print(f"[INFO]... Average Relative Error: {average_relative_error}")
                        print(f"[INFO]... Average Runtime: {average_runtime}")
                        
                        results_df = results_df._append({
                            'batch_size': batch,
                            'dim': dim,
                            'dtype': str(dtype),
                            'padding': pad,
                            'average_relative_error': average_relative_error.detach().numpy(),
                            'average_runtime': average_runtime,
                            'unique_outputs': len(unique_outputs),
                            'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                        }, ignore_index=True)
                        
                        # memory management to prevent OOM issues
                        del module
                        del input
                        torch.cuda.empty_cache()

        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv")

    if op.__name__ == "MaxUnpool1d" or op.__name__ == "MaxUnpool2d" or op.__name__ == "MaxUnpool3d":
        results_df = pd.DataFrame(columns=['batch_size', 'dim', 'dtype', 'stride', 'padding', 'kernel_size', 'average_relative_error', 'average_runtime', 'unique_outputs', 'unique_outputs_per_iteration'])
        if not isinstance(hyperparameters, MaxUnpoolParams):
            return TypeError(f"Op is a MaxUnpool, hyperparameter MaxUnpoolParams dataclass expected but the specified hyperparameter dataclass is of type: {type(hyperparameters)}")
       
        for dtype in dimensions["dtype"]:
            for batch in dimensions["batch_size"]:
                for dim in dimensions["dim"]:
                    for kernel_size in dimensions["kernel_size"]:
                        for stride in dimensions["stride"]:
                            for padding in dimensions["padding"]:
                                
                                if op.__name__ == "MaxUnpool1d": 
                                    module = torch.nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, return_indices=True, ceil_mode=False).to(device)
                                elif op.__name__ == "MaxUnpool2d":
                                    module = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, return_indices=True, ceil_mode=False).to(device)
                                else:
                                    module = torch.nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, return_indices=True, ceil_mode=False).to(device)
                                input, indices = module(data_dist(torch.zeros((batch, *dim))).to(dtype).to(device))
                                input, indices = input.to(device), indices.to(device)
                                 
                                # instantiate module with the specified hyperparameters
                                hyperparameters.padding = padding
                                hyperparameters.stride = stride
                                hyperparameters.kernel_size = kernel_size
                                module = op(**asdict(hyperparameters)) # initialise module with **kwargs by unpacking dict generated from Params dataclass
                                initialise_weights(module, weight_dist) # initialise weights with specified weight init scheme from torch.nn.init
                                module.to(dtype) # set dtype
                                module.to(device) # send model to device
                                        
                                print(f"[INFO] batch_size: {batch}\t dim:{dim}\t dtype:{dtype}\t stride:{stride}\t padding:{padding} kernel_size:{kernel_size}")
                                torch.manual_seed(42) # TODO: check if this seeds the weight init schemes
                                # input = data_dist(torch.zeros((batch, *dim))).to(dtype).to(device) # TODO: can we just initialise the data without using torch.zeros as an init buffer
                                try:
                                    torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                                    baseline_output = module(input, indices)
                                except:
                                    torch.use_deterministic_algorithms(mode=False)
                                    baseline_output = module(input, indices)
                                    
                                unique_outputs = set({baseline_output})
                                # set torch environment variables to deterministic/non-deterministic 
                                if deterministic:
                                    torch.backends.cudnn.benchmark = False # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
                                    torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                                else:
                                    torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                                
                                running_error, running_time = 0, 0
                                
                                for _ in range(iterations):
                                    torch.use_deterministic_algorithms(mode=False)
                                    start = time.time()
                                    output = module(input, indices) 
                                    end = time.time()
                                    current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                                    # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                                    running_error += current_error
                                    running_time += end - start
                                    if not any([torch.equal(output, unique_output) for unique_output in unique_outputs]):
                                        unique_outputs.add(output)   
                                
                                # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                                average_relative_error = running_error / iterations
                                average_runtime = running_time / iterations
                                print(f"[INFO]... Average Relative Error: {average_relative_error}")
                                print(f"[INFO]... Average Runtime: {average_runtime}")
                                
                                results_df = results_df._append({
                                    'batch_size': batch,
                                    'dim': dim,
                                    'dtype': str(dtype),
                                    'stride': stride,
                                    'padding': padding,
                                    'dim': dim,
                                    'average_relative_error': average_relative_error.detach().numpy(),
                                    'average_runtime': average_runtime,  
                                    'unique_outputs': len(unique_outputs),
                                    'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                                }, ignore_index=True)                                
                                
                                # memory management to prevent OOM issues
                                del module
                                del input
                                torch.cuda.empty_cache()
        
        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv") 
    
    if op.__name__ == "FractionalMaxPool2d" or op.__name__ == "FractionalMaxPool3d": # TODO: check
            
        results_df = pd.DataFrame(columns=['batch_size', 'dim', 'dtype', 'reduction_ratio', 'kernel_size', 'average_relative_error', 'average_runtime', 'unique_outputs', 'unique_outputs_per_iteration'])
        if not isinstance(hyperparameters, FractionalMaxPoolParams):
            return TypeError(f"Op is a FractionalMaxPool, hyperparameter AdaptiveAvgPoolParams dataclass expected but the specified hyperparameter dataclass is of type: {type(hyperparameters)}")
    
        for dtype in dimensions["dtype"]:
            for batch in dimensions["batch_size"]:
                for dim in dimensions["dim"]:
                    for kernel_size in dimensions["kernel_size"]:
                        for reduction_ratio in dimensions["reduction_ratio"]:
                            if op.__name__ == "FractionalMaxPool2d":
                                output_size = tuple((int(dim[-2] * reduction_ratio), int(dim[-1] * reduction_ratio)))
                            else:
                               output_size = tuple((int(dim[-3] * reduction_ratio), int(dim[-2] * reduction_ratio), int(dim[-1] * reduction_ratio))) 
                            # instantiate module with the specified hyperparameters
                            hyperparameters.output_size = output_size
                            hyperparameters.kernel_size = kernel_size
                            module = op(**asdict(hyperparameters)) # initialise module with **kwargs by unpacking dict generated from Params dataclass
                            initialise_weights(module, weight_dist) # initialise weights with specified weight init scheme from torch.nn.init
                            module.to(dtype) # set dtype
                            module.to(device) # send model to device
                                    
                            print(f"[INFO] batch_size: {batch}\t dim:{dim}\t dtype:{dtype}\t reduction_ratio:{reduction_ratio}\t output_size:{output_size} kernel_size:{kernel_size}")
                            torch.manual_seed(42) # TODO: check if this seeds the weight init schemes
                            input = data_dist(torch.zeros((batch, *dim))).to(dtype).to(device) # TODO: can we just initialise the data without using torch.zeros as an init buffer
                            input.requires_grad = True
                            # generate a deterministic baseline, if possible
                            try:
                                torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                                output = module(input) # initial baseline output
                                grad = torch.ones(output.shape).to(dtype).to(device)
                                output.backward(grad)
                                baseline_output = input.grad
                            except:
                                torch.use_deterministic_algorithms(mode=False)
                                output = module(input) # initial baseline output
                                grad = torch.ones(output.shape).to(dtype).to(device)
                                output.backward(grad)
                                baseline_output = input.grad 
                            
                            unique_outputs = set({baseline_output})
                            # set torch environment variables to deterministic/non-deterministic 
                            if deterministic:
                                torch.backends.cudnn.benchmark = False # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
                                torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                            else:
                                torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                            
                            running_error, running_time = 0, 0
                            
                            for _ in range(iterations):
                                torch.use_deterministic_algorithms(mode=False)
                                output = module(input) # initial baseline output
                                grad = torch.ones(output.shape).to(dtype).to(device)
                                start = time.time()
                                output.backward(grad)
                                end = time.time()
                                output = input.grad
                                # print(output)
                                current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                                # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                                running_error += current_error
                                running_time += end - start
                                if not any([torch.equal(output, unique_output) for unique_output in unique_outputs]):
                                    unique_outputs.add(output)   
                            
                            # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                            average_relative_error = running_error / iterations
                            average_runtime = running_time / iterations
                            print(f"[INFO]... Average Relative Error: {average_relative_error}")
                            print(f"[INFO]... Average Runtime: {average_runtime}")
                            
                            # Append results to the DataFrame
                            results_df = results_df._append({
                                'batch_size': batch,
                                'dim': dim,
                                'dtype': str(dtype),
                                'reduction_ratio': reduction_ratio,
                                'kernel_size': kernel_size,
                                'average_relative_error': average_relative_error.detach().numpy(),
                                'average_runtime': average_runtime,
                                'unique_outputs': len(unique_outputs),
                                'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                            }, ignore_index=True)
                            
                            # memory management to prevent OOM issues
                            del module
                            del input
                            torch.cuda.empty_cache()

        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv") 
 
    if op.__name__ in Conv: # TODO: check
        
        results_df = pd.DataFrame(columns=['batch_size', 'dim', 'dtype', 'kernel_size', 'stride', 'padding', 'dilation', 'group', 'average_relative_error', 'average_runtime', 'unique_outputs', 'unique_outputs_per_iteration'])

        if not isinstance(hyperparameters, ConvParams):
            return TypeError(f"Op is a Convolution, hyperparameter ConvParams dataclass expected but the specified hyperparameter dataclass is of type: {type(hyperparameters)}")
       
        for dtype in dimensions["dtype"]:
            for kernel_size in dimensions["kernel_size"]: # loop over hyperparameters before looping over dimensions, as you need to instantiate the module with hyperparameters first
                for stride in dimensions["stride"]:
                    for padding in dimensions["padding"]:
                        for dilation in dimensions["dilation"]:
                            for group in dimensions["groups"]: 
                                
                                # instantiate module with the specified hyperparameters
                                hyperparameters.kernel_size = kernel_size
                                hyperparameters.stride = stride
                                hyperparameters.padding = padding
                                hyperparameters.dilation = dilation
                                hyperparameters.group = group
                                hyperparameters.dtype = dtype
                                module = op(**asdict(hyperparameters)) # initialise module with **kwargs by unpacking dict generated from Params dataclass
                                initialise_weights(module, weight_dist) # initialise weights with specified weight init scheme from torch.nn.init
                                module.to(dtype) # set dtype
                                module.to(device) # send model to device
                                
                                for batch in dimensions["batch_size"]:
                                    for dim in dimensions["dim"]:
                                        print(f"[INFO] batch_size: {batch}\t dim:{dim}\t kernel_size:{kernel_size}\t stride:{stride}\t padding:{padding}\t dilation:{dilation}\t group:{group}\t dtype:{dtype}\t")
                                        torch.manual_seed(42) # TODO: check if this seeds the weight init schemes
                                        input = data_dist(torch.zeros((batch, *dim))).to(dtype).to(device) # TODO: can we just initialise the data without using torch.zeros as an init buffer
                                        
                                        # generate a deterministic baseline, if possible
                                        try:
                                            torch.backends.cudnn.benchmark = False # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
                                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                                            baseline_output = module(input) # initial baseline output
                                        except:
                                            torch.use_deterministic_algorithms(mode=False)
                                            baseline_output = module(input)
                                            # print(baseline_output)
                                        unique_outputs = set({baseline_output})
                                        
                                        # set torch environment variables to deterministic/non-deterministic 
                                        if deterministic:
                                            torch.backends.cudnn.benchmark = False # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
                                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                                        else:
                                            torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                                        
                                        running_error, running_time = 0, 0
                                        
                                        for _ in range(iterations):
                                            start = time.time()
                                            output = module(input)
                                            end = time.time()
                                            # print(output)
                                            current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                                            running_error += current_error
                                            running_time += end - start
                                            # print(torch.norm(output))
                                            if not any([torch.equal(output, unique_output) for unique_output in unique_outputs]):
                                                unique_outputs.add(output)  
                                        
                                        # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                                        average_relative_error = running_error / iterations
                                        average_runtime = running_time / iterations
                                        print(f"[INFO]... Average Relative Error: {average_relative_error}")
                                        print(f"[INFO]... Average Runtime: {average_runtime}")
                                        
                                        print(f"[INFO] batch_size: {batch}\t dim:{dim}\t kernel_size:{kernel_size}\t stride:{stride}\t padding:{padding}\t dilation:{dilation}\t group:{group}\t dtype:{dtype}\t")
                                                                               
                                        results_df = results_df._append({
                                            'batch_size': batch, 
                                            'dim': dim,
                                            'dtype': str(dtype),
                                            'kernel_size': kernel_size,
                                            'stride': stride,
                                            'padding': padding,
                                            "dilation": dilation,
                                            "group": group,
                                            'average_relative_error': average_relative_error.detach().numpy(),
                                            'average_runtime': average_runtime,
                                            'unique_outputs': len(unique_outputs),
                                            'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                                        }, ignore_index=True) 
                                        
                                        # del output
                                        # del baseline_output
                                    
                                    del input  
                                # memory management to prevent OOM issues
                                del module
                                torch.cuda.empty_cache()
                                
        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv")                                 
 
    if op.__name__ in ConvTranspose: # TODO: check
        
        results_df = pd.DataFrame(columns=['batch_size', 'dim', 'dtype', 'kernel_size', 'stride', 'padding', 'dilation', 'group', 'reduction_ratio', "average_relative_error", "std_dev_relative_error", "average_latency", "std_dev_latency", "average_ratio_non_equal", "std_dev_ratio_non_equal", 'unique_outputs', 'unique_outputs_per_iteration'])

        if not isinstance(hyperparameters, ConvTransposeParams):
            return TypeError(f"Op is a Convolution Transpose, hyperparameter ConvTransposeParams dataclass expected but the specified hyperparameter dataclass is of type: {type(hyperparameters)}")
       
        for dtype in dimensions["dtype"]:
            for kernel_size in dimensions["kernel_size"]: # loop over hyperparameters before looping over dimensions, as you need to instantiate the module with hyperparameters first
                for stride in dimensions["stride"]:
                    for padding in dimensions["padding"]:
                        for output_padding in dimensions["output_padding"]:
                            for dilation in dimensions["dilation"]:
                                for group in dimensions["groups"]: 
                                
                                    # instantiate module with the specified hyperparameters
                                    hyperparameters.kernel_size = kernel_size
                                    hyperparameters.stride = stride
                                    hyperparameters.padding = padding
                                    hyperparameters.dilation = dilation
                                    hyperparameters.group = group
                                    hyperparameters.dtype = dtype
                                    module = op(**asdict(hyperparameters)) # initialise module with **kwargs by unpacking dict generated from Params dataclass
                                    initialise_weights(module, weight_dist) # initialise weights with specified weight init scheme from torch.nn.init
                                    module.to(dtype) # set dtype
                                    module.to(device) # send model to device
                                    
                                    for batch in dimensions["batch_size"]:
                                        for dim in dimensions["dim"]:
                                            print(f"[INFO] batch_size: {batch}\t dim:{dim}\t kernel_size:{kernel_size}\t stride:{stride}\t padding:{padding}\t dilation:{dilation}\t output_padding: {output_padding}\t group:{group}\t dtype:{dtype}\t")
                                            torch.manual_seed(42) # TODO: check if this seeds the weight init schemes
                                            input = data_dist(torch.zeros((batch, *dim))).to(dtype).to(device) # TODO: can we just initialise the data without using torch.zeros as an init buffer
                                            
                                            # generate a deterministic baseline, if possible
                                            try:
                                                torch.backends.cudnn.benchmark = False # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
                                                torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                                                baseline_output = module(input) # initial baseline output
                                            except:
                                                torch.use_deterministic_algorithms(mode=False)
                                                baseline_output = module(input)
                                                # print(baseline_output)
                                            unique_outputs = set({baseline_output})
                                            errors = list([])
                                            latencies = list([])
                                            ratio_non_equal = list([])
                                            
                                            # set torch environment variables to deterministic/non-deterministic 
                                            if deterministic:
                                                torch.backends.cudnn.benchmark = False # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
                                                torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                                            else:
                                                torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                                            
                                            running_error, running_time = 0, 0
                                            
                                            for _ in range(iterations):
                                                start = time.time()
                                                output = module(input)
                                                end = time.time()
                                                # print(output)
                                                current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                                                # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                                                running_error += current_error
                                                running_time += end - start
                                                # print(torch.norm(output))
                                                if not any([torch.equal(output, unique_output) for unique_output in unique_outputs]):
                                                    unique_outputs.add(output)
                                                   
                                                errors.append([current_error.to(cpu)])
                                                latencies.append([end - start])
                                                num_non_equal_elements = torch.sum(baseline_output != output)
                                                total_elements = baseline_output.numel()
                                                ratio_non_equal_ = num_non_equal_elements / total_elements
                                                ratio_non_equal.append([ratio_non_equal_.to(cpu)]) 
                                                                     
                                                                                  
                                            errors = torch.tensor(errors)
                                            latencies = torch.tensor(latencies)
                                            ratio_non_equal = torch.tensor(ratio_non_equal)                     
                                            # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                                            average_relative_error = running_error / iterations
                                            average_runtime = running_time / iterations
                                            print(f"[INFO]... Average Relative Error: {average_relative_error}")
                                            print(f"[INFO]... Average Runtime: {average_runtime}")
                                            results_df = results_df._append({
                                                'batch_size': batch, 
                                                'dim': dim,
                                                'dtype': str(dtype),
                                                'kernel_size': kernel_size,
                                                'stride': stride,
                                                'padding': padding,
                                                "dilation": dilation,
                                                "group": group,
                                                # 'average_relative_error': average_relative_error.detach().numpy(),
                                                # 'average_runtime': average_runtime,
                                                'average_relative_error': torch.mean(errors).item(),
                                                'std_dev_relative_error': torch.std(errors).item(),
                                                'average_latency': torch.mean(latencies).item(),
                                                'std_dev_latency': torch.std(latencies).item(),
                                                'ratio_non_equal': ratio_non_equal_.to(cpu),
                                                'average_ratio_non_equal': torch.mean(ratio_non_equal),
                                                "std_dev_ratio_non_equal": torch.std(ratio_non_equal),
                                                'unique_outputs': len(unique_outputs),
                                                'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                                            }, ignore_index=True)
                                    # memory management to prevent OOM issues
                                    del module
                                    del input
                                    torch.cuda.empty_cache()
                                    
        results_df.to_csv(f"data/{op.__name__}_{deterministic}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}_{deterministic}.csv")
        
    if op.__name__ == "scatter_reduce":
            
        results_df = pd.DataFrame(columns=['input_dimensions', 'reduction_ratio', 'dtype', 'dim', 'reduce', 'include_self', "average_relative_error", "std_dev_relative_error", "average_latency", "std_dev_latency", "average_ratio_non_equal", "std_dev_ratio_non_equal", "unique_outputs", 'unique_outputs_per_iteration'])
        for input_dimension in dimensions["input_dimensions"]:
            for dtype in dimensions["dtype"]:
                for reduction_ratio in dimensions["reduction_ratio"]:
                    src = data_dist(torch.zeros(input_dimension)).to(dtype).to(device)
                    reduced_dim = tuple([int(input_dimension[i]*reduction_ratio) for i in range(len(input_dimension))]) # TODO: should we also use different reduction sizes per dimension?
                    index = torch.randint(low=0, high=reduced_dim[0], size=reduced_dim).to(torch.int64).to(device)
                    input = data_dist(torch.zeros(reduced_dim)).to(dtype).to(device)
                    for dim in dimensions["dim"]:
                        if dim >= len(input_dimension): break # this generalises to arbitrary dimensional input
                        for reduce in dimensions["reduce"]:
                            for include_self in dimensions["include_self"]:
                                print(f"[INFO] input_dimension:{input_dimension}\t dtype:{dtype}\t reduction_ratio:{reduction_ratio}\t dim:{dim}\t reduce:{reduce}\t include_self:{include_self}\t")
                                
                                try:
                                    torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                                    baseline_output = input.scatter_reduce(
                                        dim=dim,
                                        index=index,
                                        src=src,
                                        reduce=reduce,
                                        include_self=include_self,
                                    ) 
                                except:
                                    torch.use_deterministic_algorithms(mode=False)
                                    baseline_output = input.scatter_reduce(
                                        dim=dim,
                                        index=index,
                                        src=src,
                                        reduce=reduce,
                                        include_self=include_self,
                                    )
                                
                                unique_outputs = set({baseline_output})
                                errors = list([])
                                latencies = list([])
                                ratio_non_equal = list([])
                                # set torch environment variables to deterministic/non-deterministic 
                                if deterministic:
                                    torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                                else:
                                    torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                                
                                # running_error, running_time = 0, 0
                                
                                for _ in range(iterations):
                                    start = time.time()
                                    output = input.scatter_reduce(
                                        dim=dim,
                                        index=index,
                                        src=src,
                                        reduce=reduce,
                                        include_self=include_self,
                                    )  
                                    end = time.time()
                                    current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                                    errors.append([current_error.to(cpu)])
                                    # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                                    # running_error += current_error
                                    latencies.append([end - start])
                                    if not any([torch.equal(output, unique_output) for unique_output in unique_outputs]):
                                        unique_outputs.add(output)
                                        
                                    num_non_equal_elements = torch.sum(baseline_output != output)
                                    total_elements = baseline_output.numel()
                                    ratio_non_equal_ = num_non_equal_elements / total_elements
                                    ratio_non_equal.append([ratio_non_equal_.to(cpu)])
                                      
                                errors = torch.tensor(errors)
                                latencies = torch.tensor(latencies)
                                ratio_non_equal = torch.tensor(ratio_non_equal)
                                # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                                # average_relative_error = running_error / iterations
                                # average_runtime = running_time / iterations
                                print(f"[INFO]... Average and StdDev Relative Error: {torch.mean(errors).item(), torch.std(errors).item()}")
                                print(f"[INFO]... Average and StdDev Runtime: {torch.mean(latencies).item(), torch.std(latencies).item()}")
                                print(f"[INFO]... Unique Outputs for {iterations+1} iterations: {len(unique_outputs)}")
                                print(f"[INFO]... Unique Outputs / iterations: {len(unique_outputs)/(iterations+1)}")
                                print(f"[INFO]... Unique elementwise outputs / iterations: {ratio_non_equal_}")
                                
                                print(f"[INFO] input_dimension:{input_dimension}\t dtype:{dtype}\t reduction_ratio:{reduction_ratio}\t dim:{dim}\t reduce:{reduce}\t include_self:{include_self}\t")
                                results_df = results_df._append({
                                    'input_dimension': input_dimension, 
                                    'dim': dim,
                                    'dtype': str(dtype),
                                    'reduction_ratio': reduction_ratio,
                                    'reduce': reduce,
                                    'include_self': include_self,
                                    'average_relative_error': torch.mean(errors).item(),
                                    'std_dev_relative_error': torch.std(errors).item(),
                                    'average_latency': torch.mean(latencies).item(),
                                    'std_dev_latency': torch.std(latencies).item(),
                                    'ratio_non_equal': ratio_non_equal_.to(cpu),
                                    'average_ratio_non_equal': torch.mean(ratio_non_equal),
                                    "std_dev_ratio_non_equal": torch.std(ratio_non_equal),
                                    # 'average_runtime': average_runtime,
                                    'unique_outputs': len(unique_outputs),
                                    'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                                }, ignore_index=True)
                                
                                del unique_outputs 
                                    
                    del src
                    del index
                    del input
                    torch.cuda.empty_cache()
                    
        results_df.to_csv(f"data/{op.__name__}_{deterministic}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}_{deterministic}.csv")

    if op.__name__ == "scatter":
            
        results_df = pd.DataFrame(columns=['input_dimensions', 'reduction_ratio', 'dtype', 'dim', 'reduce', 'unique_outputs', 'unique_outputs_per_iteration'])
        for input_dimension in dimensions["input_dimensions"]:
            for dtype in dimensions["dtype"]:
                for reduction_ratio in dimensions["reduction_ratio"]:
                    src = data_dist(torch.zeros(input_dimension)).to(dtype).to(device)
                    reduced_dim = tuple([int(input_dimension[i]*reduction_ratio) for i in range(len(input_dimension))]) # TODO: should we also use different reduction sizes per dimension?
                    index = torch.randint(low=0, high=reduced_dim[0], size=reduced_dim).to(torch.int64).to(device)
                    input = data_dist(torch.zeros(reduced_dim)).to(dtype).to(device)
                    for dim in dimensions["dim"]:
                        if dim >= len(input_dimension): break # this generalises to arbitrary dimensional input
                        for reduce in dimensions["reduce"]:
                            print(f"[INFO] input_dimension:{input_dimension}\t dtype:{dtype}\t reduction_ratio:{reduction_ratio}\t dim:{dim}\t reduce:{reduce}\t")
                            
                            try:
                                torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                                baseline_output = input.scatter(
                                    dim=dim,
                                    index=index,
                                    src=src,
                                    reduce=reduce,
                                ) 
                            except:
                                torch.use_deterministic_algorithms(mode=False)
                                baseline_output = input.scatter(
                                    dim=dim,
                                    index=index,
                                    src=src,
                                    reduce=reduce,
                                )
                            
                            unique_outputs = set({baseline_output})
                            # set torch environment variables to deterministic/non-deterministic 
                            if deterministic:
                                torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                            else:
                                torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                            
                            running_error, running_time = 0, 0
                            
                            for _ in range(iterations):
                                start = time.time()
                                output = input.scatter(
                                    dim=dim,
                                    index=index,
                                    src=src,
                                    reduce=reduce,
                                )  
                                end = time.time()
                                current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                                # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                                running_error += current_error
                                running_time += end - start
                                if not any([torch.equal(output, unique_output) for unique_output in unique_outputs]):
                                    unique_outputs.add(output)    
                            
                            # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                            average_relative_error = running_error / iterations
                            average_runtime = running_time / iterations
                            print(f"[INFO]... Average Relative Error: {average_relative_error}")
                            print(f"[INFO]... Average Runtime: {average_runtime}")
                            print(f"[INFO]... Unique Outputs for {iterations+1} iterations: {len(unique_outputs)}")
                            print(f"[INFO]... Unique Outputs / iterations: {len(unique_outputs)/(iterations+1)}")
                            results_df = results_df._append({
                                'input_dimension': input_dimension, 
                                'dim': dim,
                                'dtype': str(dtype),
                                'reduction_ratio': reduction_ratio,
                                'reduce': reduce,
                                'average_relative_error': average_relative_error.detach().numpy(),
                                'average_runtime': average_runtime,
                                'unique_outputs': len(unique_outputs),
                                'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                            }, ignore_index=True)
                                        
                            del unique_outputs
                      
                    del src
                    del index
                    del input
                    torch.cuda.empty_cache() 

        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv")

    if op.__name__ == "index_select":
        
        results_df = pd.DataFrame(columns=['input_dimensions', 'reduction_ratio', 'dtype', 'dim', 'unique_outputs', 'unique_outputs_per_iteration'])
        for input_dimension in dimensions["input_dimensions"]:
            for dtype in dimensions["dtype"]:
                for reduction_ratio in dimensions["reduction_ratio"]:
                    input = data_dist(torch.zeros(input_dimension)).to(dtype).to(device)
                    input.requires_grad = True
                    reduced_dim = tuple([int(input_dimension[i]*reduction_ratio) for i in range(len(input_dimension))]) # TODO: should we also use different reduction sizes per dimension?
                    index = torch.randint(low=0, high=input_dimension[0], size=tuple((reduced_dim[0],))).to(torch.int64).to(device)
                    grad = torch.ones(reduced_dim).to(dtype).to(device) if len(input_dimension) == 1 else torch.ones(reduced_dim[0], input_dimension[1]).to(dtype).to(device) 
                    for dim in dimensions["dim"]:
                        if dim >= len(input_dimension): break # this generalises to arbitrary dimensional input
                        print(f"[INFO] input_dimension:{input_dimension}\t dtype:{dtype}\t reduction_ratio:{reduction_ratio}\t dim:{dim}\t")
                        
                        try:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                            output = torch.index_select(
                                input=input,
                                dim=dim,
                                index=index,
                            )
                            grad = torch.ones(output.shape).to(dtype).to(device)
                            output.backward(grad)
                            baseline_output = input.grad 
                        except:
                            torch.use_deterministic_algorithms(mode=False)
                            output = torch.index_select(
                                input=input,
                                dim=dim,
                                index=index,
                            )
                            grad = torch.ones(output.shape).to(dtype).to(device)
                            output.backward(grad)
                            baseline_output = input.grad
                        
                        unique_outputs = set({baseline_output})
                        # set torch environment variables to deterministic/non-deterministic 
                        if deterministic:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                        else:
                            torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                        
                        running_error, running_time = 0, 0
                        
                        for _ in range(iterations):
                            output = torch.index_select(
                                input=input,
                                dim=dim,
                                index=index,
                            )
                            grad = torch.ones(output.shape).to(dtype).to(device)
                            start = time.time()
                            output.backward(grad)
                            end = time.time()
                            
                            del grad
                            
                            current_error, _ = error(input.grad.to(cpu), baseline_output.to(cpu), tolerances=None) 
                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                            running_error += current_error
                            running_time += end - start
                            if not any([torch.equal(input.grad, unique_output) for unique_output in unique_outputs]):
                                unique_outputs.add(output)    
                        
                        # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                        average_relative_error = running_error / iterations
                        average_runtime = running_time / iterations
                        print(f"[INFO]... Average Relative Error: {average_relative_error}")
                        print(f"[INFO]... Average Runtime: {average_runtime}")
                        print(f"[INFO]... Unique Outputs for {iterations+1} iterations: {len(unique_outputs)}")
                        print(f"[INFO]... Unique Outputs / iterations: {len(unique_outputs)/(iterations+1)}")
                        results_df = results_df._append({
                            'input_dimension': input_dimension, 
                            'dim': dim,
                            'dtype': str(dtype),
                            'reduction_ratio': reduction_ratio,
                            'average_relative_error': average_relative_error.detach().numpy(),
                            'average_runtime': average_runtime,
                            'unique_outputs': len(unique_outputs),
                            'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                        }, ignore_index=True)
                        
                        del unique_outputs
                
                                
                    del index
                    del input
                    torch.cuda.empty_cache() 
        
        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv")
        
    if op.__name__ == "bmm":
        
        results_df = pd.DataFrame(columns=['batch', 'n', 'm', 'p', 'unique_outputs', 'unique_outputs_per_iteration'])
        for dtype in dimensions["dtype"]:
            for batch in dimensions["batch"]:
                for n in dimensions["n"]:
                    for m in dimensions["m"]:
                        for p in dimensions["p"]:
                            input = data_dist(torch.zeros((batch, n, m))).to(dtype).to(device) 
                            mat2 = data_dist(torch.zeros((batch, m, p))).to(dtype).to(device) 
                            print(f"[INFO] batch:{batch}\t dtype:{dtype}\t n:{n}\t m:{m}\t p:{p}\t")
                        
                            try:
                                torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                                baseline_output = torch.bmm(input, mat2)
                            except:
                                torch.use_deterministic_algorithms(mode=False)
                                baseline_output = torch.bmm(input, mat2)
                            
                            unique_outputs = set({baseline_output})
                            # set torch environment variables to deterministic/non-deterministic 
                            if deterministic:
                                torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                            else:
                                torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                            
                            running_error, running_time = 0, 0
                            
                            for _ in range(iterations):
                                start = time.time()
                                output = torch.bmm(input, mat2)
                                end = time.time()
                                
                            current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                            running_error += current_error
                            running_time += end - start
                            if not any([torch.equal(output.to(cpu), unique_output.to(cpu)) for unique_output in unique_outputs]):
                                unique_outputs.add(output)    
                        
                            # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                            average_relative_error = running_error / iterations
                            average_runtime = running_time / iterations
                            print(f"[INFO]... Average Relative Error: {average_relative_error}")
                            print(f"[INFO]... Average Runtime: {average_runtime}")
                            print(f"[INFO]... Unique Outputs for {iterations+1} iterations: {len(unique_outputs)}")
                            print(f"[INFO]... Unique Outputs / iterations: {len(unique_outputs)/(iterations+1)}")
                            
                            results_df = results_df._append({
                                'batch': batch,
                                'n': n,
                                'm': m,
                                'p': p,
                                'average_relative_error': average_relative_error.detach().numpy(),
                                'average_runtime': average_runtime,
                                'unique_outputs': len(unique_outputs),
                                'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                            }, ignore_index=True)
                             
                            del unique_outputs
                            del input
                            del mat2
                            torch.cuda.empty_cache()

        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv")
 
    if op.__name__ == "gather":
            
        results_df = pd.DataFrame(columns=['input_dimensions', 'reduction_ratio', 'dtype', 'dim', 'unique_outputs', 'unique_outputs_per_iteration'])
        for input_dimension in dimensions["input_dimensions"]:
            for dtype in dimensions["dtype"]:
                for reduction_ratio in dimensions["reduction_ratio"]:
                    index = torch.randint(low=0, high=int(input_dimension[0]*reduction_ratio), size=input_dimension).to(torch.int64).to(device)
                    input = data_dist(torch.zeros(input_dimension)).to(dtype).to(device)
                    for dim in dimensions["dim"]:
                        if dim >= len(input_dimension): break # this generalises to arbitrary dimensional input
                        print(f"[INFO] input_dimension:{input_dimension}\t dtype:{dtype}\t reduction_ratio:{reduction_ratio}\t dim:{dim}\t" )  
                        
                        try:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                            baseline_output = torch.gather(input=input, dim=dim, index=index) 
                        except:
                            torch.use_deterministic_algorithms(mode=False)
                            baseline_output = torch.gather(input=input, dim=dim, index=index) 
                           
                        unique_outputs = set({baseline_output})
                        # set torch environment variables to deterministic/non-deterministic 
                        if deterministic:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                        else:
                            torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                        
                        running_error, running_time = 0, 0
                        
                        for _ in range(iterations):
                            start = time.time()
                            output = torch.gather(input=input, dim=dim, index=index) 
                            end = time.time()
                            current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                            running_error += current_error
                            running_time += end - start
                            if not any([torch.equal(output, unique_output) for unique_output in unique_outputs]):
                                unique_outputs.add(output)    
                        
                        # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                        average_relative_error = running_error / iterations
                        average_runtime = running_time / iterations
                        print(f"[INFO]... Average Relative Error: {average_relative_error}")
                        print(f"[INFO]... Average Runtime: {average_runtime}")
                        print(f"[INFO]... Unique Outputs for {iterations+1} iterations: {len(unique_outputs)}")
                        print(f"[INFO]... Unique Outputs / iterations: {len(unique_outputs)/(iterations+1)}")
                        results_df = results_df._append({ 
                            'input_dimension': input_dimension, 
                            'dim': dim,
                            'dtype': str(dtype),
                            'reduction_ratio': reduction_ratio,
                            'average_relative_error': average_relative_error.detach().numpy(),
                            'average_runtime': average_runtime,
                            'unique_outputs': len(unique_outputs),
                            'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                        }, ignore_index=True)
                        
                        del unique_outputs
                                   
                    del index
                    del input
                    torch.cuda.empty_cache()

        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv")
 
    if op.__name__ == "index_add":
            
        results_df = pd.DataFrame(columns=['input_dimensions', 'reduction_ratio', 'dtype', 'dim', "average_relative_error", "std_dev_relative_error", "average_latency", "std_dev_latency", "average_ratio_non_equal", "std_dev_ratio_non_equal", 'unique_outputs', 'unique_outputs_per_iteration'])
        for input_dimension in dimensions["input_dimensions"]:
            for dtype in dimensions["dtype"]:
                for reduction_ratio in dimensions["reduction_ratio"]:
                    index = torch.randint(low=0, high=int(input_dimension[0]*reduction_ratio), size=(int(input_dimension[0]*reduction_ratio), )).to(torch.int64).to(device)
                    input = data_dist(torch.zeros(input_dimension)).to(dtype).to(device)
                    source = data_dist(torch.zeros(int(input_dimension[0]*reduction_ratio), input_dimension[1])).to(dtype).to(device)
                    for dim in dimensions["dim"]:
                        if dim >= len(input_dimension): break # this generalises to arbitrary dimensional input
                        print(f"[INFO] input_dimension:{input_dimension}\t dtype:{dtype}\t reduction_ratio:{reduction_ratio}\t dim:{dim}\t" )  
                        
                        try:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                            baseline_output = input.index_add(source=source, dim=dim, index=index) 
                        except:
                            torch.use_deterministic_algorithms(mode=False)
                            baseline_output = input.index_add(source=source, dim=dim, index=index) 
                            
                        unique_outputs = set({baseline_output})
                        errors = list([])
                        latencies = list([])
                        ratio_non_equal = list([])
                        # set torch environment variables to deterministic/non-deterministic 
                        if deterministic:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                        else:
                            torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                        
                        running_error, running_time = 0, 0
                        
                        for _ in range(iterations):
                            start = time.time()
                            output = input.index_add(source=source, dim=dim, index=index) 
                            end = time.time()
                            current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                            running_error += current_error
                            running_time += end - start
                            if not any([torch.equal(output, unique_output) for unique_output in unique_outputs]):
                                unique_outputs.add(output)
                            errors.append([current_error.to(cpu)])
                            latencies.append([end - start])
                            num_non_equal_elements = torch.sum(baseline_output != output)
                            total_elements = baseline_output.numel()
                            ratio_non_equal_ = num_non_equal_elements / total_elements
                            ratio_non_equal.append([ratio_non_equal_.to(cpu)])
                                      
                        errors = torch.tensor(errors)
                        latencies = torch.tensor(latencies)
                        ratio_non_equal = torch.tensor(ratio_non_equal)
                             
                        # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                        average_relative_error = running_error / iterations
                        average_runtime = running_time / iterations
                        print(f"[INFO]... Average Relative Error: {average_relative_error}")
                        print(f"[INFO]... Average Runtime: {average_runtime}")
                        print(f"[INFO]... Unique Outputs for {iterations+1} iterations: {len(unique_outputs)}")
                        print(f"[INFO]... Unique Outputs / iterations: {len(unique_outputs)/(iterations+1)}")
                        results_df = results_df._append({ 
                            'input_dimension': input_dimension, 
                            'dim': dim,
                            'dtype': str(dtype),
                            'reduction_ratio': reduction_ratio,
                            'average_relative_error': average_relative_error.detach().numpy(),
                            'average_runtime': average_runtime,
                            'average_relative_error': torch.mean(errors).item(),
                            'std_dev_relative_error': torch.std(errors).item(),
                            'average_latency': torch.mean(latencies).item(),
                            'std_dev_latency': torch.std(latencies).item(),
                            'ratio_non_equal': ratio_non_equal_.to(cpu),
                            'average_ratio_non_equal': torch.mean(ratio_non_equal),
                            "std_dev_ratio_non_equal": torch.std(ratio_non_equal),
                            'unique_outputs': len(unique_outputs),
                            'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                        }, ignore_index=True)
                        
                        del unique_outputs
                                    
                    del index
                    del input
                    torch.cuda.empty_cache()

        results_df.to_csv(f"data/{op.__name__}_{deterministic}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}_{deterministic}.csv")
 
    if op.__name__ == "index_put":
            
        results_df = pd.DataFrame(columns=['input_dimensions', 'reduction_ratio', 'dtype', 'accumulate', 'unique_outputs', 'unique_outputs_per_iteration'])
        for input_dimension in dimensions["input_dimensions"]:
            for dtype in dimensions["dtype"]:
                for reduction_ratio in dimensions["reduction_ratio"]:
                    input = data_dist(torch.zeros(input_dimension)).to(dtype).to(device) 
                    indices = tuple((
                        torch.randint(low=0, high=int(input_dimension[0]*reduction_ratio), size=(1, input_dimension[0])).to(torch.int64).to(device),
                        torch.randint(low=0, high=int(input_dimension[0]*reduction_ratio), size=(1, input_dimension[0])).to(torch.int64).to(device),
                    ))
                    values = data_dist(torch.zeros(int(input_dimension[0]))).to(dtype).to(device)
                    for accumulate in dimensions["accumulate"]:
                        print(f"[INFO] input_dimension:{input_dimension}\t dtype:{dtype}\t reduction_ratio:{reduction_ratio}\t, accumulate:{accumulate}\t")  
                        
                        try:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                            baseline_output = input.index_put(indices=indices, values=values, accumulate=accumulate) 
                        except:
                            torch.use_deterministic_algorithms(mode=False)
                            baseline_output = input.index_put(indices=indices, values=values, accumulate=accumulate) 
                            
                        unique_outputs = set({baseline_output})
                        # set torch environment variables to deterministic/non-deterministic 
                        if deterministic:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                        else:
                            torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                        
                        running_error, running_time = 0, 0
                        
                        for _ in range(iterations):
                            start = time.time()
                            output = input.index_put(indices=indices, values=values, accumulate=accumulate) 
                            end = time.time()
                            current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                            running_error += current_error
                            running_time += end - start
                            if not any([torch.equal(output, unique_output) for unique_output in unique_outputs]):
                                unique_outputs.add(output)    
                        
                        # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                        average_relative_error = running_error / iterations
                        average_runtime = running_time / iterations
                        print(f"[INFO]... Average Relative Error: {average_relative_error}")
                        print(f"[INFO]... Average Runtime: {average_runtime}")
                        print(f"[INFO]... Unique Outputs for {iterations+1} iterations: {len(unique_outputs)}")
                        print(f"[INFO]... Unique Outputs / iterations: {len(unique_outputs)/(iterations+1)}")
                        
                        results_df = results_df._append({ 
                            'input_dimension': input_dimension, 
                            'dtype': str(dtype),
                            'accumulate': accumulate,
                            'reduction_ratio': reduction_ratio,
                            'average_relative_error': average_relative_error.detach().numpy(),
                            'average_runtime': average_runtime,                           
                            'unique_outputs': len(unique_outputs),
                            'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                        }, ignore_index=True)
                        
                        del unique_outputs
                                    
                    del indices
                    del input
                    del values
                    torch.cuda.empty_cache()
                    
        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv") 
                    
    if op.__name__ == "histc":
            
        results_df = pd.DataFrame(columns=['input_dimensions', 'reduction_ratio', 'dtype', 'min', 'max', 'unique_outputs', 'unique_outputs_per_iteration'])
        for input_dimension in dimensions["input_dimensions"]:
            for dtype in dimensions["dtype"]:
                for reduction_ratio in dimensions["reduction_ratio"]:
                    for min, max in zip(dimensions["min"], dimensions["max"]):
                        input = data_dist(torch.zeros(input_dimension)).to(dtype).to(device) 
                        bins = int(input_dimension*reduction_ratio) #.to(device)
                        min, max = torch.min(input).item() * (1 + min), torch.max(input).item() * (1 - max)
                        # min, max = min.to(device), max.to(device) 
                        print(f"[INFO] input_dimension:{input_dimension}\t dtype:{dtype}\t reduction_ratio:{reduction_ratio}\t, bins:{bins}\t, min:{min}\t, max:{max}\t")  
                        try:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                            baseline_output = torch.histc(input=input, bins=bins, min=min, max=max) 
                        except:
                            torch.use_deterministic_algorithms(mode=False)
                            baseline_output = torch.histc(input=input, bins=bins, min=min, max=max) 
                            
                        unique_outputs = set({baseline_output})
                        # set torch environment variables to deterministic/non-deterministic 
                        if deterministic:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                        else:
                            torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                        
                        running_error, running_time = 0, 0
                        
                        for _ in range(iterations):
                            start = time.time()
                            output = torch.histc(input=input, bins=bins, min=min, max=max) 
                            end = time.time()
                            current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                            running_error += current_error
                            running_time += end - start
                            if not any([torch.equal(output, unique_output) for unique_output in unique_outputs]):
                                unique_outputs.add(output)    
                        
                        # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                        average_relative_error = running_error / iterations
                        average_runtime = running_time / iterations
                        print(f"[INFO]... Average Relative Error: {average_relative_error}")
                        print(f"[INFO]... Average Runtime: {average_runtime}")
                        print(f"[INFO]... Unique Outputs for {iterations+1} iterations: {len(unique_outputs)}")
                        print(f"[INFO]... Unique Outputs / iterations: {len(unique_outputs)/(iterations+1)}")
                        
                        results_df = results_df._append({ 
                            'input_dimension': input_dimension, 
                            'dtype': str(dtype),
                            'min': min,
                            'max': max,
                            'reduction_ratio': reduction_ratio,
                            'average_relative_error': average_relative_error.detach().numpy(),
                            'average_runtime': average_runtime,
                            'unique_outputs': len(unique_outputs),
                            'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                        }, ignore_index=True)
                        del unique_outputs
                                    
                    del input
                    del bins
                    del min
                    del max
                    torch.cuda.empty_cache()
                    
        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv")  

    if op.__name__ == "bincount":
            
        results_df = pd.DataFrame(columns=['input_dimensions', 'reduction_ratio', 'dtype', 'minlength', 'unique_outputs', 'unique_outputs_per_iteration'])

        for input_dimension in dimensions["input_dimensions"]:
            for dtype in dimensions["dtype"]:
                for reduction_ratio in dimensions["reduction_ratio"]:
                    input = torch.randint(low=0, high=input_dimension, size=(input_dimension, )).to(dtype).to(device) 
                    weights = torch.randint(low=0, high=input_dimension, size=(input_dimension, )).to(dtype).to(device)
                    minlength = int(input_dimension*reduction_ratio)
                    print(f"[INFO] input_dimension:{input_dimension}\t dtype:{dtype}\t reduction_ratio:{reduction_ratio}\t, minlength:{minlength}\t")  
                    try:
                        torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                        baseline_output = torch.bincount(input=input, weights=weights, minlength=minlength) 
                    except:
                        torch.use_deterministic_algorithms(mode=False)
                        baseline_output = torch.bincount(input=input, weights=weights, minlength=minlength) 
                        
                    unique_outputs = set({baseline_output})
                    # set torch environment variables to deterministic/non-deterministic 
                    if deterministic:
                        torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                    else:
                        torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                    
                    running_error, running_time = 0, 0
                    
                    for _ in range(iterations):
                        start = time.time()
                        output = torch.bincount(input=input, weights=weights, minlength=minlength) 
                        end = time.time()
                        current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                        # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                        running_error += current_error
                        running_time += end - start
                        if not any([torch.equal(output, unique_output) for unique_output in unique_outputs]):
                            unique_outputs.add(output)    
                    
                    # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                    average_relative_error = running_error / iterations
                    average_runtime = running_time / iterations
                    print(f"[INFO]... Average Relative Error: {average_relative_error}")
                    print(f"[INFO]... Average Runtime: {average_runtime}")
                    print(f"[INFO]... Unique Outputs for {iterations+1} iterations: {len(unique_outputs)}")
                    print(f"[INFO]... Unique Outputs / iterations: {len(unique_outputs)/(iterations+1)}")
                    
                    results_df = results_df._append({ 
                        'input_dimension': input_dimension, 
                        'dtype': str(dtype),
                        'minlength': minlength,
                        'reduction_ratio': reduction_ratio,
                        'average_relative_error': average_relative_error.detach().numpy(),
                        'average_runtime': average_runtime,
                        'unique_outputs': len(unique_outputs),
                        'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                    }, ignore_index=True)
                        
                    del unique_outputs
                                    
                    del input
                    del weights
                    del minlength
                    torch.cuda.empty_cache()
                    
        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv")  

    if op.__name__ == "kthvalue":
            
        results_df = pd.DataFrame(columns=['input_dimensions', 'dim', 'dtype', 'k', 'unique_outputs', 'unique_outputs_per_iteration'])

        for input_dimension in dimensions["input_dimensions"]:
            for dtype in dimensions["dtype"]:
                input = data_dist(torch.zeros(input_dimension)).to(dtype).to(device)
                k = int(torch.randint(low=0, high=input_dimension[0], size=(1,)))
                for dim in dimensions["dim"]:
                    if dim >= len(input_dimension): break # this generalises to arbitrary dimensional input
                    for keepdim in dimensions["keepdim"]:
                        print(f"[INFO] input_dimension:{input_dimension}\t dtype:{dtype}\t k:{k}\t dim:{dim}\t")  
                        
                        try:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                            baseline_output = torch.kthvalue(input=input, k=k, dim=dim, keepdim=keepdim) 
                        except:
                            torch.use_deterministic_algorithms(mode=False)
                            baseline_output = torch.kthvalue(input=input, k=k, dim=dim, keepdim=keepdim) 
                            
                        unique_outputs = set({baseline_output})
                        # set torch environment variables to deterministic/non-deterministic 
                        if deterministic:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                        else:
                            torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                        
                        running_error, running_time = 0, 0
                        
                        for _ in range(iterations):
                            start = time.time()
                            output = torch.kthvalue(input=input, k=k, dim=dim, keepdim=keepdim) 
                            end = time.time()
                            current_error, _ = error(output.values, baseline_output.values, tolerances=None) 
                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                            running_error += current_error
                            running_time += end - start
                            if not any([torch.equal(output.values.to(cpu), unique_output.values.to(cpu)) and torch.equal(output.indices.to(cpu), unique_output.indices.to(cpu)) for unique_output in unique_outputs]):
                                unique_outputs.add(output)    
                        
                        # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                        average_relative_error = running_error / iterations
                        average_runtime = running_time / iterations
                        print(f"[INFO]... Average Relative Error: {average_relative_error}")
                        print(f"[INFO]... Average Runtime: {average_runtime}")
                        print(f"[INFO]... Unique Outputs for {iterations+1} iterations: {len(unique_outputs)}")
                        print(f"[INFO]... Unique Outputs / iterations: {len(unique_outputs)/(iterations+1)}")
                        results_df = results_df._append({ 
                            'input_dimension': input_dimension, 
                            'dtype': str(dtype),
                            'dim': dim,
                            'k': k,
                            'average_relative_error': average_relative_error.detach().cpu().numpy(),
                            'average_runtime': average_runtime,
                            'unique_outputs': len(unique_outputs),
                            'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                        }, ignore_index=True)
                         
                        del unique_outputs
                        
                                    
                del input
                torch.cuda.empty_cache()
                
        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv")  

    if op.__name__ == "median":
            
        results_df = pd.DataFrame(columns=['input_dimensions', 'dim', 'dtype', 'keepdim', 'unique_outputs', 'unique_outputs_per_iteration'])
        for input_dimension in dimensions["input_dimensions"]:
            for dtype in dimensions["dtype"]:
                input = data_dist(torch.zeros(input_dimension)).to(dtype).to(device)
                for dim in dimensions["dim"]:
                    if dim >= len(input_dimension): break # this generalises to arbitrary dimensional input
                    for keepdim in dimensions["keepdim"]:
                        print(f"[INFO] input_dimension:{input_dimension}\t dtype:{dtype}\t dim:{dim}\t keepdim:{keepdim}\t")  
                        
                        try:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                            baseline_output = torch.median(input=input, dim=dim, keepdim=keepdim) 
                        except:
                            torch.use_deterministic_algorithms(mode=False)
                            baseline_output = torch.median(input=input, dim=dim, keepdim=keepdim) 
                            
                        unique_outputs = set({baseline_output})
                        # set torch environment variables to deterministic/non-deterministic 
                        if deterministic:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                        else:
                            torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                        
                        running_error, running_time = 0, 0
                        
                        for _ in range(iterations):
                            start = time.time()
                            output = torch.median(input=input, dim=dim, keepdim=keepdim) 
                            end = time.time()
                            current_error, _ = error(output.values, baseline_output.values, tolerances=None) 
                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                            running_error += current_error
                            running_time += end - start
                            if not any([torch.equal(output.values.to(cpu), unique_output.values.to(cpu)) and torch.equal(output.indices.to(cpu), unique_output.indices.to(cpu)) for unique_output in unique_outputs]):
                                unique_outputs.add(output)    
                        
                        # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                        average_relative_error = running_error / iterations
                        average_runtime = running_time / iterations
                        print(f"[INFO]... Average Relative Error: {average_relative_error}")
                        print(f"[INFO]... Average Runtime: {average_runtime}")
                        print(f"[INFO]... Unique Outputs for {iterations+1} iterations: {len(unique_outputs)}")
                        print(f"[INFO]... Unique Outputs / iterations: {len(unique_outputs)/(iterations+1)}")
                       
                        results_df = results_df._append({ 
                            'input_dimension': input_dimension, 
                            'dtype': str(dtype),
                            'dim': dim,
                            'keepdim': keepdim,
                            'average_relative_error': average_relative_error.detach().cpu().numpy(),
                            'average_runtime': average_runtime,
                            'unique_outputs': len(unique_outputs),
                            'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                        }, ignore_index=True)
                          
                        del unique_outputs
                                    
                del input
                torch.cuda.empty_cache()
        
        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv")  
                
    if op.__name__ == "cumsum":
            
        results_df = pd.DataFrame(columns=['input_dimensions', 'dim', 'dtype', 'keepdim', 'unique_outputs', 'unique_outputs_per_iteration'])
        for input_dimension in dimensions["input_dimensions"]:
            for dtype in dimensions["dtype"]:
                input = data_dist(torch.zeros(input_dimension)).to(dtype).to(device)
                for dim in dimensions["dim"]:
                    if dim >= len(input_dimension): break # this generalises to arbitrary dimensional input
                    print(f"[INFO] input_dimension:{input_dimension}\t dtype:{dtype}\t dim:{dim}\t")  
                    
                    try:
                        torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                        baseline_output = torch.cumsum(input=input, dim=dim) 
                    except:
                        torch.use_deterministic_algorithms(mode=False)
                        baseline_output = torch.cumsum(input=input, dim=dim) 
                        
                    unique_outputs = set({baseline_output})
                    # set torch environment variables to deterministic/non-deterministic 
                    if deterministic:
                        torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                    else:
                        torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                    
                    running_error, running_time = 0, 0
                    
                    for _ in range(iterations):
                        start = time.time()
                        output = torch.cumsum(input=input, dim=dim) 
                        end = time.time()
                        current_error, _ = error(output, baseline_output, tolerances=None) 
                        # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                        running_error += current_error
                        running_time += end - start
                        if not any([torch.equal(output.to(cpu), unique_output.to(cpu)) for unique_output in unique_outputs]):
                            unique_outputs.add(output)    
                    
                    # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                    average_relative_error = running_error / iterations
                    average_runtime = running_time / iterations
                    print(f"[INFO]... Average Relative Error: {average_relative_error}")
                    print(f"[INFO]... Average Runtime: {average_runtime}")
                    print(f"[INFO]... Unique Outputs for {iterations+1} iterations: {len(unique_outputs)}")
                    print(f"[INFO]... Unique Outputs / iterations: {len(unique_outputs)/(iterations+1)}")
                   
                    results_df = results_df._append({ 
                        'input_dimension': input_dimension, 
                        'dtype': str(dtype),
                        'dim': dim,
                        'average_relative_error': average_relative_error.detach().cpu().numpy(),
                        'average_runtime': average_runtime,
                        'unique_outputs': len(unique_outputs),
                        'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                    }, ignore_index=True)
                     
                    del unique_outputs
                                    
                del input
                torch.cuda.empty_cache()
                
        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv") 
                                
    if op.__name__ == "AvgPool3d": # TODO: check
        
        
        results_df = pd.DataFrame(columns=['batch_size', 'dim', 'dtype', 'kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'average_relative_error', 'average_runtime', 'unique_outputs', 'unique_outputs_per_iteration'])
        if not isinstance(hyperparameters, AvgPoolParams):
            return TypeError(f"Op is a AvgPool, hyperparameter AvgPoolParams dataclass expected but the specified hyperparameter dataclass is of type: {type(hyperparameters)}")
       
        for dtype in dimensions["dtype"]:
            for kernel_size in dimensions["kernel_size"]: # loop over hyperparameters before looping over dimensions, as you need to instantiate the module with hyperparameters first
                for stride in dimensions["stride"]:
                    for padding in dimensions["padding"]:
                        for ceil_mode in dimensions["ceil_mode"]:
                            for count_include_pad in dimensions["count_include_pad"]: 
                                
                                # instantiate module with the specified hyperparameters
                                hyperparameters.kernel_size = kernel_size
                                hyperparameters.stride = stride
                                hyperparameters.padding = padding
                                hyperparameters.ceil_mode = ceil_mode
                                hyperparameters.count_include_pad = count_include_pad
                                hyperparameters.dtype = dtype
                                module = op(**asdict(hyperparameters)) # initialise module with **kwargs by unpacking dict generated from Params dataclass
                                initialise_weights(module, weight_dist) # initialise weights with specified weight init scheme from torch.nn.init
                                module.to(dtype) # set dtype
                                module.to(device) # send model to device
                                
                                for batch in dimensions["batch_size"]:
                                    for dim in dimensions["dim"]:
                                        print(f"[INFO] batch_size: {batch}\t dim:{dim}\t kernel_size:{kernel_size}\t stride:{stride}\t padding:{padding}\t ceil_mode:{ceil_mode}\t count_include_pad:{count_include_pad}\t dtype:{dtype}\t")
                                        torch.manual_seed(42) # TODO: check if this seeds the weight init schemes
                                        input = data_dist(torch.zeros((batch, *dim))).to(dtype).to(device) # TODO: can we just initialise the data without using torch.zeros as an init buffer
                                        input.requires_grad = True
                                        # generate a deterministic baseline, if possible
                                        try:
                                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                                            output = module(input) # initial baseline output
                                            grad = torch.ones(output.shape).to(dtype).to(device)
                                            output.backward(grad)
                                            baseline_output = input.grad
                                        except:
                                            torch.use_deterministic_algorithms(mode=False)
                                            output = module(input) # initial baseline output
                                            grad = torch.ones(output.shape).to(dtype).to(device)
                                            output.backward(grad)
                                            baseline_output = input.grad
                                        
                                        unique_outputs = set({baseline_output})
                                        # set torch environment variables to deterministic/non-deterministic 
                                        if deterministic:
                                            torch.backends.cudnn.benchmark = False # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
                                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                                        else:
                                            torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                                        
                                        running_error, running_time = 0, 0
                                        
                                        for _ in range(iterations):
                                            torch.use_deterministic_algorithms(mode=False)
                                            output = module(input) # initial baseline output
                                            grad = torch.ones(output.shape).to(dtype).to(device)
                                            start = time.time()
                                            output.backward(grad)
                                            end = time.time()
                                            output = input.grad
                                            # print(output)
                                            current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                                            running_error += current_error
                                            running_time += end - start
                                            if not any([torch.equal(output.to(cpu), unique_output.to(cpu)) for unique_output in unique_outputs]):
                                                unique_outputs.add(output)    
                                        
                                        # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                                        average_relative_error = running_error / iterations
                                        average_runtime = running_time / iterations
                                        print(f"[INFO]... Average Relative Error: {average_relative_error}")
                                        print(f"[INFO]... Average Runtime: {average_runtime}")
                                        
                                        results_df = results_df._append({
                                            'batch_size': batch, 
                                            'dim': dim,
                                            'dtype': str(dtype),
                                            'kernel_size': kernel_size,
                                            'stride': stride,
                                            'padding': padding,
                                            "count_include_pad": count_include_pad,
                                            "ceil_mode": ceil_mode,
                                            'average_relative_error': average_relative_error.detach().numpy(),
                                            'average_runtime': average_runtime,
                                            'unique_outputs': len(unique_outputs),
                                            'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                                        }, ignore_index=True) 
                                         
                                # memory management to prevent OOM issues
                                del module
                                del input
                                torch.cuda.empty_cache()
                                
        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv") 
                                
    if op.__name__ == "AdaptiveAvgPool2d" or op.__name__ == "AdaptiveMaxPool2d" or op.__name__ == "AdaptiveAvgPool3d": # TODO: check
        
        results_df = pd.DataFrame(columns=['batch_size', 'dim', 'dtype', 'reduction_ratio', 'output_size', 'unique_outputs', 'unique_outputs_per_iteration'])
        if not isinstance(hyperparameters, AdaptiveAvgPoolParams):
            return TypeError(f"Op is a AdaptiveAvgPool or AdaptiveMaxPool, hyperparameter AdaptiveAvgPoolParams dataclass expected but the specified hyperparameter dataclass is of type: {type(hyperparameters)}")
       
        for dtype in dimensions["dtype"]:
            for batch in dimensions["batch_size"]:
                for dim in dimensions["dim"]:
                    for reduction_ratio in dimensions["reduction_ratio"]:
                        if op.__name__ == "AdaptiveAvgPool3d":    
                            output_size = tuple((int(dim[-3] * reduction_ratio), int(dim[-2] * reduction_ratio), int(dim[-1] * reduction_ratio)))
                        else:
                            output_size = tuple((int(dim[-2] * reduction_ratio), int(dim[-1] * reduction_ratio)))
                          
                        # instantiate module with the specified hyperparameters
                        hyperparameters.output_size = output_size
                        module = op(**asdict(hyperparameters)) # initialise module with **kwargs by unpacking dict generated from Params dataclass
                        initialise_weights(module, weight_dist) # initialise weights with specified weight init scheme from torch.nn.init
                        module.to(dtype) # set dtype
                        module.to(device) # send model to device
                                
                        print(f"[INFO] batch_size: {batch}\t dim:{dim}\t dtype:{dtype}\t reduction_ratio:{reduction_ratio}\t output_size:{output_size}")
                        torch.manual_seed(42) # TODO: check if this seeds the weight init schemes
                        input = data_dist(torch.zeros((batch, *dim))).to(dtype).to(device) # TODO: can we just initialise the data without using torch.zeros as an init buffer
                        input.requires_grad = True
                        # generate a deterministic baseline, if possible
                        try:
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                            output = module(input) # initial baseline output
                            grad = torch.ones(output.shape).to(dtype).to(device)
                            output.backward(grad)
                            baseline_output = input.grad
                        except:
                            torch.use_deterministic_algorithms(mode=False)
                            output = module(input) # initial baseline output
                            grad = torch.ones(output.shape).to(dtype).to(device)
                            output.backward(grad)
                            baseline_output = input.grad 
                        unique_outputs = set({baseline_output})
                        # set torch environment variables to deterministic/non-deterministic 
                        if deterministic:
                            torch.backends.cudnn.benchmark = False # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                        else:
                            torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                        
                        running_error, running_time = 0, 0
                        
                        for _ in range(iterations):
                            torch.use_deterministic_algorithms(mode=False)
                            output = module(input) # initial baseline output
                            grad = torch.ones(output.shape).to(dtype).to(device)
                            start = time.time()
                            output.backward(grad)
                            end = time.time()
                            output = input.grad
                            # print(output)
                            current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                            running_error += current_error
                            running_time += end - start
                            if not any([torch.equal(output.to(cpu), unique_output.to(cpu)) for unique_output in unique_outputs]):
                                unique_outputs.add(output)    
                        
                        # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                        average_relative_error = running_error / iterations
                        average_runtime = running_time / iterations
                        print(f"[INFO]... Average Relative Error: {average_relative_error}")
                        print(f"[INFO]... Average Runtime: {average_runtime}")
                        
                        results_df = results_df._append({ 
                            'batch_size': batch, 
                            'dtype': str(dtype),
                            'dim': dim,
                            'reduction_ratio': reduction_ratio,
                            "output_size": output_size,
                            'average_relative_error': average_relative_error.detach().numpy(),
                            'average_runtime': average_runtime,
                            'unique_outputs': len(unique_outputs),
                            'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                            
                        }, ignore_index=True)
                        
                        # memory management to prevent OOM issues
                        del module
                        del input
                        torch.cuda.empty_cache()
        
        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv") 

    if op.__name__ == "MaxPool3d": # TODO: check
        
        results_df = pd.DataFrame(columns=['batch_size', 'dim', 'dtype', 'kernel_size', 'stride', 'padding', 'dilation', 'ceil_mode', 'average_relative_error', 'average_runtime', 'unique_outputs', 'unique_outputs_per_iteration'])
        
        if not isinstance(hyperparameters, MaxPoolParams):
            return TypeError(f"Op is a MaxPool, hyperparameter MaxPoolParams dataclass expected but the specified hyperparameter dataclass is of type: {type(hyperparameters)}")
       
        for dtype in dimensions["dtype"]:
            for kernel_size in dimensions["kernel_size"]: # loop over hyperparameters before looping over dimensions, as you need to instantiate the module with hyperparameters first
                for stride in dimensions["stride"]:
                    for padding in dimensions["padding"]:
                        for ceil_mode in dimensions["ceil_mode"]:
                            for dilation in dimensions["dilation"]: 
                                
                                # instantiate module with the specified hyperparameters
                                hyperparameters.kernel_size = kernel_size
                                hyperparameters.stride = stride
                                hyperparameters.padding = padding
                                hyperparameters.ceil_mode = ceil_mode
                                hyperparameters.dilation = dilation
                                hyperparameters.dtype = dtype
                                module = op(**asdict(hyperparameters)) # initialise module with **kwargs by unpacking dict generated from Params dataclass
                                initialise_weights(module, weight_dist) # initialise weights with specified weight init scheme from torch.nn.init
                                module.to(dtype) # set dtype
                                module.to(device) # send model to device
                                
                                for batch in dimensions["batch_size"]:
                                    for dim in dimensions["dim"]:
                                        print(f"[INFO] batch_size: {batch}\t dim:{dim}\t kernel_size:{kernel_size}\t stride:{stride}\t padding:{padding}\t ceil_mode:{ceil_mode}\t dilation:{dilation}\t dtype:{dtype}\t")
                                        torch.manual_seed(42) # TODO: check if this seeds the weight init schemes
                                        input = data_dist(torch.zeros((batch, *dim))).to(dtype).to(device) # TODO: can we just initialise the data without using torch.zeros as an init buffer
                                        input.requires_grad = True
                                        # generate a deterministic baseline, if possible
                                        try:
                                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithmsa
                                            output = module(input) # initial baseline output
                                            grad = torch.ones(output.shape).to(dtype).to(device)
                                            output.backward(grad)
                                            baseline_output = input.grad
                                        except:
                                            torch.use_deterministic_algorithms(mode=False)
                                            output = module(input) # initial baseline output
                                            grad = torch.ones(output.shape).to(dtype).to(device)
                                            output.backward(grad)
                                            baseline_output = input.grad 
                                        unique_outputs = set({baseline_output})
                                        # set torch environment variables to deterministic/non-deterministic 
                                        if deterministic:
                                            torch.backends.cudnn.benchmark = False # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
                                            torch.use_deterministic_algorithms(mode=True) # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
                                        else:
                                            torch.use_deterministic_algorithms(mode=False) # set determinism mode, if available
                                        
                                        running_error, running_time = 0, 0
                                        
                                        for _ in range(iterations):
                                            torch.use_deterministic_algorithms(mode=False)
                                            output = module(input) # initial baseline output
                                            grad = torch.ones(output.shape).to(dtype).to(device)
                                            start = time.time()
                                            output.backward(grad)
                                            end = time.time()
                                            output = input.grad
                                            # print(output)
                                            current_error, _ = error(output.to(cpu), baseline_output.to(cpu), tolerances=None) 
                                            # TODO: are we passing by reference in the error function? Looks like we are copying tensors, since we get an OOM error in the error function call when not sending to cpu
                                            running_error += current_error
                                            running_time += end - start
                                            if not any([torch.equal(output.to(cpu), unique_output.to(cpu)) for unique_output in unique_outputs]):
                                                unique_outputs.add(output)     
                                        # timing and benchmarking deviations TODO: compare S_nd to S_d like @mathieut
                                        average_relative_error = running_error / iterations
                                        average_runtime = running_time / iterations
                                        print(f"[INFO]... Average Relative Error: {average_relative_error}")
                                        print(f"[INFO]... Average Runtime: {average_runtime}")
                                       
                                        results_df = results_df._append({
                                            'batch_size': batch, 
                                            'dim': dim,
                                            'dtype': str(dtype),
                                            'kernel_size': kernel_size,
                                            'stride': stride,
                                            'padding': padding,
                                            "dilation": dilation,
                                            "ceil_mode": ceil_mode,
                                            'average_relative_error': average_relative_error.detach().numpy(),
                                            'average_runtime': average_runtime,
                                            'unique_outputs': len(unique_outputs),
                                            'unique_outputs_per_iteration': len(unique_outputs)/(iterations+1),
                                        }, ignore_index=True)  
        
                                # memory management to prevent OOM issues
                                del module
                                del input
                                torch.cuda.empty_cache()

        results_df.to_csv(f"data/{op.__name__}.csv", index=False)
        print(f"{op.__name__} Benchmark saved to {op.__name__}.csv")
                             
if __name__ == "__main__":
    
    deterministic = False
    device = gpu
    iterations = 1000
    
    # conv2d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(3, 256, 256), (3, 512, 512), (3, 1024, 1024)]),
    #     "kernel_size": list([(3, 3), (5, 5), (7, 7)]),
    #     "stride": list([1, 3, 5]),
    #     "padding": list([0, 1]),
    #     "dilation": list([1, 2, 5]),
    #     "groups": list([1, 2, 3]),
    #     "dtype": list([torch.float32]), # TODO: check why relative error is NaN with fp16
    # }
   
    # print("=========================== Conv2d benchmark ======================================") 
    # benchmark(
    #     op = nn.Conv2d,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = conv2d_dimensions,
    #     hyperparameters = ConvParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = None,
    #     iterations = iterations,
    # )
    
    # conv3d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(3, 64, 64, 64), ]),
    #     "kernel_size": list([(3, 3, 3), (5, 5, 5), (7, 7, 7)]),
    #     "stride": list([1, 3, 5]),
    #     "padding": list([0, 1]),
    #     "dilation": list([1, 2, 5]),
    #     "groups": list([1, 2, 3]),
    #     "dtype": list([torch.float32]), # TODO: check why relative error is NaN with fp16
    # }
   
    # print("=========================== Conv3d benchmark ======================================") 
    # benchmark(
    #     op = nn.Conv3d,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = conv3d_dimensions,
    #     hyperparameters = ConvParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = None,
    #     iterations = iterations,
    # )
    
    # conv1d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(3, 256), (3, 512), (3, 1024)]),
    #     "kernel_size": list([(3,), (5,), (7,)]),
    #     "stride": list([1, 3, 5]),
    #     "padding": list([0, 1]),
    #     "dilation": list([1, 2, 5]),
    #     "groups": list([1, 2, 3]),
    #     "dtype": list([torch.float32]), # TODO: check why relative error is NaN with fp16
    # }
   
    # print("=========================== Conv1d benchmark ======================================") 
    # benchmark(
    #     op = nn.Conv1d,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = conv1d_dimensions,
    #     hyperparameters = ConvParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = None,
    #     iterations = iterations,
    # )

    # convtranspose2d_dimensions = {
    #     "batch_size": list([1, ]),
    #     "dim": list([(1, 100, 100), (1, 200, 200), (1, 300, 300), (1, 400, 400), (1, 500, 500), (1, 600, 600), (1, 700, 700), (1, 800, 800), (1, 900, 900), (1, 1000, 1000)]),
    #     "kernel_size": list([(3, 3), (5, 5), (7, 7), (9, 9), (11, 11)]),
    #     "stride": list([1,]),
    #     "padding": list([0,]),
    #     "output_padding": list([0,]),
    #     "dilation": list([2, ]),
    #     "groups": list([1, ]),
    #     "dtype": list([torch.float32]), # TODO: check why relative error is NaN with fp16
    # }
   
    # print("=========================== ConvTranspose2d benchmark ======================================") 
    # benchmark(
    #     op = nn.ConvTranspose2d,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = convtranspose2d_dimensions,
    #     hyperparameters = ConvTransposeParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = None,
    #     iterations = iterations,
    # )
    
    # convtranspose3d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(3, 64, 64, 64)]),
    #     "kernel_size": list([(3, 3, 3), (5, 5, 5), (7, 7, 7)]),
    #     "stride": list([1, 3, 5]),
    #     "padding": list([0, 1]),
    #     "output_padding": list([0, 1]),
    #     "dilation": list([1, 2, 5]),
    #     "groups": list([1, 2, 3]),
    #     "dtype": list([torch.float32]), # TODO: check why relative error is NaN with fp16
    # }
   
    # print("=========================== ConvTranspose3d benchmark ======================================") 
    # benchmark(
    #     op = nn.ConvTranspose3d,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = convtranspose3d_dimensions,
    #     hyperparameters = ConvTransposeParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = None,
    #     iterations = iterations,
    # )
    
    # convtranspose1d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(3, 256), (3, 512), (3, 1024,)]),
    #     "kernel_size": list([(3,), (5,), (7,)]),
    #     "stride": list([1, 3, 5]),
    #     "padding": list([0, 1]),
    #     "output_padding": list([0, 1]),
    #     "dilation": list([1, 2, 5]),
    #     "groups": list([1, 2, 3]),
    #     "dtype": list([torch.float32]), # TODO: check why relative error is NaN with fp16
    # }
   
    # print("=========================== ConvTranspose1d benchmark ======================================") 
    # benchmark(
    #     op = nn.ConvTranspose1d,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = convtranspose1d_dimensions,
    #     hyperparameters = ConvTransposeParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = None,
    #     iterations = iterations,
    # )
        
    scatter_reduce_dimensions = {
        "input_dimensions": list([(1_000, ), (2_000, ), (3_000, ), (4_000, ), (5_000, ), (6_000, ), (7_000, ), (8_000, ), (9_000, ), (10_000, ), (11_000, ), (12_000, ), (13_000, ), (14_000, ), (15_000, )]), # 2d reductions
        "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]),
        "dim": list([0, 1]),
        "reduce": list(["sum", "mean"]),
        "include_self": list([True]),
        "dtype" : list([torch.float32]), # TODO: why do we get NaNs with float16?
    }
    
    print("=========================== Scatter Reduce Benchmark ======================================") 
    benchmark(
        op = torch.scatter_reduce,
        weight_dist = nn.init.normal_,
        data_dist = nn.init.normal_, 
        dimensions = scatter_reduce_dimensions,
        hyperparameters = None,
        device = device,
        deterministic = deterministic,
        autograd = False,
        dtype = torch.float32,
        iterations = iterations,
    )
    
    # scatter_dimensions = {
    #     "input_dimensions": list([(10_000, ), (1_000, 1_000)]), # 2d reductions
    #     "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]),
    #     "dim": list([0, 1]),
    #     "reduce": list(["add", "multiply"]),
    #     "dtype" : list([torch.float32]), # TODO: why do we get NaNs with float16?
    # }
    
    # print("=========================== Scatter Benchmark ======================================") 
    # benchmark(
    #     op = torch.scatter,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = scatter_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32,
    #     iterations = iterations,
    # )
    
    # index_select_dimensions = {
    #     "input_dimensions": list([(1_000_000, ), (10_000, 10_000)]), # 2d reductions
    #     "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]),
    #     "dim": list([0, 1]),
    #     "dtype" : list([torch.float32]), # TODO: why do we get NaNs with float16?
    # }
    
    # print("=========================== Index Select Benchmark ======================================") 
    # benchmark(
    #     op = torch.index_select,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = index_select_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # bmm_dimensions = {
    #     "batch": list([1, 32, 64]), # 2d reductions
    #     "n": list((512, 1024)),
    #     "m": list((512, 1024)),
    #     "p": list((512, 1024)),
    #     "dtype" : list([torch.float32]), # TODO: why do we get NaNs with float16?
    # }
    
    # print("=========================== Batch MatMul (torch.bmm) Benchmark ======================================") 
    # benchmark(
    #     op = torch.bmm,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = bmm_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )

    # gather_dimensions = {
    #     "input_dimensions": list([(32, 32), (1_000_000, ), (1000, 1000)]), # 2d reductions
    #     "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]),
    #     "dim": list([0, 1]),
    #     "dtype" : list([torch.float32]), # TODO: why do we get NaNs with float16?
    # }
    
    # print("=========================== Gather Benchmark ======================================") 
    # benchmark(
    #     op = torch.gather,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = gather_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )

    index_add_dimensions = {
        "input_dimensions": list([(10, 10), (20, 20), (30, 30), (40, 40), (50, 50), (60, 60), (70, 70), (80, 80), (90, 90), (100, 100),]),
                                #   (200, 200), (300, 300), (400, 400), (500, 500), (600, 600), (700, 700), (800, 800), (900, 900), (1000, 1000)]), # 2d reductions
        "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]),
        "dim": list([0,]),
        "dtype" : list([torch.float32]), # TODO: why do we get NaNs with float16?
    }
    
    print("=========================== Index Add Benchmark ======================================") 
    benchmark(
        op = torch.index_add,
        weight_dist = nn.init.normal_,
        data_dist = nn.init.normal_, 
        dimensions = index_add_dimensions,
        hyperparameters = None,
        device = device,
        deterministic = deterministic,
        autograd = False,
        dtype = torch.float32, # TODO: why do we get NaNs with float16?
        iterations = iterations,
    )
    
    # index_copy_dimensions = {
    #     "input_dimensions": list([(32, 32), (1000, 1000)]), # 2d reductions
    #     "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]),
    #     "dim": list([0,]),
    #     "dtype" : list([torch.float32]), # TODO: why do we get NaNs with float16?
    # }
    
    # print("=========================== Index Copy Benchmark ======================================") 
    # benchmark(
    #     op = torch.index_copy,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = index_copy_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )


    # index_put_dimensions = {
    #     "input_dimensions": list([(32, 32), (1000, 1000)]), # 2d reductions
    #     "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]),
    #     "accumulate": list([True, False]),
    #     "dtype" : list([torch.float32]), # TODO: why do we get NaNs with float16?
    # }
    
    # print("=========================== Index Put Benchmark ======================================") 
    # benchmark(
    #     op = torch.index_put,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = index_put_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )

    # histc_dimensions = {
    #     "input_dimensions": list([10_000, 100_000]),
    #     "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]),
    #     "dtype": list([torch.float32]), # TODO: why do we get NaNs with float16?
    #     "min": list([0, 0.1, 0.2]),
    #     "max": list([0, 0.1, 0.2]),
    # }

    # print("=========================== Histc Benchmark ======================================") 
    # benchmark(
    #     op = torch.histc,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = histc_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # bincount_dimensions = {
    #     "input_dimensions": list([10_000, 100_000]),
    #     "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]),
    #     "dtype": list([torch.int64]), # TODO: notimplemented for cuda for float?
    # }

    # print("=========================== BinCount Benchmark ======================================") 
    # benchmark(
    #     op = torch.bincount,
    #     weight_dist = None, 
    #     data_dist = None, 
    #     dimensions = bincount_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # kthvalue_dimensions = {
    #     "input_dimensions": list([(10_000,), (100_000, )]),
    #     "k": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]),
    #     "dim": list([0, 1]),
    #     "keepdim": list([False, True]),
    #     "dtype": list([torch.int64]), # TODO: notimplemented for cuda for float?
    # }
    
    # print("=========================== Kthvalue Benchmark ======================================") 
    # benchmark(
    #     op = torch.kthvalue,
    #     weight_dist = nn.init.normal_, 
    #     data_dist = nn.init.normal_, 
    #     dimensions = kthvalue_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # median_dimensions = {
    #     "input_dimensions": list([(10_000,), (1000, 1000)]),
    #     "dim": list([0, 1]),
    #     "keepdim": list([False, True]),
    #     "dtype": list([torch.int64]), # TODO: notimplemented for cuda for float?
    # }
    
    # print("=========================== Median Benchmark ======================================") 
    # benchmark(
    #     op = torch.median,
    #     weight_dist = nn.init.normal_, 
    #     data_dist = nn.init.normal_, 
    #     dimensions = median_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # cumsum_dimensions = {
    #     "input_dimensions": list([(10_000,), (1000, 1000), (1_000_000, )]),
    #     "dim": list([0, 1]),
    #     "dtype": list([torch.float16, torch.float32, torch.int32]), # TODO: notimplemented for cuda for float?
    # }
    
    # print("=========================== CumSum Benchmark ======================================") 
    # benchmark(
    #     op = torch.cumsum,
    #     weight_dist = nn.init.normal_, 
    #     data_dist = nn.init.normal_, 
    #     dimensions = cumsum_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
        
    # avgpool3d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(3, 64, 64, 64)]),
    #     "kernel_size": list([(3, 3, 3), (5, 5, 5), (7, 7, 7)]),
    #     "stride": list([1, 3, 5]),
    #     "padding": list([0, 1]),
    #     "ceil_mode": list([True, False]),
    #     "count_include_pad": list([True, False]),
    #     "dtype": list([torch.float32]), # TODO: check why relative error is NaN with fp16
    # }
   
    # print("=========================== AvgPool3d benchmark ======================================") 
    # benchmark(
    #     op = nn.AvgPool3d,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = avgpool3d_dimensions,
    #     hyperparameters = AvgPoolParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = None,
    #     iterations = iterations,
    # )
    
    # adaptiveavgpool2d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(3, 256, 256), (3, 512, 512)]),
    #     "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
    #     "dtype": list([torch.float32]), # TODO: check why relative error is NaN with fp16
    # }
   
    # print("=========================== AdaptiveAvgPool2d benchmark ======================================") 
    # benchmark(
    #     op = nn.AdaptiveAvgPool2d,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = adaptiveavgpool2d_dimensions,
    #     hyperparameters = AdaptiveAvgPoolParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = None,
    #     iterations = iterations,
    # )
    
    # adaptiveavgpool3d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(3, 64, 64, 64)]),
    #     "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
    #     "dtype": list([torch.float32]), # TODO: check why relative error is NaN with fp16
    # }
   
    # print("=========================== AdaptiveAvgPool3d benchmark ======================================") 
    # benchmark(
    #     op = nn.AdaptiveAvgPool3d,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = adaptiveavgpool3d_dimensions,
    #     hyperparameters = AdaptiveAvgPoolParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = None,
    #     iterations = iterations,
    # )
    
    # maxpool3d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(3, 64, 64, 64),]),
    #     "kernel_size": list([(3, 3, 3), (5, 5, 5), (7, 7, 7)]),
    #     "stride": list([1, 3, 5]),
    #     "padding": list([0, 1]),
    #     "dilation": list([1, 2]),
    #     "ceil_mode": list([True, False]),
    #     "dtype": list([torch.float32]), # TODO: check why relative error is NaN with fp16
    # }
   
    # print("=========================== MaxPool3d benchmark ======================================") 
    # benchmark(
    #     op = nn.MaxPool3d,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = maxpool3d_dimensions,
    #     hyperparameters = MaxPoolParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = None,
    #     iterations = iterations,
    # ) 
    
    
    # adaptivemaxpool2d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(3, 256, 256), (3, 512, 512)]),
    #     "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
    #     "dtype": list([torch.float32]), # TODO: check why relative error is NaN with fp16
    # }
   
    # print("=========================== AdaptiveAvgPool2d benchmark ======================================") 
    # benchmark(
    #     op = nn.AdaptiveAvgPool2d,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = adaptivemaxpool2d_dimensions,
    #     hyperparameters = AdaptiveAvgPoolParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = None,
    #     iterations = iterations,
    # )
    
    # fractionalmaxpool2d_dimensions = {
    #     "kernel_size": list([3, 5, 7]),
    #     "batch_size": list([1, 2]),
    #     "dim": list([(3, 256, 256), (3, 512, 512)]),
    #     "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
    #     "dtype": list([torch.float32]), # TODO: check why relative error is NaN with fp16
    # }
   
    # print("=========================== FractionalMaxPool2d benchmark ======================================") 
    # benchmark(
    #     op = nn.FractionalMaxPool2d,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = fractionalmaxpool2d_dimensions,
    #     hyperparameters = FractionalMaxPoolParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = None,
    #     iterations = iterations,
    # )
    
    # fractionalmaxpool3d_dimensions = {
    #     "kernel_size": list([3, 5, 7]),
    #     "batch_size": list([1, 2]),
    #     "dim": list([(3, 64, 64, 64)]),
    #     "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
    #     "dtype": list([torch.float32]), # TODO: check why relative error is NaN with fp16
    # }
   
    # print("=========================== FractionalMaxPool3d benchmark ======================================") 
    # benchmark(
    #     op = nn.FractionalMaxPool3d,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = fractionalmaxpool3d_dimensions,
    #     hyperparameters = FractionalMaxPoolParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = None,
    #     iterations = iterations,
    # )
    
    # maxunpool1d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(1000, 1000), (1_000_000, )]),
    #     "stride": list([1]),
    #     "kernel_size": list([3, 5]),
    #     "padding": list([0, 1]),
    #     "dtype": list([torch.float32]), # TODO: notimplemented for cuda for float?
    # }
    
    # print("=========================== MaxUnpool1d Benchmark ======================================") 
    # benchmark(
    #     op = nn.MaxUnpool1d,
    #     weight_dist = nn.init.normal_, 
    #     data_dist = nn.init.normal_, 
    #     dimensions = maxunpool1d_dimensions,
    #     hyperparameters = MaxUnpoolParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # maxunpool2d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(1000, 1000),]),
    #     "stride": list([1]),
    #     "kernel_size": list([3, 5]),
    #     "padding": list([0, 1]),
    #     "dtype": list([torch.float32]), # TODO: notimplemented for cuda for float?
    # }
    
    # print("=========================== MaxUnpool2d Benchmark ======================================") 
    # benchmark(
    #     op = nn.MaxUnpool2d,
    #     weight_dist = nn.init.normal_, 
    #     data_dist = nn.init.normal_, 
    #     dimensions = maxunpool2d_dimensions,
    #     hyperparameters = MaxUnpoolParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # maxunpool3d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(64, 64, 64),]),
    #     "stride": list([1]),
    #     "kernel_size": list([3, 5]),
    #     "padding": list([0, 1]),
    #     "dtype": list([torch.float32]), # TODO: notimplemented for cuda for float?
    # }
    
    # print("=========================== MaxUnpool3d Benchmark ======================================") 
    # benchmark(
    #     op = nn.MaxUnpool3d,
    #     weight_dist = nn.init.normal_, 
    #     data_dist = nn.init.normal_, 
    #     dimensions = maxunpool3d_dimensions,
    #     hyperparameters = MaxUnpoolParams(),
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # reflectionpad1d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(1000, 1000), (1_000_000, )]),
    #     "pad": list([0, 1, 2]),
    #     "dtype": list([torch.float32]), # TODO: notimplemented for cuda for float?
    # }
    
    # print("=========================== ReflectionPad1d Benchmark ======================================") 
    # benchmark(
    #     op = nn.ReflectionPad1d,
    #     weight_dist = nn.init.normal_, 
    #     data_dist = nn.init.normal_, 
    #     dimensions = reflectionpad1d_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # reflectionpad2d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(1000, 1000)]),
    #     "pad": list([0, 1, 2]),
    #     "dtype": list([torch.float32]), # TODO: notimplemented for cuda for float?
    # }
    
    # print("=========================== ReflectionPad2d Benchmark ======================================") 
    # benchmark(
    #     op = nn.ReflectionPad2d,
    #     weight_dist = nn.init.normal_, 
    #     data_dist = nn.init.normal_, 
    #     dimensions = reflectionpad2d_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # reflectionpad3d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(64, 64, 64)]),
    #     "pad": list([0, 1, 2]),
    #     "dtype": list([torch.float32]), # TODO: notimplemented for cuda for float?
    # }
    
    # print("=========================== ReflectionPad3d Benchmark ======================================") 
    # benchmark(
    #     op = nn.ReflectionPad3d,
    #     weight_dist = nn.init.normal_, 
    #     data_dist = nn.init.normal_, 
    #     dimensions = reflectionpad3d_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # replicationpad1d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(1000, 1000), (1_000_000, )]),
    #     "pad": list([0, 1, 2]),
    #     "dtype": list([torch.float32]), # TODO: notimplemented for cuda for float?
    # }
    
    # print("=========================== ReplicationPad1d Benchmark ======================================") 
    # benchmark(
    #     op = nn.ReplicationPad1d,
    #     weight_dist = nn.init.normal_, 
    #     data_dist = nn.init.normal_, 
    #     dimensions = replicationpad1d_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # replicationpad2d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(1000, 1000),]),
    #     "pad": list([0, 1, 2]),
    #     "dtype": list([torch.float32]), # TODO: notimplemented for cuda for float?
    # }
    
    # print("=========================== ReplicationPad2d Benchmark ======================================") 
    # benchmark(
    #     op = nn.ReplicationPad2d,
    #     weight_dist = nn.init.normal_, 
    #     data_dist = nn.init.normal_, 
    #     dimensions = replicationpad2d_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # replicationpad3d_dimensions = {
    #     "batch_size": list([1, 2]),
    #     "dim": list([(64, 64, 64),]),
    #     "pad": list([0, 1, 2]),
    #     "dtype": list([torch.float32]), # TODO: notimplemented for cuda for float?
    # }
    
    # print("=========================== ReplicationPad3d Benchmark ======================================") 
    # benchmark(
    #     op = nn.ReplicationPad3d,
    #     weight_dist = nn.init.normal_, 
    #     data_dist = nn.init.normal_, 
    #     dimensions = replicationpad3d_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # put_dimensions = {
    #     "input_dimensions": list([(32, 32), (1000, 1000)]), # 2d reductions
    #     "reduction_ratio": list([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]),
    #     "accumulate": list([True, False]),
    #     "dtype" : list([torch.float32]), # TODO: why do we get NaNs with float16?
    # }
    
    # print("=========================== Put Benchmark ======================================") 
    # benchmark(
    #     op = torch.index_put,
    #     weight_dist = nn.init.normal_,
    #     data_dist = nn.init.normal_, 
    #     dimensions = put_dimensions,
    #     hyperparameters = None,
    #     device = device,
    #     deterministic = deterministic,
    #     autograd = False,
    #     dtype = torch.float32, # TODO: why do we get NaNs with float16?
    #     iterations = iterations,
    # )
    
    # # pass
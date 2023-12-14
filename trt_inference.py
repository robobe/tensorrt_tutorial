from pytorch_model import preprocess_image, postprocess
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt


ONNX_FILE_PATH = "/home/user/tmp/onnx_demo/resnet50.onnx"
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_serialized_network(network, config)
    context = engine.create_execution_context()
    print("Completed creating Engine")
    return engine, context

def parse_or_load(self):
		logger= trt.Logger(trt.Logger.INFO)
		#we want to show logs of type info and above (warnings, errors)
		
		if os.path.exists(self.enginepath):
			logger.log(trt.Logger.INFO, 'Found pre-existing engine file')
			with open(self.enginepath, 'rb') as f:
				rt=trt.Runtime(logger)
				engine=rt.deserialize_cuda_engine(f.read())

			return engine, logger
 
		else: #parse and build if no engine found
			with trt.Builder(logger) as builder:
				builder.max_batch_size=self.max_batch_size
				#setting max_batch_size isn't strictly necessary in this case
				#since the onnx file already has that info, but its a good practice
				
				network_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
				
				#since the onnx file was exported with an explicit batch dim,
				#we need to tell this to the builder. We do that with EXPLICIT_BATCH flag
				
				with builder.create_network(network_flag) as net:
				
					with trt.OnnxParser(net, logger) as p:
						#create onnx parser which will read onnx file and
						#populate the network object `net`					
						with open(self.onnxpath, 'rb') as f:
							if not p.parse(f.read()):
								for err in range(p.num_errors):
									print(p.get_error(err))
							else:
								logger.log(trt.Logger.INFO, 'Onnx file parsed successfully')

						net.get_input(0).dtype=trt.DataType.HALF
						net.get_output(0).dtype=trt.DataType.HALF
						#we set the inputs and outputs to be float16 type to enable
						#maximum fp16 acceleration. Also helps for int8
						
						config=builder.create_builder_config()
						#we specify all the important parameters like precision, 
						#device type, fallback in config object

						config.max_workspace_size = self.maxworkspace

						if self.precision_str in ['FP16', 'INT8']:
							config.flags = ((1<<self.precision)|(1<<self.allowGPUFallback))
							config.DLA_core=self.dla_core
						# DLA core (0 or 1 for Jetson AGX/NX/Orin) to be used must be 
						# specified at engine build time. An engine built for DLA0 will 
						# not work on DLA1. As such, to use two DLA engines simultaneously, 
						# we must build two different engines.

						config.default_device_type=self.device
						#if device is set to GPU, DLA_core has no effect

						config.profiling_verbosity = trt.ProfilingVerbosity.VERBOSE
						#building with verbose profiling helps debug the engine if there are
						#errors in inference output. Does not impact throughput.

						if self.precision_str=='INT8' and self.calibrator is None:
							logger.log(trt.Logger.ERROR, 'Please provide calibrator')
							#can't proceed without a calibrator
							quit()
						elif self.precision_str=='INT8' and self.calibrator is not None:
							config.int8_calibrator=self.calibrator
							logger.log(trt.Logger.INFO, 'Using INT8 calibrator provided by user')

						logger.log(trt.Logger.INFO, 'Checking if network is supported...')
						
						if builder.is_network_supported(net, config):
							logger.log(trt.Logger.INFO, 'Network is supported')
							#tensorRT engine can be built only if all ops in network are supported.
							#If ops are not supported, build will fail. In this case, consider using 
							#torch-tensorrt integration. We might do a blog post on this in the future.
						else:
							logger.log(trt.Logger.ERROR, 'Network contains operations that are not supported by TensorRT')
							logger.log(trt.Logger.ERROR, 'QUITTING because network is not supported')
							quit()

						if self.device==trt.DeviceType.DLA:
							dla_supported=0
							logger.log(trt.Logger.INFO, 'Number of layers in network: {}'.format(net.num_layers))
							for idx in range(net.num_layers):
								if config.can_run_on_DLA(net.get_layer(idx)):
									dla_supported+=1

							logger.log(trt.Logger.INFO, f'{dla_supported} of {net.num_layers} layers are supported on DLA')

						logger.log(trt.Logger.INFO, 'Building inference engine...')
						engine=builder.build_engine(net, config)
						#this will take some time

						logger.log(trt.Logger.INFO, 'Inference engine built successfully')

						with open(self.enginepath, 'wb') as s:
							s.write(engine.serialize())
						logger.log(trt.Logger.INFO, f'Inference engine saved to {self.enginepath}')
						
		return engine, logger



def main():
    # initialize TensorRT engine and parse ONNX model
    engine, context = build_engine(ONNX_FILE_PATH)
    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()


    # preprocess input data
    host_input = np.array(preprocess_image("turkish_coffee.jpg").numpy(), dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)

    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    # postprocess results
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    postprocess(output_data)


if __name__ == '__main__':
    main()

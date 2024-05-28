import time
from sd_evaluate import calculate_clip_score, np2image
from torch.profiler import profile, record_function, ProfilerActivity

# def profiling_unet_trace(pipe, tdtype):
#     # , 
#     with profile(activities=[ProfilerActivity.CPU]) as prof:
#         pipe.unet(sample=torch.randn(2, 4, 96, 96).to(memory_format=torch.channels_last).to(dtype=tdtype), timestep=torch.tensor(921), encoder_hidden_states=torch.randn(2, 77, 1024).to(dtype=tdtype))
#     prof.export_chrome_trace("trace_unet_ipex.json")

def elapsed_time(pipeline, prompt, height, width, stm=None, nb_pass=10, num_inference_steps=20, saved_img_fn=None, profiling_pt=False):
    for i in range(nb_pass):
        start = time.time()
        # Profiling
        if profiling_pt:
            # ProfilerActivity.CUDA
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                images = pipeline(prompt, num_inference_steps=num_inference_steps, output_type="np", height=height, width=width).images
            prof.export_chrome_trace("trace_unet_ipex_"+str(i)+".json")
        # Not profiling.
        else:
            images = pipeline(prompt, num_inference_steps=num_inference_steps, output_type="np", height=height, width=width).images
        end = time.time()
        if stm is not None:
            stm.add_tm(end - start)
    
    output_shape=images[0].shape
    print("output shape=", output_shape)

    if stm is not None:
        stm.add_comments(f"Output Shape:{output_shape}")
    if saved_img_fn is not None:
        img = np2image(images[0])
        img.save(saved_img_fn)

    return stm
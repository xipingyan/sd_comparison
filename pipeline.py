import time
from sd_evaluate import calculate_clip_score, np2image

def elapsed_time(pipeline, prompt, height, width, stm=None, nb_pass=10, num_inference_steps=20, saved_img_fn=None):
    for _ in range(nb_pass):
        start = time.time()
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
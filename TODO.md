goals:
make MotionGPT3 stable for varieties of motions of different length
hook HTTPS API into prompt generation / eval pipeline
evaluate the things

current state:
motion over ~200 frames is unstable
no pipeline is built

notes:
ryan - we are using a single reletively small VAE, see configs/mldvae.yaml, forcing longer clips into only 256 dimensions is probably unreasonable (emoji generator used single 512 dim encoder...). also seems that humanml3d (training dataset) is mostly limited to very short clips < 200 frames. this is all to say, the model is trained to generate very quick motion so the breakdown with increased size makes perfect sense. we may want to invest in finding a better training dataset and experimenting with training... perhaps becomes a cool reason to get into NJIT's HPC :)
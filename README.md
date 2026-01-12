# channel grouped iterative temporal frequency convolutional recurrent network (CGTFCRN)
Implemented based on PyTorch, using a channel grouped temporal frequency convolutional recurrent network for speech denoising.  
With 15.8K parameters
## test
We have stored several noisy speech audio samples from different scenarios in the test_wavs directory, and the pre-trained model is placed in the checkpoints directory. You can run the infer.py script to perform testing.
## train
The training script is located in the train folder. The data used for training should be processed according to the format specified in traindataset.

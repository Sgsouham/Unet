# Unet- Convolutional Networks for Biomedical Image Segmentation 

## Idea

This is a paper implementation of the UNET model in Pytorch


![image](https://tuatini.me/content/images/2017/09/u-net-architecture.png)

As you can see from the above figure, the model looks like a U shape (hence the name). 
Most important thing to notice here is that the Network is a simple Encoder decoder Network but with skip connections as well. Also the block of 3 CONV->ReLU always gets repeated which is accompanied by a Max-Pooling layer.

This is more of a vanilla implementation with no fancy stuff. I just implemented this in here to try out with pytorch. 


If you want more information on the orginal paper, you can definitely check it out ***[here](https://arxiv.org/abs/1505.04597)***.

## Ways in which this can be used
  - You can use this as a starting stage while working on a Unet project
  - You can refer to this if you want to work with implementing an paper (Always start off with the architecture. It's the easiest).

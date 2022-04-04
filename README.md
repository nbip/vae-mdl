# vae-mdl
VAE with mixture of discretized logistics

## Resources
[transformers from scratch](https://towardsdatascience.com/7-things-you-didnt-know-about-the-transformer-a70d93ced6b2)

https://github.com/addtt/ladder-vae-pytorch
https://github.com/addtt/ladder-vae-pytorch/blob/master/lib/likelihoods.py#L117

## Notes
In a decoder, if you are using `stride=2` from your second-to-last to last layer, each set of 2x2 pixel is a linear projection of some single pixel in the previous layer. Maybe that is not desirable. Instead have one more conv layer in the end with `stride=1`  

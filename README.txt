First, download the VGGFace2 Dataset. It is used for training the GAN in the first step.
It can be found on "https://drive.google.com/file/d/1tz-QYIJVc-8b2jXnJTpQ8Bc8o0gRMdZW/view?usp=sharing".
The data should be unzipped and placed as followed:
| - Project2
	| - code
	| - examples
	| - model
	| - result
	| - data
		| - train
			| - n000006
			| - n000007
			| - ...

Second, train the GAN:
	python train_GAN.py
It takes a very long time, ~48 hours.

Third, train the encoder:
	python train_enc.py
It takes ~2 hours.

I am sure you don't do the above step, so I put the pretrained models in model/*.pth.

After training the encoder and decoder, we can use them to accomplish several tasks:
1. face comparison:
	python example.py -example comparison
	# A face comparison among me and my parents.
2. face clustering:
	python example.py -example cluster
	# I use 10 faces of Zhou Shen and 10 faces of Shan Yichun, who are my favorite singers, as the example.
3. tsne visualization:
	python example.py -example tsne
	# Use tsne to visualize the 20 faces described in 2. The results will be outputted to '../examples/tsne/tsne.jpg'
4. Face transfer:
	python example.py -example transfer
	# Show a graduate shift from one face to another. Here I use an example of myself, from 2013 to 2017. The results will be outputted to '../examples/transfer/results'
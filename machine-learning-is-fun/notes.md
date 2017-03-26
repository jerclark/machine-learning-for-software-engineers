## 1 - machine learning is fun.
Read machine learning is fun. It was nice that many of the concepts were review or at least familiar. There's a good visualization of what the error function looks like for linear regression, and how taking the derivative of a sample on the error function will determine which way to "go down the bowl" - i.e. gradient descent.


## 2 - Super mario maker levels
* Incredible example of the 'ernest hemmingway' bot author
* Really helpful to see how determining the 'style' of earnest hemmingway would translate to determining the 'style' of super mario levels.
* Never really thought to pull 'level' data from the memory of an old cartridge. 
* Interesting to think about mapping each 'square to a character'
* Nice visualizations of to think of each vertical column in the level graphic as a sample. Then using a neural network

## 3 - Deep learining and convolutional neural networks
* Good overview of image recognition through handwritting recognition of letter 8
* Map the pixels to numbers
* Initial issue is that the training data only knows perfectly centered 8s
* Brute force number 1
    * Sliding window (recenter the test area in a grid until the 8 is found)
* Brute force number 2 - more data a deep neural network
    * Create more training data with scripts that position the 8 all over the image background
    * Use more intermediate nodes to get the best results
    [<img src="https://cdn-images-1.medium.com/max/800/1*wfmpsoFqWKC7VadjTJxwnQ.png">](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721#.m3kruuxe7)
* CONVOLUTION
    * We want to achieve "translation invariance" - a model that understands an 8 is an 8, no matter where in the image.
    * Think of how humans could easily identify an 8 regardless of postion or background
    * The Steps
        * Break the image into tiny overlapping tiles
        * Put each tile into a small (shallow) neural network. And use the same weights in the neural network for each tile.
        * Save the result of each tile into a new array
        * Downsample using "max pooling" - separate the new, smaller array into two by two grids, and take the biggest number of each. This will reduce the array to 25% its original size.
        * Then pump the downsampled/maxpooled data into a 'fully connected' neural network, and seek out an '8'.
        
* Building a bird classifier
   * Find data sets (the article mentions some bird/non-bird image data sets)
   * Author suggests in ML, having more data is almost always more important than good algorithms!
   * Great sample code for DNN with bird image detection!
* Results
    * Review of 'accuracy' terms
        * True positives: We thought 'its a bird' and it was
        * True negatives: We thought 'not a bird' and it wasn't
        * False Positives: We thought 'its a bird' and it wasn't
        * False Negatives: We thought 'its not a bird' and it WAS (this can be bad...think of cancer - "hey, you don't have it"...but you do)
        * These can be distilled into:
            * Accuracy: True positives  / All positive guesses (how often do we guess right!)
            * Recall: True Positives / All birds in data set (How many did we find!)
            * Hi accuracy/Lower recall == Hey, we don't find em all, but when we identify one, we're pretty sure it's right!
            
        
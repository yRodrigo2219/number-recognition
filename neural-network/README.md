#### The JavaScript version DOESN'T work properly

- It's at least 1000x slower than the numpy version.
- It gets out of memory really easily (can't load more than like, 100 images for training).
- It tries to mimic too much of the python syntactic sugars.
- Doesn't use performatic libs.

##### Improvements

- Using a lib that uses ndarray and more performatic linear algebra operations ([numjs](https://github.com/nicolaspanel/numjs), [stdlib](https://github.com/stdlib-js/stdlib), etc). This will, hopefully, have a huge impact.
- In the book, he mentions some improvements that can be made in the python code, don't know how much impact that can make.

##### Disclaimers

- Python3 code converted by [Michał Dobrzanski](https://github.com/MichalDanielDobrzanski/DeepLearningPython)
- Original version made by [Michael Nielsen](https://github.com/mnielsen/neural-networks-and-deep-learning)
- [Michael Nielsen's book](http://neuralnetworksanddeeplearning.com/chap1.html)
- [MNIST data set](http://yann.lecun.com/exdb/mnist/)

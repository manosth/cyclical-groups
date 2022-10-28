# Learning unfolded networks with a cyclic group stucture
Official code repository for "[Learning unfolded networks with a cyclic group stucture](https://manosth.github.io/files/papers/TheodosisBa_UnfoldedCyclical_NeurReps22.pdf)", an extended abstract accepted at the NeurIPS Workshop for [Symmetry and Geometry in Neural Representations](neurreps.org).

It consists of an unfolded architecture for learning networks whose layer weights have a cyclic group structure. The default architecture is an unfolded network with 4 layers where the majority of the filters are generated as 60 degree rotations of the layer's "basis" weights. Trianing should take about 30-60 mins to run on a GeForce 1080 and reaches ~75% accuracy on CIFAR10.

The data are expected to be (or are downloaded) in the user's home folder and to train type `python3 main.py`.

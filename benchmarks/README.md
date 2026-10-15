# Benchmarks

This directory contains the code to train and evaluate models for two components that `pipeline` relies on:

* `detection` contains code and model for evaluating face detection on cartoon images and animated films
* `recognition` contains code and models for evaluationg face recognition on that same data

Both train entirely from public datasets (iCartoonFace, WIDER Face) with
download scripts included, and each is evaluated on two sets: the public
iCartoonFace test splits (those numbers are reproducible from a clone with no
special access) and a 100-film annotation set, which is derived from the
film corpus and is not publicly releasable.

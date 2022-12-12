# Hydrological_modelling [Work In Progress]
This is a project I am working for the course hydrological modelling of TU Delft. The goal of the course is to make a rainfall runoff model. This can be done in many ways however I decided to take a little bit of an unconvential route. Inspired by the people from https://neuralhydrology.github.io/ some deep learning models will get the job done.
Please don't use any of these code for real projects other than playing around as this is mostly a hobby project, morphed in such way that I can hand it in for a course and get credits for it.

The idea is to use the GRDC discharge data together with other remote sensing data products and see if it is possible to train a LSTM neural network to predict a single gauge because this training process can be done quite fast (my laptop does not resemble a super computer... yet). If I have more time left, I want to see if it is possible to train a network that is able to predict multiple gauges. After analysing the GRDC data, it seemed that the stations located in Namibia and South Africa have many data points, which makes it interesting to train the models on.

Note work is still in progress and therefore the project might seem a little bit cluttered.
